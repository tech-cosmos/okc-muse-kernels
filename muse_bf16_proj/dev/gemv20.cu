// gemv19: gemv18's TMA-ring streamer over lossless 12-bit weights (m = 16/32, same _pack18 stage layout), FP32 output
// only, with load balancing. Per-SM HBM bandwidth varies by ~10% (gemv18 blocks ended 21-27 us at pooler m16 with
// equal static ranges), so: each block first streams a static stream-K range of the first U - TAIL stages, then grabs
// tail items of IS stages from a global counter until none are left. The producer warp decides everything: it tags
// every ring slot with its stage u (+ x buffer, + "run ends here" flag); consumers flush partials at run ends; x is
// double-buffered and loaded by the producer when a stage needs another k-chunk. All partials go to the FP32 output
// by red.global.add (finisher warp); a tiny zero kernel (PDL primary) zeroes the output and the counter first.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <cudaTypedefs.h>
namespace g20 {
constexpr int ESC = 8, NWC = 8;
#ifdef G20_TRACE
__device__ unsigned long long trace[132 * 64];
__device__ __forceinline__ unsigned long long gt() { unsigned long long v; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(v)); return v; }
#define TR(slot) trace[blockIdx.x * 64 + (slot)] = gt()
#else
#define TR(slot)
#endif
__device__ __forceinline__ unsigned prmt(unsigned a, unsigned sel) {
    unsigned r;
    asm("prmt.b32 %0, %1, 0, %2;" : "=r"(r) : "r"(a), "r"(sel));
    return r;
}
__device__ __forceinline__ unsigned lop_aob(unsigned a, unsigned b, unsigned c) {  // s34: (a & b) | c
    unsigned r;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}
template <int P>
__device__ __forceinline__ unsigned expo(unsigned nw) {
    const unsigned s = P == 0 ? nw << 7 : P == 1 ? nw << 3 : P == 2 ? nw >> 1 : nw >> 5;
    const unsigned z = gridDim.y >> 16, m2 = 0x07800780u >> z, cc = 0x38003800u >> z;  // s34 DEC4: opaque (== the immediates)
    unsigned r;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(s), "r"(m2), "r"(cc));  // (s & m2) | cc
    return r;
}
template <int P>  // s35 DEC3: one prmt (exponent constant 0x38 from a register) + one lop3 per BF16 pair
__device__ __forceinline__ unsigned decw(unsigned sw, unsigned nw, unsigned sel4, unsigned sel3) {
#ifdef PJ_DEC4  // s34 layout (sign in the byte, code nibbles): for A/B builds only
    return lop_aob(prmt(sw, sel4), 0x807f807fu, expo<P>(nw));
#else
    const unsigned sh = P == 0 ? nw : P == 1 ? nw << 4 : P == 2 ? nw << 8 : __funnelshift_r(nw, nw, 4);
    const unsigned k38 = 0x38383838u >> (gridDim.y >> 16);  // opaque register (== the constant)
    unsigned p;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(p) : "r"(sw), "r"(k38), "r"(sel3));
    return lop_aob(sh, 0x87008700u, p);
#endif
}
__device__ __forceinline__ uint4 dec8(unsigned s0, unsigned s1, unsigned nw) {
    uint4 r;
    r.x = decw<0>(s0, nw, 0x9180u, 0x4140u);
    r.y = decw<1>(s0, nw, 0xB3A2u, 0x4342u);
    r.z = decw<2>(s1, nw, 0x9180u, 0x4140u);
    r.w = decw<3>(s1, nw, 0xB3A2u, 0x4342u);
    return r;
}
__device__ __forceinline__ void mma(float* c, const uint4& a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm4(unsigned addr, unsigned& d0, unsigned& d1, unsigned& d2, unsigned& d3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3) : "r"(addr));
}
__device__ __forceinline__ void bar_init(unsigned b, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(count));
}
__device__ __forceinline__ void bar_expect(unsigned b, int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bar_arrive(unsigned b) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory");
}
__device__ __forceinline__ void bar_wait(unsigned b, unsigned parity) {
    asm volatile("{\n.reg .pred p;\nW20_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W20_%=;\n}"
                 :: "r"(b), "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk(unsigned dst, const void* src, int bytes, unsigned b) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(dst), "l"(src), "r"(bytes), "r"(b) : "memory");
}
#ifdef G20_NOWG
__device__ __forceinline__ void wg_fence() {}
__device__ __forceinline__ void wg_commit() {}
template <int K> __device__ __forceinline__ void wg_wait() {}
#else
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int K>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(K) : "memory"); }
#endif
__device__ __forceinline__ void fence_r(unsigned& r) { asm volatile("" : "+r"(r)::"memory"); }
__device__ __forceinline__ void fence_f(float& r) { asm volatile("" : "+f"(r)::"memory"); }
__device__ __forceinline__ void tma2d(unsigned dst, const CUtensorMap* map, int c0, int c1, unsigned bar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                 :: "r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
// K-major SW128 B descriptor (8 token rows x 128 B core groups, 1024 B apart)
__device__ __forceinline__ unsigned long long sw128_desc(unsigned a) {
    return (unsigned long long)((a & 0x3FFFF) >> 4) | (1ull << 16) | ((unsigned long long)(1024 >> 4) << 32) | (1ull << 62);
}
template <int M> __device__ __forceinline__ void wgmma(float* d, const unsigned* a, unsigned long long desc);
template <> __device__ __forceinline__ void wgmma<16>(float* d, const unsigned* a, unsigned long long desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, %12, 1, 1, 1, 0;"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
template <> __device__ __forceinline__ void wgmma<32>(float* d, const unsigned* a, unsigned long long desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %20, 1, 1, 1, 0;"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
                   "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
// one ring stage of a consumer warp: AP atoms of its 16-row group, decoded into af (one A fragment per k16), then
// 2 * AP wgmma m64nMk16 of its warpgroup (4 row groups = 64 rows) over the x chunk (SW128, 64 k per M * 128 B block)
template <int M, int AP>
__device__ __forceinline__ void stage(float* acc, unsigned (&af)[2 * AP][4], unsigned sb, unsigned nb, unsigned xbuf, int kk0) {
#pragma unroll
    for (int j = 0; j < AP; ++j) {
        uint4 sv;
        uint2 nv;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(sv.x), "=r"(sv.y), "=r"(sv.z), "=r"(sv.w) : "r"(sb + j * 512));
        asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(nv.x), "=r"(nv.y) : "r"(nb + j * 256));
#ifdef G20_NODEC
        const uint4 a0 = make_uint4(sv.x, sv.y, nv.x, nv.y), a1 = make_uint4(sv.z, sv.w, nv.y ^ sv.x, nv.x ^ sv.y);
#else
        const uint4 a0 = dec8(sv.x, sv.y, nv.x), a1 = dec8(sv.z, sv.w, nv.y);
#endif
        af[2 * j][0] = a0.x; af[2 * j][1] = a0.y; af[2 * j][2] = a0.z; af[2 * j][3] = a0.w;
        af[2 * j + 1][0] = a1.x; af[2 * j + 1][1] = a1.y; af[2 * j + 1][2] = a1.z; af[2 * j + 1][3] = a1.w;
    }
    wg_fence();
#pragma unroll
    for (int j = 0; j < 2 * AP; ++j) {
        const int kk = kk0 + j * 16;
#ifdef G20_NOMMA
        acc[j & 7] += __uint_as_float(af[j][0] ^ af[j][1] ^ af[j][2] ^ af[j][3] ^ (unsigned)kk ^ xbuf);
#else
        wgmma<M>(acc, af[j], sw128_desc(xbuf + (kk >> 6) * M * 128 + (kk & 63) * 2));
#endif
    }
    wg_commit();
}
// M tokens (16/32), KC k per chunk, AP atoms (32 k) per warp per stage, NS ring stages, IS stages per tail item.
// out FP32 [M, n] (zeroed by the primary or in-kernel), ctr int. tail: stages left for dynamic items.
// smem (1024-aligned): ring [NS][SB] | x [XB][KC/64][M][128 B] SW128 | pbuf [2][128 rows][M] f32 | barriers | tags
// consumers: 8 warps = 2 warpgroups; warp w = 16-row group w of the 128-row tile (same packing as gemv19, rg 8)
#ifndef G20_NPB
#define G20_NPB 2
#endif
constexpr int NPB = G20_NPB;  // partial buffers (1: less smem, for 2 blocks per SM)
template <int M, int KC, int AP, int NS, int IS, int PRE, int XB, int TS, int RG, int RPW>
#ifndef G20_MINB
#define G20_MINB 1
#endif
__global__ void __launch_bounds__(320, G20_MINB) gemm(const unsigned char* __restrict__ WP, const float2* __restrict__ H,
                                               float scale, const __grid_constant__ CUtensorMap xmap,
                                               const __nv_bfloat16* __restrict__ x,
                                               float* __restrict__ out, int* __restrict__ ctr, int n, int k, int tail, int npre) {
    static_assert(TS == 1 && RG == 8 && RPW == 1, "g20: rg 8, ts 1, rpw 1 only");
    constexpr int NCW = 8, NT = M / 8, NA = KC / 32, SPK = NA / AP, SB = NCW * AP * 768, XBF = KC * 2 * M, PBF = NCW * 16 * M;
    static_assert(NA % AP == 0 && SPK % IS == 0 && KC % 64 == 0 && SB % 1024 == 0, "config");
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    unsigned char* smem = smem_raw + ((1024 - ((unsigned)__cvta_generic_to_shared(smem_raw) & 1023)) & 1023);
    const unsigned ring = (unsigned)__cvta_generic_to_shared(smem), xs = ring + NS * SB;
    float* pbuf = (float*)(smem + NS * SB + XB * XBF);
    const unsigned bars = (unsigned)__cvta_generic_to_shared(pbuf + NPB * PBF);
    int* tag = (int*)(pbuf + NPB * PBF) + 2 * (2 * NS + 8);  // per slot: u | xb << 29 | last x use << 30 | run end << 31 (-1: no more work)
    int* ptag = tag + NS;                                 // per pbuf: tile (-1: consumers done)
    const unsigned xfull = bars + 16 * NS, xempty = xfull + 16, pfull = xfull + 32, pempty = xfull + 48;
    const int t = threadIdx.x, lane = t & 31, w = t >> 5, g = lane >> 2, c = lane & 3;
    const int nrg = n / 16, ntr = nrg / NCW, S = k / KC, KA = k / 32;
    const int U = S * ntr * SPK, US = U - tail;
    if (t == 0) {
        for (int i = 0; i < NS; ++i) { bar_init(bars + 8 * i, 1); bar_init(bars + 8 * (NS + i), NCW); }
        for (int i = 0; i < 2; ++i) {
            bar_init(xfull + 8 * i, 1); bar_init(xempty + 8 * i, NCW); bar_init(pfull + 8 * i, NCW); bar_init(pempty + 8 * i, 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (t == 0) TR(0);
    if (w == NCW) {  // producer: one lane walks the static range, then the tail items
        if (lane == 0) {
            asm volatile("prefetch.tensormap [%0];" :: "l"(&xmap) : "memory");
            int ra = (int)((long)US * blockIdx.x / gridDim.x), rb = (int)((long)US * (blockIdx.x + 1) / gridDim.x);
            int xks[2] = {-1, -1}, xuse[2] = {0, 0}, xb = 1, i = 0;
            if (PRE && npre > 0 && ra < rb) {  // stream the first stages' W while the previous kernel runs
                const int ks0 = ra / (ntr * SPK);
                for (; i < npre && ra < rb && ra / (ntr * SPK) == ks0; ++i, ++ra) {
                    const int st = ra % SPK;
                    const bool xl = ra + 1 == rb || (ra + 1) / (ntr * SPK) != ks0;
                    tag[i] = ra | (xl ? 1 << 30 : 0) | (ra + 1 == rb || st + 1 == SPK ? 1 << 31 : 0);  // x buffer 0
                    bar_expect(bars + 8 * i, SB);
                    bulk(ring + i * SB, WP + (long)ra * SB, SB, bars + 8 * i);
                }
                asm volatile("griddepcontrol.wait;" ::: "memory");
                xb = 0; xks[0] = ks0; xuse[0] = 1;
                bar_expect(xfull, XBF);
                for (int b = 0; b < KC / 64; ++b) tma2d(xs + b * M * 128, &xmap, ks0 * KC + b * 64, 0, xfull);
            } else {
                asm volatile("griddepcontrol.wait;" ::: "memory");  // x, out, ctr ready
            }
            for (;;) {
                if (ra >= rb) {  // next run: a tail item
                    if (tail == 0) break;
                    const int it = atomicAdd(ctr, 1);
                    if (it * IS >= tail) break;
                    ra = US + it * IS; rb = ra + IS;
                }
                for (int u = ra; u < rb; ++u, ++i) {
                    const int ks = u / (ntr * SPK), st = u % SPK;
                    if (ks != xks[xb]) {  // next x buffer, load chunk ks (consumers release a buffer after its last use)
                        xb = XB == 2 ? xb ^ 1 : 0;
                        if (xuse[xb]) bar_wait(xempty + 8 * xb, (xuse[xb] - 1) & 1);
                        ++xuse[xb];
                        xks[xb] = ks;
                        bar_expect(xfull + 8 * xb, XBF);
                        for (int b = 0; b < KC / 64; ++b) tma2d(xs + xb * XBF + b * M * 128, &xmap, ks * KC + b * 64, 0, xfull + 8 * xb);
                    }
                    const int slot = i % NS;
                    if (i >= NS) bar_wait(bars + 8 * (NS + slot), ((i / NS) - 1) & 1);
                    const bool end = u + 1 == rb || st + 1 == SPK;
                    const bool xl = u + 1 == rb || (u + 1) / (ntr * SPK) != ks;
                    if (xl) xks[xb] = -1;
                    tag[slot] = u | xb << 29 | (xl ? 1 << 30 : 0) | (end ? 1 << 31 : 0);
                    if (i < 24) TR(40 + i);
                    const unsigned full = bars + 8 * slot;
#ifdef G20_NOW
                    bar_arrive(full);
#else
                    bar_expect(full, SB);
                    bulk(ring + slot * SB, WP + (long)u * SB, SB, full);
#endif
                }
                ra = rb;
            }
            const int slot = i % NS;  // end marker
            if (i >= NS) bar_wait(bars + 8 * (NS + slot), ((i / NS) - 1) & 1);
            tag[slot] = -1;
            bar_arrive(bars + 8 * slot);
            TR(1);
        }
        return;
    }
    if (w == NCW + 1) {  // finisher: pbuf [128 rows][M] -> out [M][n] by red.global.add.v4 (lane = token, 4 rows)
        asm volatile("griddepcontrol.wait;" ::: "memory");
        unsigned target = 0;
        if (PRE == 2) {  // zero a slice of out, then arrive on the epoch counter ctr[1] (+gridDim.x per launch, never reset)
            float4* o4 = (float4*)out;
            for (int i = blockIdx.x * 32 + lane; i < M * n / 4; i += gridDim.x * 32) o4[i] = make_float4(0.f, 0.f, 0.f, 0.f);
            __syncwarp();
            if (lane == 0) {
                __threadfence();
                const unsigned old = atomicAdd((unsigned*)ctr + 1, 1u);
                target = old - old % gridDim.x + gridDim.x;
            }
        }
        for (int q = 0;; ++q) {
            const int pb = q % NPB;
            bar_wait(pfull + 8 * pb, (q / NPB) & 1);
            const int tr = ptag[pb];
            if (tr < 0) break;
            if (PRE == 2 && q == 0) {
                if (lane == 0) {
                    unsigned v;
                    do { asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(v) : "l"(ctr + 1) : "memory"); } while ((int)(v - target) < 0);
                }
                __syncwarp();
            }
            const float* src = pbuf + pb * PBF;
            float* wt = out + (long)tr * NCW * 16;
#pragma unroll 4
            for (int i = lane; i < PBF / 4; i += 32) {
                const int tok = i % M, r4 = (i / M) * 4;
                const float a0 = src[r4 * M + tok] * scale, a1 = src[(r4 + 1) * M + tok] * scale;
                const float a2 = src[(r4 + 2) * M + tok] * scale, a3 = src[(r4 + 3) * M + tok] * scale;
#ifndef G20_NORED
                asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(wt + (long)tok * n + r4), "f"(a0), "f"(a1), "f"(a2), "f"(a3) : "memory");
#else
                if (a0 == 1.2345f) wt[0] = a1 + a2 + a3;
#endif
            }
            __syncwarp();
            if (lane == 0) bar_arrive(pempty + 8 * pb);
        }
        return;
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    float acc[NT * 4];  // wgmma m64nM accumulator: per n8 tile jj, [4] = mma.sync C layout (rows g, g + 8; tokens 2c, 2c + 1)
#pragma unroll
    for (int e = 0; e < NT * 4; ++e) acc[e] = 0.f;
    unsigned afA[2 * AP][4], afB[2 * AP][4];
    const int wr = w;  // the warp's row group in the tile
    int q = 0, st0 = 0, prev_end = 1, prev_xl = 1;
    unsigned xph[2] = {0, 0};
    float4 hd0, hd1;
    for (int i = 0;; ++i) {
        const int slot = i % NS;
        bar_wait(bars + 8 * slot, (i / NS) & 1);
        const int tg = tag[slot];
        if (t == 0 && i < 24) TR(16 + i);
        if (tg == -1) break;
        const int u = tg & 0x1FFFFFFF, xb = (tg >> 29) & 1;
        const bool end = tg < 0, xl = (tg >> 30) & 1;
        const int ks = u / (ntr * SPK), r = u % (ntr * SPK), tr = r / SPK, st = r % SPK;
        if (prev_xl) {  // first stage of an x use: wait for it
            bar_wait(xfull + 8 * xb, xph[xb]);
            xph[xb] ^= 1;
        }
        prev_xl = xl;
        if (prev_end) {  // run start: its escape headers (rows g, g + 8 of the warp's row group; lane c: slots 2c, 2c + 1)
            st0 = st;
            const float2* hp0 = H + (long)((tr * NCW + wr) * 16 + g) * ESC;
            hd0 = __ldg((const float4*)hp0 + c);
            hd1 = __ldg((const float4*)(hp0 + 8 * ESC) + c);
        }
        prev_end = end;
        const unsigned xbuf = xs + xb * XBF;
        const unsigned sb = ring + slot * SB + wr * AP * 512 + lane * 16, nb = ring + slot * SB + NCW * AP * 512 + wr * AP * 256 + lane * 8;
        // double-buffered A fragments: this stage's group is in flight while the next stage decodes into the other set
#ifdef G20_SB
        stage<M, AP>(acc, afA, sb, nb, xbuf, st * AP * 32);
#else
        if (i & 1) stage<M, AP>(acc, afB, sb, nb, xbuf, st * AP * 32);
        else stage<M, AP>(acc, afA, sb, nb, xbuf, st * AP * 32);
#endif
        __syncwarp();
        if (lane == 0) bar_arrive(bars + 8 * (NS + slot));  // W of this slot is in registers (decoded)
        if (!end) {
#ifdef G20_SB
            wg_wait<0>();
#pragma unroll
            for (int j = 0; j < 2 * AP; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) fence_r(afA[j][e]);
            continue;
#endif
            wg_wait<1>();  // the previous stage's group is done: its A set may be overwritten next
#pragma unroll
            for (int j = 0; j < 2 * AP; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) { if (i & 1) fence_r(afA[j][e]); else fence_r(afB[j][e]); }
            continue;
        }
        wg_wait<0>();
#pragma unroll
        for (int j = 0; j < 2 * AP; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) { fence_r(afA[j][e]); fence_r(afB[j][e]); }
#pragma unroll
        for (int e = 0; e < NT * 4; ++e) fence_f(acc[e]);
        // run end: rows wr*16 + g (+8), tokens jj*8 + 2c (+1), k in [klo, khi)
        const int nat = (st - st0 + 1) * AP;
        const int klo = ks * KC + st0 * AP * 32, khi = ks * KC + (st + 1) * AP * 32;
        float v[NT][4];
#pragma unroll
        for (int jj = 0; jj < NT; ++jj)
#pragma unroll
            for (int e = 0; e < 4; ++e) { v[jj][e] = acc[jj * 4 + e]; acc[jj * 4 + e] = 0.f; }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const float4 hq = h ? hd1 : hd0;
#pragma unroll
            for (int e2 = 0; e2 < ESC; ++e2) {  // empty slot = (col 0, 2^-15): skipped (real escapes: |value| < 2^-15)
                const int src = (lane & ~3) | (e2 >> 1);
                const float hx = __shfl_sync(0xffffffffu, (e2 & 1) ? hq.z : hq.x, src);
                const float hy = __shfl_sync(0xffffffffu, (e2 & 1) ? hq.w : hq.y, src);
                const int col = __float_as_int(hx);
#ifdef G20_NOFIX
                if (hy == 1.2345f)
#else
                if (hy != 0x1p-15f && col >= klo && col < khi) 
#endif
                {  // two fmas: exact when the dot holds only this weight (probe rows)
                    const int kk = col - ks * KC;
                    const unsigned char* xc = smem + NS * SB + xb * XBF + (kk >> 6) * M * 128 + (kk & 7) * 2;
                    const int ch = (kk & 63) >> 3;
#pragma unroll
                    for (int jj = 0; jj < NT; ++jj)
#pragma unroll
                        for (int b = 0; b < 2; ++b) {
                            const int tok = jj * 8 + 2 * c + b;
                            const float xv = __bfloat162float(*(const __nv_bfloat16*)(xc + tok * 128 + ((ch ^ (tok & 7)) << 4)));
                            v[jj][2 * h + b] = fmaf(xv, hy, fmaf(xv, -0x1p-15f, v[jj][2 * h + b]));
                        }
                }
            }
        }
        if (xl) {  // done with this x buffer (wgmma groups done, fixups above were its last reads)
            __syncwarp();
            if (lane == 0) bar_arrive(xempty + 8 * xb);
        }
        if (nat == KA && PRE != 2) {  // the whole dot: plain stores
#pragma unroll
            for (int jj = 0; jj < NT; ++jj)
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    out[(long)(jj * 8 + 2 * c + (e & 1)) * n + (tr * NCW + wr) * 16 + g + 8 * (e >> 1)] = v[jj][e] * scale;
            continue;
        }
        const int pb = q % NPB;  // hand the partials to the finisher: pbuf[pb][row group * 16 + row16][tok]
        if (q >= NPB) bar_wait(pempty + 8 * pb, ((q / NPB) - 1) & 1);
        {
            float* pw = pbuf + pb * PBF + (wr * 16 + g) * M + 2 * c;
#pragma unroll
            for (int jj = 0; jj < NT; ++jj)
#pragma unroll
                for (int h = 0; h < 2; ++h) *(float2*)(pw + 8 * h * M + jj * 8) = make_float2(v[jj][2 * h], v[jj][2 * h + 1]);
        }
        if (t == 0) ptag[pb] = tr;
        __syncwarp();
        if (lane == 0) bar_arrive(pfull + 8 * pb);
        ++q;
    }
    wg_wait<0>();
    {  // done: tell the finisher
        const int pb = q % NPB;
        if (q >= NPB) bar_wait(pempty + 8 * pb, ((q / NPB) - 1) & 1);
        if (t == 0) ptag[pb] = -1;
        __syncwarp();
        if (lane == 0) bar_arrive(pfull + 8 * pb);
    }
    if (t == 0) TR(4);
}
// PDL primary: lets the gemm launch at once, zeroes its output and counter (the gemm waits before touching them)
__global__ void zero(float4* __restrict__ p, int n4, int* __restrict__ ctr) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if (blockIdx.x == 0 && threadIdx.x == 0) *ctr = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) p[i] = make_float4(0.f, 0.f, 0.f, 0.f);
}
// test predecessor: lets its dependent launch, then writes x = src (n8 uint4) and spins ~ns_ nanoseconds
__global__ void pred(const uint4* __restrict__ src, uint4* __restrict__ dst, int n8, int ns_) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n8; i += gridDim.x * blockDim.x) dst[i] = src[i];
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t0));
    do { asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t1)); } while (t1 - t0 < (unsigned long long)ns_);
}
// diagnostics: a bare TMA ring stream (no math) of `total` bytes, NS stages of SB bytes, one 288-thread block per SM
__global__ void __launch_bounds__(288, 1) stream(const unsigned char* W, long total, int SB, int NS, float* out) {
    extern __shared__ __align__(128) unsigned char smem[];
    const unsigned ring = (unsigned)__cvta_generic_to_shared(smem), bars = ring + NS * SB;
    const int t = threadIdx.x, w = t >> 5, lane = t & 31;
    const long U = total / SB, u0 = U * blockIdx.x / gridDim.x, u1 = U * (blockIdx.x + 1) / gridDim.x;
    if (t == 0) {
        for (int i = 0; i < NS; ++i) { bar_init(bars + 8 * i, 1); bar_init(bars + 8 * (NS + i), 8); }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (w == 8) {
        if (lane == 0)
            for (long u = u0, i = 0; u < u1; ++u, ++i) {
                const int slot = i % NS;
                if (i >= NS) bar_wait(bars + 8 * (NS + slot), ((i / NS) - 1) & 1);
                bar_expect(bars + 8 * slot, SB);
                bulk(ring + slot * SB, W + u * SB, SB, bars + 8 * slot);
            }
        return;
    }
    float acc = 0.f;
    for (long u = u0, i = 0; u < u1; ++u, ++i) {
        const int slot = i % NS;
        bar_wait(bars + 8 * slot, (i / NS) & 1);
        const float4* p = (const float4*)(smem + slot * SB);
        for (int j = t; j < SB / 16; j += 256) { const float4 v = p[j]; acc += v.x + v.w; }
        __syncwarp();
        if (lane == 0) bar_arrive(bars + 8 * (NS + slot));
    }
    if (acc == 1.2345f) out[0] = acc;
}
// BF16 copy of a gemm's FP32 output, as its PDL dependent (for consumers that need BF16)
__global__ void cvt(const float4* __restrict__ src, uint2* __restrict__ dst, int n4) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
        const float4 v = src[i];
        const __nv_bfloat162 a = __floats2bfloat162_rn(v.x, v.y), b = __floats2bfloat162_rn(v.z, v.w);
        dst[i] = make_uint2(*(const unsigned*)&a, *(const unsigned*)&b);
    }
}
}  // namespace g20
void g20_cvt(torch::Tensor src, torch::Tensor dst) {
    const int n4 = src.numel() / 4;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((n4 + 255) / 256 < 132 ? (n4 + 255) / 256 : 132); cfg.blockDim = dim3(256);
    cfg.stream = c10::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr; cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, g20::cvt, (const float4*)src.data_ptr(), (uint2*)dst.data_ptr(), n4);
}
void g20_stream(torch::Tensor w, int64_t sb, int64_t ns, torch::Tensor out) {
    const int sm = sb * ns + 16 * ns;
    cudaFuncSetAttribute(g20::stream, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
    g20::stream<<<132, 288, sm, c10::cuda::getCurrentCUDAStream()>>>((const unsigned char*)w.data_ptr(), w.numel(), sb, ns, out.data_ptr<float>());
}
void g20_pred(torch::Tensor src, torch::Tensor dst, int64_t ns_) {
    g20::pred<<<32, 256, 0, c10::cuda::getCurrentCUDAStream()>>>((const uint4*)src.data_ptr(), (uint4*)dst.data_ptr(), src.numel() * src.element_size() / 16, ns_);
}
void g20_zero(torch::Tensor out, torch::Tensor ctr) {
    g20::zero<<<32, 256, 0, c10::cuda::getCurrentCUDAStream()>>>((float4*)out.data_ptr(), out.numel() / 4, ctr.data_ptr<int>());
}
#ifdef G20_TRACE
torch::Tensor g20_trace() {
    auto o = torch::zeros({132, 64}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    cudaMemcpyFromSymbol(o.data_ptr(), g20::trace, sizeof(unsigned long long) * 132 * 64);
    return o;
}
#endif
// mode 0: a zero kernel (out, ctr) first, the gemm its PDL dependent. mode 1: out (and ctr if tail) already zero
// (stream-ordered before the previous kernel), the gemm a PDL dependent of the previous kernel, streaming W early.
// rpw: row groups per consumer warp (2: one x fragment load feeds two row groups; consumer warps = rg * ts / rpw).
// rg: 16-row groups per tile = consumer warps (x ts); the packed stage layout depends on it (pack19 rg).
// ts: consumer warps per 16-row group (2: each takes half the tokens; the W decode is done twice).
// xb: x buffers (2: the next k-chunk's x loads while the current one is in use; 1: less smem, the load waits).
// npre (modes 1, 2): weight stages streamed before griddepcontrol.wait (0: wait first, x first; use 0 when the
// previous kernel leaves nothing to overlap). mode 2: as 1, but the gemm zeroes out itself (grid barrier on the epoch counter ctr[1]: one int per call site
// and grid size, any initial multiple of the grid, e.g. 0; tail must be 0).
// out: FP32 [m, n] (overwritten), ctr: int32 [>= 1] scratch. grid: blocks (<= #SMs, 1 per SM). tail: dynamic stages.
static PFN_cuTensorMapEncodeTiled_v12000 g20_encode = nullptr;
// x tensor map: bf16 [m, k], box {64 k, m tokens}, SW128 (encoded at every launch from the x actually passed: under
// graph capture it is baked into the kernel node with that buffer)
static void g20_xmap(CUtensorMap& map, void* ptr, int m, int k) {
    if (!g20_encode) {
        cudaDriverEntryPointQueryResult q;
        cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", (void**)&g20_encode, 12000, cudaEnableDefault, &q);
        TORCH_CHECK(g20_encode, "g20: no cuTensorMapEncodeTiled");
    }
    cuuint64_t dims[2] = {(cuuint64_t)k, (cuuint64_t)m};
    cuuint64_t strides[1] = {(cuuint64_t)k * 2};
    cuuint32_t box[2] = {64, (cuuint32_t)m};
    cuuint32_t es[2] = {1, 1};
    CUresult r = g20_encode(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, ptr, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "g20: cuTensorMapEncodeTiled failed ", (int)r);
}
void g20_gemm(torch::Tensor wp, torch::Tensor h, double scale, torch::Tensor x, torch::Tensor out, torch::Tensor ctr,
              int64_t kc, int64_t ap, int64_t ns, int64_t is, int64_t grid, int64_t tail, int64_t mode, int64_t npre, int64_t xb, int64_t ts, int64_t rg, int64_t rpw) {
    const int m = x.size(0), k = x.size(1), n = out.size(1);
    const int sm = 1024 + ns * 8 * ap * 768 + xb * m * kc * 2 + g20::NPB * 8 * 16 * m * 4 + 4 * (ns + 2) + 8 * (2 * ns + 8) + 8;
    CUtensorMap xmap;
    g20_xmap(xmap, x.data_ptr(), m, k);
    TORCH_CHECK(n % (16 * rg) == 0 && k % kc == 0 && sm <= 227 * 1024 && tail % is == 0 && (mode < 2 || tail == 0) && npre <= ns, "g20: shape");
    auto stream = c10::cuda::getCurrentCUDAStream();
    if (!mode) g20::zero<<<32, 256, 0, stream>>>((float4*)out.data_ptr(), m * n / 4, ctr.data_ptr<int>());
#ifdef G20_ONLY2
    TORCH_CHECK(mode == 2, "g20: built for mode 2 only");
#define G20_KERN(M_, KC_, AP_, NS_, IS_, XB_, TS_, RG_, RPW_) g20::gemm<M_, KC_, AP_, NS_, IS_, 2, XB_, TS_, RG_, RPW_>
#else
#define G20_KERN(M_, KC_, AP_, NS_, IS_, XB_, TS_, RG_, RPW_) (mode == 2 ? g20::gemm<M_, KC_, AP_, NS_, IS_, 2, XB_, TS_, RG_, RPW_> : mode ? g20::gemm<M_, KC_, AP_, NS_, IS_, 1, XB_, TS_, RG_, RPW_> : g20::gemm<M_, KC_, AP_, NS_, IS_, 0, XB_, TS_, RG_, RPW_>)
#endif
#define G20_CASE(M_, KC_, AP_, NS_, IS_, XB_, TS_, RG_, RPW_) \
    if (m == M_ && kc == KC_ && ap == AP_ && ns == NS_ && is == IS_ && xb == XB_ && ts == TS_ && rg == RG_ && rpw == RPW_) { \
        auto kern = G20_KERN(M_, KC_, AP_, NS_, IS_, XB_, TS_, RG_, RPW_); \
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, sm); \
        cudaFuncSetAttribute(kern, cudaFuncAttributePreferredSharedMemoryCarveout, 100); \
        { int nb_ = 0, nsm_ = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb_, kern, 320, sm); \
          cudaDeviceGetAttribute(&nsm_, cudaDevAttrMultiProcessorCount, 0); \
          TORCH_CHECK(mode < 2 || nb_ * nsm_ >= grid, "g20: grid not co-resident (grid barrier would hang): ", nb_, " blocks/SM"); } \
        cudaLaunchConfig_t cfg = {}; \
        cfg.gridDim = dim3(grid); cfg.blockDim = dim3(320); cfg.dynamicSmemBytes = sm; cfg.stream = stream; \
        cudaLaunchAttribute attr[1]; \
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; attr[0].val.programmaticStreamSerializationAllowed = 1; \
        cfg.attrs = attr; cfg.numAttrs = 1; \
        cudaLaunchKernelEx(&cfg, kern, (const unsigned char*)wp.data_ptr(), (const float2*)h.data_ptr(), (float)scale, xmap, \
            (const __nv_bfloat16*)x.data_ptr(), out.data_ptr<float>(), ctr.data_ptr<int>(), n, k, (int)tail, (int)npre); \
        C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }
    G20_CONFIGS
    TORCH_CHECK(false, "g20: no config");
}
