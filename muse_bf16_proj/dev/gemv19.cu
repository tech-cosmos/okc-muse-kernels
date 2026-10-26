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
namespace g19 {
#ifdef G19_RAW  // s24: plain BF16 weights (stages = mma A fragments, [atom][s][lane][16 B]): no decode, no escapes, scale 1
constexpr int WB = 1024;
#else
constexpr int WB = 768;  // bytes per 16-row x 32-k atom: 512 sign|mantissa + 256 codes
#endif
constexpr int ESC = 8, NWC = 8;
#ifdef G19_TRACE
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
template <int P>
__device__ __forceinline__ unsigned expo(unsigned nw) {
    const unsigned s = P == 0 ? nw << 7 : P == 1 ? nw << 3 : P == 2 ? nw >> 1 : nw >> 5;
    return (s & 0x07800780u) | 0x38003800u;
}
__device__ __forceinline__ uint4 dec8(unsigned s0, unsigned s1, unsigned nw) {
    uint4 r;
    r.x = (prmt(s0, 0x9180u) & 0x807f807fu) | expo<0>(nw);
    r.y = (prmt(s0, 0xB3A2u) & 0x807f807fu) | expo<1>(nw);
    r.z = (prmt(s1, 0x9180u) & 0x807f807fu) | expo<2>(nw);
    r.w = (prmt(s1, 0xB3A2u) & 0x807f807fu) | expo<3>(nw);
    return r;
}
__device__ __forceinline__ void mma(float* c, const uint4& a, unsigned b0, unsigned b1) {
#ifdef G19_MMANV  // not volatile: a pure register op, free to be scheduled among the decode / ldmatrix work
    asm(
#else
    asm volatile(
#endif
                 "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
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
#ifdef G19_SPIN  // diag: non-suspending test_wait spin instead of try_wait
    asm volatile("{\n.reg .pred p;\nW19_%=:\nmbarrier.test_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W19_%=;\n}"
                 :: "r"(b), "r"(parity) : "memory");
#else
    asm volatile("{\n.reg .pred p;\nW19_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W19_%=;\n}"
                 :: "r"(b), "r"(parity) : "memory");
#endif
}
__device__ __forceinline__ void bulk(unsigned dst, const void* src, int bytes, unsigned b) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(dst), "l"(src), "r"(bytes), "r"(b) : "memory");
}
// M tokens (16/32), KC k per chunk, AP atoms (32 k) per warp per stage, NS ring stages, IS stages per tail item.
// out FP32 [M, n] (zeroed by the primary), ctr int (zeroed by the primary). tail: stages left for dynamic items.
// smem: ring [NS][SB] | x [2][M][XS] | pbuf [2][128 rows][M] f32 | barriers | tag [NS] int | ptag [2] int
#ifndef G19_NPB
#define G19_NPB 2
#endif
constexpr int NPB = G19_NPB;  // partial buffers (1: less smem, for 2 blocks per SM)
#ifndef G19_MINB
#define G19_MINB 1
#endif
template <int M, int KC, int AP, int NS, int IS, int PRE, int XB, int TS, int RG, int RPW>
__global__ void __launch_bounds__((RG * TS / RPW + 2) * 32, G19_MINB) gemm(const unsigned char* __restrict__ WP, const float2* __restrict__ H,
                                                         float scale, const __nv_bfloat16* __restrict__ x,
                                                         float* __restrict__ out, int* __restrict__ ctr, int n, int k, int tail, int npre,
                                                         __nv_bfloat16* __restrict__ ob, int tm) {
    // PRE 3: BF16 output ob [M, n]; out = FP32 workspace that is ZERO on entry and left zero (the finisher that completes
    // a tile's K reads it back, writes BF16, re-zeroes it); ctr[2 + tile] = atoms done per tile (+KA per call, never reset)
    constexpr int NWC = RG;  // 16-row groups per tile (one stage covers RG x AP atoms)
    static_assert(RPW == 1 || TS == 1, "config");
    constexpr int NCW = NWC * TS / RPW, MT = M / TS, NT = MT / 8, NA = KC / 32, SPK = NA / AP, SB = NWC * AP * WB, XS = KC * 2 + 16, PBF = NWC * 16 * M;
    static_assert(NA % AP == 0 && SPK % IS == 0 && (NT % 2 == 0 || NT == 1) && M <= 32, "config");  // NT even: x B fragments by ldmatrix.x4 (NT 1: one x4 = both k16 halves)
    extern __shared__ __align__(128) unsigned char smem[];
    const unsigned ring = (unsigned)__cvta_generic_to_shared(smem), xs = ring + NS * SB;
    float* pbuf = (float*)(smem + NS * SB + XB * M * XS);
    const unsigned bars = (unsigned)__cvta_generic_to_shared(pbuf + NPB * PBF);
#ifdef G19_XQ  // x chunk in 4 quarters, one mbarrier each (xq[buffer][quarter], after the other barriers): stages start on their quarter
#ifndef G19_XPAR
#error "G19_XQ needs G19_XPAR"
#endif
    constexpr int XQB = 16;
#else
    constexpr int XQB = 0;
#endif
    int* tag = (int*)(pbuf + NPB * PBF) + 2 * (2 * NS + 8) + XQB;  // after the barriers. per slot: u | xb << 29 | last x use << 30 | run end << 31 (-1: no more work)
    int* ptag = tag + NS;                                 // per pbuf: tile | atoms << 12 (-1: consumers done)
    // PRE 3: tile read-back buffer [M][NWC * 16] f32 (bulk copy) + its mbarrier
    const unsigned cbuf = ((unsigned)__cvta_generic_to_shared(ptag + 2) + 127u) & ~127u, cbar = cbuf + M * NWC * 64;
    // full[NS], empty[NS], xfull[2], xempty[2], pfull[2], pempty[2]
    const unsigned xfull = bars + 16 * NS, xempty = xfull + 16, pfull = xfull + 32, pempty = xfull + 48, xq = xfull + 64;
    const int t = threadIdx.x, lane = t & 31, w = t >> 5, g = lane >> 2, c = lane & 3;
    const int nrg = n / 16, ntr = nrg / NWC, S = k / KC, KA = k / 32;
    const int U = S * ntr * SPK, US = U - tail;
    // tm: tile-major stage order (pack19 tm): stage u = (tile, k-chunk, st): tiles complete throughout the kernel
    // (mode 3: their BF16 conversions overlap the stream instead of all landing in the tail)
    auto KSof = [&](int u_) { return tm ? (u_ / SPK) % S : u_ / (ntr * SPK); };
    if (t == 0) {
        for (int i = 0; i < NS; ++i) { bar_init(bars + 8 * i, 1); bar_init(bars + 8 * (NS + i), NCW); }
        for (int i = 0; i < 2; ++i) {
            bar_init(xfull + 8 * i, 1); bar_init(xempty + 8 * i, NCW); bar_init(pfull + 8 * i, NCW); bar_init(pempty + 8 * i, 1);
        }
        if (XQB) for (int i = 0; i < 8; ++i) bar_init(xq + 8 * i, 1);
        if (PRE == 3) bar_init(cbar, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (t == 0) TR(0);
#ifdef G19_TRACE
    if (t == 0) { unsigned sm_; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_)); trace[blockIdx.x * 64 + 10] = sm_; }
#endif
#ifdef G19_XPAR
    if (w == NCW) {  // producer warp, all lanes in lockstep: lane 0 issues W / tags / barriers, x rows go one per lane
        int ra = (int)((long)US * blockIdx.x / gridDim.x), rb = (int)((long)US * (blockIdx.x + 1) / gridDim.x);
        int xks[2] = {-1, -1}, xuse[2] = {0, 0}, xb = 1, i = 0;
        auto xload = [&](int b_, int ks_) {
#ifdef G19_XQ
            if (lane == 0) for (int q4 = 0; q4 < 4; ++q4) bar_expect(xq + 8 * (b_ * 4 + q4), M * KC / 2);
            __syncwarp();
            for (int i4 = lane; i4 < 4 * M; i4 += 32) {  // quarter-major: quarter 0 of every row first
                const int q4 = i4 / M, qq = i4 % M;
                bulk(xs + b_ * M * XS + qq * XS + q4 * (KC / 2), x + (long)qq * k + (long)ks_ * KC + q4 * (KC / 4), KC / 2, xq + 8 * (b_ * 4 + q4));
            }
#else
            if (lane == 0) bar_expect(xfull + 8 * b_, M * KC * 2);
            __syncwarp();
            for (int qq = lane; qq < M; qq += 32) bulk(xs + b_ * M * XS + qq * XS, x + (long)qq * k + (long)ks_ * KC, KC * 2, xfull + 8 * b_);
#endif
        };
        if (PRE && npre > 0 && ra < rb) {  // out zeroed long before: stream the first stages' W while the previous kernel runs
            const int ks0 = KSof(ra);
            for (; i < npre && ra < rb && KSof(ra) == ks0; ++i, ++ra) {
                const int st = ra % SPK;
                const bool xl = ra + 1 == rb || KSof(ra + 1) != ks0;
                if (lane == 0) {
                    tag[i] = ra | (xl ? 1 << 30 : 0) | (ra + 1 == rb || st + 1 == SPK ? 1 << 31 : 0);  // x buffer 0
                    bar_expect(bars + 8 * i, SB);
                    bulk(ring + i * SB, WP + (long)ra * SB, SB, bars + 8 * i);
                }
            }
            asm volatile("griddepcontrol.wait;" ::: "memory");
            xb = 0; xks[0] = ks0; xuse[0] = 1;
            xload(0, ks0);
        } else {
            asm volatile("griddepcontrol.wait;" ::: "memory");  // x, out, ctr ready
        }
        for (;;) {
            if (ra >= rb) {  // next run: a tail item
                if (tail == 0) break;
                int it = 0;
                if (lane == 0) it = atomicAdd(ctr, 1);
                it = __shfl_sync(0xffffffffu, it, 0);
                if (it * IS >= tail) break;
                ra = US + it * IS; rb = ra + IS;
            }
            for (int u = ra; u < rb; ++u, ++i) {
                const int ks = KSof(u), st = u % SPK;
                if (ks != xks[xb]) {  // next x buffer, load chunk ks (consumers release a buffer after its last use)
                    xb = XB == 2 ? xb ^ 1 : 0;
                    if (xuse[xb]) bar_wait(xempty + 8 * xb, (xuse[xb] - 1) & 1);
                    ++xuse[xb];
                    xks[xb] = ks;
                    xload(xb, ks);
                }
                const int slot = i % NS;
                if (i >= NS) bar_wait(bars + 8 * (NS + slot), ((i / NS) - 1) & 1);
                const bool end = u + 1 == rb || st + 1 == SPK;
                const bool xl = u + 1 == rb || KSof(u + 1) != ks;
                if (xl) xks[xb] = -1;  // released by the consumers after this stage: reload even if the next run has this ks
                if (lane == 0) {
                    tag[slot] = u | xb << 29 | (xl ? 1 << 30 : 0) | (end ? 1 << 31 : 0);
                    if (i < 24) TR(40 + i);
                    const unsigned full = bars + 8 * slot;
#ifdef G19_NOW
                    bar_arrive(full);
#else
                    bar_expect(full, SB);
                    bulk(ring + slot * SB, WP + (long)u * SB, SB, full);
#endif
                }
            }
            ra = rb;
        }
        const int slot = i % NS;  // end marker
        if (i >= NS) bar_wait(bars + 8 * (NS + slot), ((i / NS) - 1) & 1);
        if (lane == 0) {
            tag[slot] = -1;
            bar_arrive(bars + 8 * slot);
            TR(1);
        }
        return;
    }
#else
    if (w == NCW) {  // producer: one lane walks the static range, then the tail items
        if (lane == 0) {
            int ra = (int)((long)US * blockIdx.x / gridDim.x), rb = (int)((long)US * (blockIdx.x + 1) / gridDim.x);
            int xks[2] = {-1, -1}, xuse[2] = {0, 0}, xb = 1, i = 0;
            if (PRE && npre > 0 && ra < rb) {  // out zeroed long before: stream the first stages' W while the previous kernel runs
                const int ks0 = KSof(ra);
                for (; i < npre && ra < rb && KSof(ra) == ks0; ++i, ++ra) {
                    const int st = ra % SPK;
                    const bool xl = ra + 1 == rb || KSof(ra + 1) != ks0;
                    tag[i] = ra | (xl ? 1 << 30 : 0) | (ra + 1 == rb || st + 1 == SPK ? 1 << 31 : 0);  // x buffer 0
                    bar_expect(bars + 8 * i, SB);
                    bulk(ring + i * SB, WP + (long)ra * SB, SB, bars + 8 * i);
                }
                asm volatile("griddepcontrol.wait;" ::: "memory");
                xb = 0; xks[0] = ks0; xuse[0] = 1;
                bar_expect(xfull, M * KC * 2);
                for (int qq = 0; qq < M; ++qq) bulk(xs + qq * XS, x + (long)qq * k + (long)ks0 * KC, KC * 2, xfull);
            } else {
                asm volatile("griddepcontrol.wait;" ::: "memory");  // x, out, ctr ready
            }
            for (;;) {
                if (ra >= rb) {  // next run: a tail item
                    if (tail == 0) break;
                    const int it = atomicAdd(ctr, 1);  // grabbed when needed: the ring's NS stages hide its latency
                    if (it * IS >= tail) break;
                    ra = US + it * IS; rb = ra + IS;
                }
                for (int u = ra; u < rb; ++u, ++i) {
                    const int ks = KSof(u), st = u % SPK;
                    if (ks != xks[xb]) {  // next x buffer, load chunk ks (consumers release a buffer after its last use)
                        xb = XB == 2 ? xb ^ 1 : 0;
                        if (xuse[xb]) bar_wait(xempty + 8 * xb, (xuse[xb] - 1) & 1);
                        ++xuse[xb];
                        xks[xb] = ks;
                        bar_expect(xfull + 8 * xb, M * KC * 2);
                        for (int qq = 0; qq < M; ++qq) bulk(xs + xb * M * XS + qq * XS, x + (long)qq * k + (long)ks * KC, KC * 2, xfull + 8 * xb);
                    }
                    const int slot = i % NS;
                    if (i >= NS) bar_wait(bars + 8 * (NS + slot), ((i / NS) - 1) & 1);
                    const bool end = u + 1 == rb || st + 1 == SPK;
                    const bool xl = u + 1 == rb || KSof(u + 1) != ks;
                    if (xl) xks[xb] = -1;  // released by the consumers after this stage: reload even if the next run has this ks
                    tag[slot] = u | xb << 29 | (xl ? 1 << 30 : 0) | (end ? 1 << 31 : 0);
                    if (i < 24) TR(40 + i);
                    const unsigned full = bars + 8 * slot;
#ifdef G19_NOW
                    bar_arrive(full);  // consumers compute on stale smem: their throughput alone
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
#endif
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
        int ncv = 0;
        for (int q = 0;; ++q) {
            const int pb = q % NPB;
            bar_wait(pfull + 8 * pb, (q / NPB) & 1);
            const int ptg = ptag[pb];
            if (ptg < 0) break;
            const int tr = ptg & 0xFFF;
            if (PRE == 2 && q == 0) {  // every block has zeroed its slice
                if (lane == 0) {
                    unsigned v;
                    do { asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(v) : "l"(ctr + 1) : "memory"); } while ((int)(v - target) < 0);
                }
                __syncwarp();
            }
            const float* src = pbuf + pb * PBF;
            float* wt = out + (long)tr * NWC * 16;
#pragma unroll 4
            for (int i = lane; i < PBF / 4; i += 32) {
                const int tok = i % M, r4 = (i / M) * 4;
                const float a0 = src[r4 * M + tok] * scale, a1 = src[(r4 + 1) * M + tok] * scale;
                const float a2 = src[(r4 + 2) * M + tok] * scale, a3 = src[(r4 + 3) * M + tok] * scale;
                asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(wt + (long)tok * n + r4), "f"(a0), "f"(a1), "f"(a2), "f"(a3) : "memory");
            }
            if (PRE == 3) __threadfence();  // this lane's reds before the tile ticket
            __syncwarp();
            if (lane == 0) bar_arrive(pempty + 8 * pb);
            if (lane == 0 && q < 2) TR(2 + q);
            if (PRE == 3) {
                int last = 0;
                if (lane == 0) {
                    const int na = ptg >> 12, old = atomicAdd(ctr + 2 + tr, na);
                    last = (old + na) % KA == 0;
                }
                if (__shfl_sync(0xffffffffu, last, 0)) {  // all of the tile's K is in: FP32 -> BF16, re-zero
                    // one bulk copy per token row (512 B for RG 8): a single round trip instead of a chain of loads
                    if (lane == 0) {
                        __threadfence();
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                        bar_expect(cbar, M * NWC * 64);
                        for (int tok = 0; tok < M; ++tok) bulk(cbuf + tok * NWC * 64, wt + (long)tok * n, NWC * 64, cbar);
                    }
                    bar_wait(cbar, ncv & 1);
                    ++ncv;
                    const float4* cb = (const float4*)(smem + (cbuf - (unsigned)__cvta_generic_to_shared(smem)));
#pragma unroll 4
                    for (int i = lane; i < M * NWC * 4; i += 32) {
                        const int tok = i / (NWC * 4), r4 = (i % (NWC * 4)) * 4;
                        const float4 a = cb[i];
                        __stcg((float4*)(wt + (long)tok * n + r4), make_float4(0.f, 0.f, 0.f, 0.f));
                        const __nv_bfloat162 b0 = __floats2bfloat162_rn(a.x, a.y), b1 = __floats2bfloat162_rn(a.z, a.w);
                        *(uint2*)(ob + (long)tok * n + (long)tr * NWC * 16 + r4) = make_uint2(*(const unsigned*)&b0, *(const unsigned*)&b1);
                    }
                    __syncwarp();  // cbuf reads done before the next bulk copy
                }
            }
        }
        return;
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (t == 0) TR(9);
    // RPW = 2: warp w owns row groups 2w, 2w + 1 of the tile (one x fragment load feeds both: half the ldmatrix traffic)
    float acc[RPW][NT][4], acc2[RPW][NT][4];
#pragma unroll
    for (int r = 0; r < RPW; ++r)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[r][j][e] = acc2[r][j][e] = 0.f;
    // ldmatrix row address: matrix q = lane / 8 (x4: n-tile pair member q / 2, k half q % 2), row = lane % 8
    // TS = 2: warps 2r, 2r + 1 share row group r (both decode it), tokens [th * MT, th * MT + MT)
    const int wr = (w / TS) * RPW, th = w % TS;  // wr: the warp's first row group in the tile
    // NT 1 (M 8): matrix q = (k16 step s = q / 2, k half q % 2) of the single n-tile
    const unsigned xrow = NT == 1 ? (th * MT + (lane & 7)) * XS + ((lane >> 3) & 1) * 16 + (lane >> 4) * 32
                                  : (th * MT + (lane >> 4) * 8 + (lane & 7)) * XS + ((lane >> 3) & 1) * 16;
    int q = 0, st0 = 0, prev_end = 1, prev_xl = 1;
    unsigned xph[2] = {0, 0}, xpc[2] = {0, 0};
    float4 hd0[RPW], hd1[RPW];
    for (int i = 0;; ++i) {
        const int slot = i % NS;
        bar_wait(bars + 8 * slot, (i / NS) & 1);
        const int tg = tag[slot];
        if (t == 0 && i < 24) TR(16 + i);
        if (tg == -1) break;
        const int u = tg & 0x1FFFFFFF, xb = (tg >> 29) & 1;
        const bool end = tg < 0, xl = (tg >> 30) & 1;
        const int ks = KSof(u), tr = tm ? u / (S * SPK) : (u % (ntr * SPK)) / SPK, st = u % SPK;
#ifdef G19_XQ
        if (prev_xl) { xpc[xb] = xph[xb]; xph[xb] ^= 1; }  // a new x use of buffer xb: its quarters' phase parity
        prev_xl = xl;
        for (int q4 = (st * AP * 32) / (KC / 4); q4 <= ((st + 1) * AP * 32 - 1) / (KC / 4); ++q4) bar_wait(xq + 8 * (xb * 4 + q4), xpc[xb]);
        if (false) {
#else
        if (prev_xl) {  // first stage of an x use: wait for it
#endif
#ifndef G19_NOXW  // diag: never wait for x (garbage results): the x arrival's share of the time
            bar_wait(xfull + 8 * xb, xph[xb]);
#endif
            if (t == 0 && i == 0) TR(5);
            xph[xb] ^= 1;
        }
        prev_xl = xl;
#ifndef G19_RAW
        if (prev_end) {  // run start: its escape headers (rows g, g + 8 of the warp's row groups; lane c: slots 2c, 2c + 1)
            st0 = st;
#pragma unroll
            for (int rr = 0; rr < RPW; ++rr) {
                const float2* hp0 = H + (long)((tr * NWC + wr + rr) * 16 + g) * ESC;
                hd0[rr] = __ldg((const float4*)hp0 + c);
                hd1[rr] = __ldg((const float4*)(hp0 + 8 * ESC) + c);
            }
        }
#endif
        prev_end = end;
        const unsigned xbase = xs + xb * M * XS + xrow;
        const unsigned sb = ring + slot * SB + wr * AP * 512 + lane * 16, nb = ring + slot * SB + NWC * AP * 512 + wr * AP * 256 + lane * 8;
#if defined(G19_OLDLOOP) || defined(G19_NOMMA) || defined(G19_NOX)
        static_assert(RPW == 1, "diagnostic loops: RPW 1 only");
#pragma unroll
        for (int j = 0; j < AP; ++j) {
            uint4 sv;
            uint2 nv;
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(sv.x), "=r"(sv.y), "=r"(sv.z), "=r"(sv.w) : "r"(sb + j * 512));
            asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(nv.x), "=r"(nv.y) : "r"(nb + j * 256));
            const unsigned kb = (st * AP + j) * 64;  // byte offset of the atom's k within the chunk
#ifdef G19_NOMMA
            acc[0][0][j & 3] += __uint_as_float(sv.x ^ sv.y ^ sv.z ^ sv.w ^ nv.x ^ nv.y); continue;
#endif
#ifdef G19_NOX
            { const uint4 af = dec8(sv.x, sv.y, nv.x), af2 = dec8(sv.z, sv.w, nv.y);
              acc[0][0][j & 3] += __uint_as_float(af.x ^ af.y ^ af.z ^ af.w ^ af2.x ^ af2.y ^ af2.z ^ af2.w); continue; }
#endif
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                const uint4 af = dec8(s ? sv.z : sv.x, s ? sv.w : sv.y, s ? nv.y : nv.x);
#pragma unroll
                for (int jj = 0; jj + 1 < NT; jj += 2) {
                    unsigned b0, b1, b2, b3;
                    ldsm4(xbase + jj * 8 * XS + kb + s * 32, b0, b1, b2, b3);
                    mma(s ? acc2[0][jj] : acc[0][jj], af, b0, b1);
                    mma(s ? acc2[0][jj + 1] : acc[0][jj + 1], af, b2, b3);
                }
            }
        }
#else
        // groups of G atoms: all shared loads of the group first, then the x fragments, then decode + mma (the asm
        // statements are volatile: issued in program order, so independent loads must come first to overlap)
        constexpr int G = NT * RPW == 2 ? AP : AP < 2 ? AP : 2;
#pragma unroll
        for (int j0 = 0; j0 < AP; j0 += G) {
#ifdef G19_RAW
            uint4 fa[RPW][G][2];
            const unsigned rb_ = ring + slot * SB + wr * AP * 1024 + lane * 16;
#pragma unroll
            for (int rr = 0; rr < RPW; ++rr)
#pragma unroll
                for (int j = 0; j < G; ++j)
#pragma unroll
                    for (int s = 0; s < 2; ++s)
                        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(fa[rr][j][s].x), "=r"(fa[rr][j][s].y), "=r"(fa[rr][j][s].z), "=r"(fa[rr][j][s].w) : "r"(rb_ + rr * AP * 1024 + (j0 + j) * 1024 + s * 512));
#else
            uint4 sv[RPW][G];
            uint2 nv[RPW][G];
#pragma unroll
            for (int rr = 0; rr < RPW; ++rr)
#pragma unroll
                for (int j = 0; j < G; ++j) {
                    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(sv[rr][j].x), "=r"(sv[rr][j].y), "=r"(sv[rr][j].z), "=r"(sv[rr][j].w) : "r"(sb + rr * AP * 512 + (j0 + j) * 512));
                    asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(nv[rr][j].x), "=r"(nv[rr][j].y) : "r"(nb + rr * AP * 256 + (j0 + j) * 256));
                }
#endif
            unsigned bx[G][2][NT][2];
#pragma unroll
            for (int j = 0; j < G; ++j)
                if (NT == 1) ldsm4(xbase + (st * AP + j0 + j) * 64, bx[j][0][0][0], bx[j][0][0][1], bx[j][1][0][0], bx[j][1][0][1]);
#pragma unroll
            for (int j = 0; j < G; ++j)
#pragma unroll
                for (int s = 0; s < 2; ++s)
#pragma unroll
                    for (int jj = 0; jj + 1 < NT; jj += 2)
                        ldsm4(xbase + jj * 8 * XS + (st * AP + j0 + j) * 64 + s * 32, bx[j][s][jj][0], bx[j][s][jj][1], bx[j][s][jj + 1][0], bx[j][s][jj + 1][1]);
#pragma unroll
            for (int j = 0; j < G; ++j)
#pragma unroll
                for (int rr = 0; rr < RPW; ++rr)
#pragma unroll
                    for (int s = 0; s < 2; ++s) {
#ifdef G19_RAW
                        const uint4 af = fa[rr][j][s];
#else
                        const uint4 af = dec8(s ? sv[rr][j].z : sv[rr][j].x, s ? sv[rr][j].w : sv[rr][j].y, s ? nv[rr][j].y : nv[rr][j].x);
#endif
#pragma unroll
                        for (int jj = 0; jj < NT; ++jj) mma(s ? acc2[rr][jj] : acc[rr][jj], af, bx[j][s][jj][0], bx[j][s][jj][1]);
                    }
        }
#endif
#ifdef G19_TRACE
        if (t == 0 && i == 0) { TR(6); volatile float z_ = hd0[0].x + hd1[0].y; TR(7); }
        if (t == 0 && i == 1) TR(8);
#endif
        __syncwarp();
        if (lane == 0) bar_arrive(bars + 8 * (NS + slot));
        if (!end) continue;  // (a last x use always ends a run: released below, after the escape fixups read x)
        // run end: rows rg*16 + g (+8), tokens jj*8 + 2c (+1), k in [klo, khi)
        if (t == 0 && q == 0) TR(11);
        const int nat = (st - st0 + 1) * AP;
        const int klo = ks * KC + st0 * AP * 32, khi = ks * KC + (st + 1) * AP * 32;
#if defined(G19_OLDFIX) || defined(G19_RFIX)
        constexpr bool sfix = false;
#else
        const bool sfix = !(nat == KA && PRE != 2);  // pbuf path: escape fixups applied in smem after the store (below)
#endif
        float v[RPW][NT][4];
#pragma unroll
        for (int rr = 0; rr < RPW; ++rr)
#pragma unroll
            for (int jj = 0; jj < NT; ++jj)
#pragma unroll
                for (int e = 0; e < 4; ++e) { v[rr][jj][e] = acc[rr][jj][e] + acc2[rr][jj][e]; acc[rr][jj][e] = acc2[rr][jj][e] = 0.f; }
#pragma unroll
        for (int rr = 0; rr < (WB == 1024 ? 0 : RPW); ++rr)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                if (sfix) break;
                const float4 hq = h ? hd1[rr] : hd0[rr];
#ifndef G19_OLDFIX
                // which slots any lane of the warp needs (owner lanes test their own 2 slots; OR over the 8 row lanes g):
                // the loop below then skips empty slots by warp-uniform branches instead of a serialized shfl chain
                unsigned f0 = __ballot_sync(0xffffffffu, hq.y != 0x1p-15f && __float_as_int(hq.x) >= klo && __float_as_int(hq.x) < khi);
                unsigned f1 = __ballot_sync(0xffffffffu, hq.w != 0x1p-15f && __float_as_int(hq.z) >= klo && __float_as_int(hq.z) < khi);
                f0 |= f0 >> 16; f0 |= f0 >> 8; f0 |= f0 >> 4;
                f1 |= f1 >> 16; f1 |= f1 >> 8; f1 |= f1 >> 4;
#endif
#pragma unroll
                for (int e2 = 0; e2 < ESC; ++e2) {  // empty slot = (col 0, 2^-15): skipped (real escapes: |value| < 2^-15)
#ifndef G19_OLDFIX
                    if (!((((e2 & 1) ? f1 : f0) >> (e2 >> 1)) & 1u)) continue;
#endif
#ifdef G19_NOFIX
                    continue;
#endif
                    const int src = (lane & ~3) | (e2 >> 1);
                    const float hx = __shfl_sync(0xffffffffu, (e2 & 1) ? hq.z : hq.x, src);
                    const float hy = __shfl_sync(0xffffffffu, (e2 & 1) ? hq.w : hq.y, src);
                    const int col = __float_as_int(hx);
                    if (hy != 0x1p-15f && col >= klo && col < khi) {  // two fmas: exact when the dot holds only this weight (probe rows)
                        const unsigned char* xc = smem + NS * SB + xb * M * XS + (col - ks * KC) * 2;
#pragma unroll
                        for (int jj = 0; jj < NT; ++jj)
#pragma unroll
                            for (int b = 0; b < 2; ++b) {
                                const float xv = __bfloat162float(*(const __nv_bfloat16*)(xc + (th * MT + jj * 8 + 2 * c + b) * XS));
                                v[rr][jj][2 * h + b] = fmaf(xv, hy, fmaf(xv, -0x1p-15f, v[rr][jj][2 * h + b]));
                            }
                    }
                }
            }
        if (t == 0 && q == 0) TR(12);
        if (xl && !sfix) {  // done with this x buffer (the fixups above were its last reads)
            __syncwarp();
            if (lane == 0) bar_arrive(xempty + 8 * xb);
        }
        if (nat == KA && PRE != 2) {  // the whole dot: plain stores (PRE 2: out is being zeroed by other blocks)
#pragma unroll
            for (int rr = 0; rr < RPW; ++rr)
#pragma unroll
                for (int jj = 0; jj < NT; ++jj)
#pragma unroll
                    for (int e = 0; e < 4; ++e)
                    {
                        const long o_ = (long)(th * MT + jj * 8 + 2 * c + (e & 1)) * n + (tr * NWC + wr + rr) * 16 + g + 8 * (e >> 1);
                        if (PRE == 3) ob[o_] = __float2bfloat16_rn(v[rr][jj][e] * scale);
                        else out[o_] = v[rr][jj][e] * scale;
                    }
            continue;
        }
        const int pb = q % NPB;  // hand the partials to the finisher: pbuf[pb][row group * 16 + row16][tok]
        if (q >= NPB) bar_wait(pempty + 8 * pb, ((q / NPB) - 1) & 1);
        if (t == 0 && q == 1) TR(13);
#pragma unroll
        for (int rr = 0; rr < RPW; ++rr) {
            float* pw = pbuf + pb * PBF + ((wr + rr) * 16 + g) * M + th * MT + 2 * c;
#pragma unroll
            for (int jj = 0; jj < NT; ++jj)
#pragma unroll
                for (int h = 0; h < 2; ++h) *(float2*)(pw + 8 * h * M + jj * 8) = make_float2(v[rr][jj][2 * h], v[rr][jj][2 * h + 1]);
        }
#if !defined(G19_OLDFIX) && !defined(G19_RFIX)
        // escape fixups on the stored partials, one escape at a time over the warp (lane = token): the in-range slots
        // (lane c holds slots 2c, 2c + 1 of rows g, g + 8) are found by ballots and walked in lane order, warp-uniformly
        if (WB != 1024) {
            __syncwarp();
            const unsigned char* xt = smem + NS * SB + xb * M * XS + (th * MT + lane) * XS - ks * KC * 2;
#pragma unroll
            for (int rr = 0; rr < RPW; ++rr)
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int pr = 0; pr < 2; ++pr) {
                        const float4 hq = h ? hd1[rr] : hd0[rr];
                        const float hx = pr ? hq.z : hq.x, hy = pr ? hq.w : hq.y;
                        unsigned b = __ballot_sync(0xffffffffu, hy != 0x1p-15f && __float_as_int(hx) >= klo && __float_as_int(hx) < khi);
                        while (b) {
                            const int src = __ffs(b) - 1;
                            b &= b - 1;
                            const int col = __float_as_int(__shfl_sync(0xffffffffu, hx, src));
                            const float hv = __shfl_sync(0xffffffffu, hy, src);
                            if (lane < MT) {
                                float* pp = pbuf + pb * PBF + ((wr + rr) * 16 + (src >> 2) + 8 * h) * M + th * MT + lane;
                                const float xv = __bfloat162float(*(const __nv_bfloat16*)(xt + col * 2));
                                *pp = fmaf(xv, hv, fmaf(xv, -0x1p-15f, *pp));
                            }
                        }
                    }
            __syncwarp();
        }
        if (xl) {  // done with this x buffer (the fixups above were its last reads)
            __syncwarp();
            if (lane == 0) bar_arrive(xempty + 8 * xb);
        }
#endif
        if (t == 0) ptag[pb] = tr | nat << 12;
        __syncwarp();
        if (lane == 0) bar_arrive(pfull + 8 * pb);
        if (t == 0 && q == 0) TR(14);
        if (t == 0 && q == 1) TR(15);
        ++q;
    }
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
}  // namespace g19
void g19_cvt(torch::Tensor src, torch::Tensor dst) {
    const int n4 = src.numel() / 4;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((n4 + 255) / 256 < 132 ? (n4 + 255) / 256 : 132); cfg.blockDim = dim3(256);
    cfg.stream = c10::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr; cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, g19::cvt, (const float4*)src.data_ptr(), (uint2*)dst.data_ptr(), n4);
}
void g19_stream(torch::Tensor w, int64_t sb, int64_t ns, torch::Tensor out) {
    const int sm = sb * ns + 16 * ns;
    cudaFuncSetAttribute(g19::stream, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
    g19::stream<<<132, 288, sm, c10::cuda::getCurrentCUDAStream()>>>((const unsigned char*)w.data_ptr(), w.numel(), sb, ns, out.data_ptr<float>());
}
void g19_pred(torch::Tensor src, torch::Tensor dst, int64_t ns_) {
    g19::pred<<<32, 256, 0, c10::cuda::getCurrentCUDAStream()>>>((const uint4*)src.data_ptr(), (uint4*)dst.data_ptr(), src.numel() * src.element_size() / 16, ns_);
}
void g19_zero(torch::Tensor out, torch::Tensor ctr) {
    g19::zero<<<32, 256, 0, c10::cuda::getCurrentCUDAStream()>>>((float4*)out.data_ptr(), out.numel() / 4, ctr.data_ptr<int>());
}
#ifdef G19_TRACE
torch::Tensor g19_trace() {
    auto o = torch::zeros({132, 64}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    cudaMemcpyFromSymbol(o.data_ptr(), g19::trace, sizeof(unsigned long long) * 132 * 64);
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
// mode 3: BF16 out ob; out = FP32 workspace [m, n], zero before the first call and left zero; ctr int32 [>= 2 + n / (16 rg)]
// zero before the first call (one per call site; tail must be 0). No predecessor needed (x read after griddepcontrol.wait).
void g19_gemm_ob(torch::Tensor wp, torch::Tensor h, double scale, torch::Tensor x, torch::Tensor out, torch::Tensor ctr,
              int64_t kc, int64_t ap, int64_t ns, int64_t is, int64_t grid, int64_t tail, int64_t mode, int64_t npre, int64_t xb, int64_t ts, int64_t rg, int64_t rpw,
              __nv_bfloat16* ob, int tm);
void g19_gemm(torch::Tensor wp, torch::Tensor h, double scale, torch::Tensor x, torch::Tensor out, torch::Tensor ctr,
              int64_t kc, int64_t ap, int64_t ns, int64_t is, int64_t grid, int64_t tail, int64_t mode, int64_t npre, int64_t xb, int64_t ts, int64_t rg, int64_t rpw) {
    TORCH_CHECK(mode < 3, "g19: mode 3 = g19_gemm_bf16");
    g19_gemm_ob(wp, h, scale, x, out, ctr, kc, ap, ns, is, grid, tail, mode, npre, xb, ts, rg, rpw, nullptr, 0);
}
void g19_gemm_bf16(torch::Tensor wp, torch::Tensor h, double scale, torch::Tensor x, torch::Tensor ws, torch::Tensor ob, torch::Tensor ctr,
              int64_t kc, int64_t ap, int64_t ns, int64_t grid, int64_t npre, int64_t xb, int64_t ts, int64_t rg, int64_t rpw, int64_t tm) {
    TORCH_CHECK(ob.scalar_type() == at::kBFloat16 && ob.sizes() == ws.sizes() && ctr.numel() >= 2 + ws.size(1) / (16 * rg), "g19_gemm_bf16: args");
    g19_gemm_ob(wp, h, scale, x, ws, ctr, kc, ap, ns, 1, grid, 0, 3, npre, xb, ts, rg, rpw, (__nv_bfloat16*)ob.data_ptr(), (int)tm);
}
void g19_gemm_ob(torch::Tensor wp, torch::Tensor h, double scale, torch::Tensor x, torch::Tensor out, torch::Tensor ctr,
              int64_t kc, int64_t ap, int64_t ns, int64_t is, int64_t grid, int64_t tail, int64_t mode, int64_t npre, int64_t xb, int64_t ts, int64_t rg, int64_t rpw,
              __nv_bfloat16* ob, int tm) {
    const int m = x.size(0), k = x.size(1), n = out.size(1);
    const int sm = ns * rg * ap * g19::WB + xb * m * (kc * 2 + 16) + g19::NPB * rg * 16 * m * 4 + 4 * (ns + 2) + 8 * (2 * ns + 8) + 8
#ifdef G19_XQ
                 + 64
#endif

                 + (mode == 3 ? 128 + m * rg * 64 + 16 : 0);
    TORCH_CHECK(n % (16 * rg) == 0 && k % kc == 0 && sm <= 227 * 1024 && tail % is == 0 && (mode < 2 || tail == 0) && npre <= ns && n / (16 * rg) <= 4096, "g19: shape");
    auto stream = c10::cuda::getCurrentCUDAStream();
    if (!mode) g19::zero<<<32, 256, 0, stream>>>((float4*)out.data_ptr(), m * n / 4, ctr.data_ptr<int>());
#ifdef G19_ONLY2
    TORCH_CHECK(mode == 2, "g19: built for mode 2 only");
#define G19_KERN(M_, KC_, AP_, NS_, IS_, XB_, TS_, RG_, RPW_) g19::gemm<M_, KC_, AP_, NS_, IS_, 2, XB_, TS_, RG_, RPW_>
#elif defined(G19_ONLY23)
    TORCH_CHECK(mode >= 2, "g19: built for modes 2, 3 only");
#define G19_KERN(M_, KC_, AP_, NS_, IS_, XB_, TS_, RG_, RPW_) (mode == 3 ? g19::gemm<M_, KC_, AP_, NS_, IS_, 3, XB_, TS_, RG_, RPW_> : g19::gemm<M_, KC_, AP_, NS_, IS_, 2, XB_, TS_, RG_, RPW_>)
#else
#define G19_KERN(M_, KC_, AP_, NS_, IS_, XB_, TS_, RG_, RPW_) (mode == 3 ? g19::gemm<M_, KC_, AP_, NS_, IS_, 3, XB_, TS_, RG_, RPW_> : mode == 2 ? g19::gemm<M_, KC_, AP_, NS_, IS_, 2, XB_, TS_, RG_, RPW_> : mode ? g19::gemm<M_, KC_, AP_, NS_, IS_, 1, XB_, TS_, RG_, RPW_> : g19::gemm<M_, KC_, AP_, NS_, IS_, 0, XB_, TS_, RG_, RPW_>)
#endif
#define G19_CASE(M_, KC_, AP_, NS_, IS_, XB_, TS_, RG_, RPW_) \
    if (m == M_ && kc == KC_ && ap == AP_ && ns == NS_ && is == IS_ && xb == XB_ && ts == TS_ && rg == RG_ && rpw == RPW_) { \
        auto kern = G19_KERN(M_, KC_, AP_, NS_, IS_, XB_, TS_, RG_, RPW_); \
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, sm); \
        if (G19_MINB > 1) { cudaFuncSetAttribute(kern, cudaFuncAttributePreferredSharedMemoryCarveout, 100); \
          int nb_ = 0, nsm_ = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb_, kern, (RG_ * TS_ / RPW_ + 2) * 32, sm); \
          cudaDeviceGetAttribute(&nsm_, cudaDevAttrMultiProcessorCount, 0); \
          TORCH_CHECK(mode < 2 || nb_ * nsm_ >= grid, "g19: grid not co-resident (grid barrier would hang): ", nb_, " blocks/SM"); } \
        cudaLaunchConfig_t cfg = {}; \
        cfg.gridDim = dim3(grid); cfg.blockDim = dim3((RG_ * TS_ / RPW_ + 2) * 32); cfg.dynamicSmemBytes = sm; cfg.stream = stream; \
        cudaLaunchAttribute attr[1]; \
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; attr[0].val.programmaticStreamSerializationAllowed = 1; \
        cfg.attrs = attr; cfg.numAttrs = 1; \
        cudaLaunchKernelEx(&cfg, kern, (const unsigned char*)wp.data_ptr(), (const float2*)h.data_ptr(), (float)scale, \
            (const __nv_bfloat16*)x.data_ptr(), out.data_ptr<float>(), ctr.data_ptr<int>(), n, k, (int)tail, (int)npre, ob, tm); \
        C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }
    G19_CONFIGS
    TORCH_CHECK(false, "g19: no config");
}
