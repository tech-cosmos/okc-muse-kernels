// gemv18: m = 16/32 over lossless 12-bit weights (_pack18: one contiguous block per stage), streamed by TMA bulk copies into an smem
// ring by one producer warp, so many bytes are in flight per SM without load instructions. NWC = 8 consumer warps,
// each owns one 16-row group of a 128-row tile (W = mma A operand, decoded in registers; x = B from smem by ldmatrix,
// x[:, k-chunk] loaded by bulk copies). Work = the flattened (k-chunk ks, tile, stage of AP atoms) space, split
// evenly over the grid (stream-K): a block's range is cut into segments (one (ks, tile) each); a segment covering all
// of k writes BF16 directly, else its partials go to an FP32 workspace (red.global.add.v2) and a per-row-group atom
// counter tells the warp completing the row group to write BF16 and re-zero (workspace and counters self-clean).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
namespace g18 {
constexpr int ESC = 8, NWC = 8;
#ifdef G18_TRACE
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
__device__ __forceinline__ void ldsm2(unsigned addr, unsigned& d0, unsigned& d1) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(d0), "=r"(d1) : "r"(addr));
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
    asm volatile("{\n.reg .pred p;\nW18_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W18_%=;\n}"
                 :: "r"(b), "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk(unsigned dst, const void* src, int bytes, unsigned b) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(dst), "l"(src), "r"(bytes), "r"(b) : "memory");
}
// M tokens (16/32), KC k per chunk, AP atoms (32 k) per warp per stage, NS ring stages, PDL: wait for the producer
// kernel (griddepcontrol.wait) only before touching x / ws / out.
// Warps: NWC consumers, 1 producer (TMA ring), 1 finisher (split segments: partials from smem -> red.global.add.v4 into
// ws, counters, last-arriver BF16 conversion; its global round trips overlap the consumers' streaming).
template <int M, int KC, int AP, int NS, int PDL, int F32>
__global__ void __launch_bounds__((NWC + 2) * 32, 1) gemm(const unsigned char* __restrict__ WP,
                                                         const float2* __restrict__ H, float scale, const __nv_bfloat16* __restrict__ x,
                                                         __nv_bfloat16* __restrict__ out, float* __restrict__ ws,  // F32: ws = the FP32 output [M, n] (zeroed before: PDL primary)
 int* __restrict__ cnt,
                                                         int n, int k) {
    constexpr int NT = M / 8, NA = KC / 32, SPK = NA / AP, SB = NWC * AP * 768, XS = KC * 2 + 16, PBF = NWC * 16 * M;
    static_assert(NA % AP == 0, "config");
    extern __shared__ __align__(128) unsigned char smem[];
    const unsigned ring = (unsigned)__cvta_generic_to_shared(smem), xs = ring + NS * SB;
    float* pbuf = (float*)(smem + NS * SB + M * XS);  // [2][NWC * 16 rows][M] segment partials for the finisher
    const unsigned bars = (unsigned)__cvta_generic_to_shared(pbuf + 2 * PBF);  // full[NS], empty[NS], xbar, pfull[2], pempty[2]
    const unsigned xbar = bars + 16 * NS, pfull = xbar + 8, pempty = xbar + 24;
    const int t = threadIdx.x, lane = t & 31, w = t >> 5, g = lane >> 2, c = lane & 3;
    const int nrg = n / 16, ntr = nrg / NWC, S = k / KC, KA = k / 32;
    const long U = (long)S * ntr * SPK;
    const int u0 = (int)(U * blockIdx.x / gridDim.x), u1 = (int)(U * (blockIdx.x + 1) / gridDim.x);
    if (t == 0) {
        for (int i = 0; i < NS; ++i) { bar_init(bars + 8 * i, 1); bar_init(bars + 8 * (NS + i), NWC); }
        bar_init(xbar, 1);
        bar_init(pfull, NWC); bar_init(pfull + 8, NWC); bar_init(pempty, 1); bar_init(pempty + 8, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (t == 0) TR(0);
    if (w == NWC) {  // producer
        if (lane == 0 && u0 < u1) {
            auto xload = [&]() {  // x[:, chunk of u0] first: nothing is consumable before it lands
                const int ks = u0 / (ntr * SPK);
                bar_expect(xbar, M * KC * 2);
                for (int qq = 0; qq < M; ++qq) bulk(xs + qq * XS, x + (long)qq * k + (long)ks * KC, KC * 2, xbar);
            };
            if (!PDL) xload();
#ifdef G18_XPRE
            if (PDL) { asm volatile("griddepcontrol.wait;" ::: "memory"); xload(); }
            constexpr int PW = 0;  // x issued before W
#else
            constexpr int PW = PDL;
#endif
#ifdef G18_XFIRST
            if (!PDL) bar_wait(xbar, 0);
#endif
            for (int u = u0, i = 0; u < u1; ++u, ++i) {
                if (PW && i == (NS < u1 - u0 ? NS : u1 - u0)) { asm volatile("griddepcontrol.wait;" ::: "memory"); xload(); }
                const int slot = i % NS;
                if (i >= NS) bar_wait(bars + 8 * (NS + slot), ((i / NS) - 1) & 1);
                const unsigned full = bars + 8 * slot, dst = ring + slot * SB;
                bar_expect(full, SB);
                #ifdef G18_ILV
                bulk(dst, WP + ((long)i * gridDim.x + blockIdx.x) * SB, SB, full);  // interleaved repack: stage i of every block adjacent
#else
                bulk(dst, WP + (long)u * SB, SB, full);
#endif  // stage u = [NWC x AP atoms of bytes][NWC x AP atoms of codes]
            }
            if (PW && u1 - u0 <= NS) { asm volatile("griddepcontrol.wait;" ::: "memory"); xload(); }
            TR(1);
        }
        return;
    }
    if (w == NWC + 1) {  // finisher: same segment walk as the consumers
        int q = 0, st0 = 0;
        for (int u = u0; u < u1; ++u) {
            const int ks = u / (ntr * SPK), r = u % (ntr * SPK), tr = r / SPK, st = r % SPK;
            if (u == u0 || st == 0) st0 = st;
            if (u + 1 < u1 && st + 1 < SPK) continue;
            const int nat = (st - st0 + 1) * AP;
            if (nat == KA) continue;  // written directly by the consumers
            (void)ks;
            const int pb = q & 1;
            bar_wait(pfull + 8 * pb, (q >> 1) & 1);
            if (F32) {  // pbuf [128 rows][M] -> ws [M][n]: lane = (token, 4 rows)
                if (q == 0) asm volatile("griddepcontrol.wait;" ::: "memory");
                const float* src = pbuf + pb * PBF;
                float* wt = ws + (long)tr * NWC * 16;
#pragma unroll 4
                for (int i = lane; i < PBF / 4; i += 32) {
                    const int tok = i % M, r4 = (i / M) * 4;
                    const float a0 = src[r4 * M + tok] * scale, a1 = src[(r4 + 1) * M + tok] * scale;
                    const float a2 = src[(r4 + 2) * M + tok] * scale, a3 = src[(r4 + 3) * M + tok] * scale;
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(wt + (long)tok * n + r4), "f"(a0), "f"(a1), "f"(a2), "f"(a3) : "memory");
                }
                __syncwarp();
                if (lane == 0) bar_arrive(pempty + 8 * pb);
                ++q;
                continue;
            }
            const float4* src = (const float4*)(pbuf + pb * PBF);
            float* wt = ws + (long)tr * NWC * 16 * M;  // the tile's 128 rows x M, contiguous
#pragma unroll 4
            for (int i = lane; i < PBF / 4; i += 32) {
                const float4 v = src[i];
                asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(wt + 4 * i), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
            }
            __syncwarp();
            if (lane == 0) bar_arrive(pempty + 8 * pb);
            if (lane == 0 && q < 4) TR(40 + 6 * q);
            ++q;
#ifdef G18_NOFIN
            continue;
#endif
            asm volatile("fence.acq_rel.gpu;" ::: "memory");
            __syncwarp();
            int last = 0;
            if (lane < NWC) last = atomicAdd(cnt + tr * NWC + lane, nat) + nat == KA;
            unsigned mask = __ballot_sync(0xffffffffu, last);
            if (lane == 0 && q < 5) TR(35 + 6 * q);
            if (!mask) continue;
            asm volatile("fence.acq_rel.gpu;" ::: "memory");
            constexpr int PR = 16 * M / 4 / 32;  // float4 per lane per row group
#pragma unroll
            for (int h = 0; h < 2; ++h) {  // row groups 4h..4h+3: loads first, then zero + convert
                float4 v[4][PR];
#pragma unroll
                for (int rr = 0; rr < 4; ++rr)
#pragma unroll
                    for (int j = 0; j < PR; ++j)
                        if (mask >> (4 * h + rr) & 1) v[rr][j] = __ldcg((const float4*)(wt + (4 * h + rr) * 16 * M) + lane + 32 * j);
#pragma unroll
                for (int rr = 0; rr < 4; ++rr)
#pragma unroll
                    for (int j = 0; j < PR; ++j)
                        if (mask >> (4 * h + rr) & 1) {
                            const int i = lane + 32 * j;
                            __stcg((float4*)(wt + (4 * h + rr) * 16 * M) + i, make_float4(0.f, 0.f, 0.f, 0.f));
                            const int row = (tr * NWC + 4 * h + rr) * 16 + (4 * i) / M, tok = (4 * i) % M;
                            out[(long)tok * n + row] = __float2bfloat16(v[rr][j].x * scale);
                            out[(long)(tok + 1) * n + row] = __float2bfloat16(v[rr][j].y * scale);
                            out[(long)(tok + 2) * n + row] = __float2bfloat16(v[rr][j].z * scale);
                            out[(long)(tok + 3) * n + row] = __float2bfloat16(v[rr][j].w * scale);
                        }
            }
            if (lane < NWC && (mask >> lane & 1)) cnt[tr * NWC + lane] = 0;
            if (lane == 0 && q < 5) TR(36 + 6 * q);
        }
        return;
    }
    if (PDL) asm volatile("griddepcontrol.wait;" ::: "memory");
    float acc[NT][4], acc2[NT][4];
#pragma unroll
    for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = acc2[j][0] = acc2[j][1] = acc2[j][2] = acc2[j][3] = 0.f;
    // ldmatrix row address: matrix q = lane / 8 (x4: n-tile pair member q / 2, k half q % 2), row = lane % 8
    const unsigned xbase = xs + ((lane >> 4) * 8 + (lane & 7)) * XS + ((lane >> 3) & 1) * 16;
    int cur = -1, st0 = 0, q = 0;
    unsigned xph = 0;
    // escape headers of rows g, g + 8 of the current segment (lane c: slots 2c, 2c + 1), loaded one segment ahead
    auto hload = [&](int u, float4& a, float4& b) {
        const int tr = (u % (ntr * SPK)) / SPK;
        const float4* hp = (const float4*)(H + (long)((tr * NWC + w) * 16 + g) * ESC) + c;
        a = __ldg(hp);
        b = __ldg(hp + 8 * ESC / 2);
    };
    float4 hd0, hd1;
    if (u0 < u1) hload(u0, hd0, hd1);
    for (int u = u0, i = 0; u < u1; ++u, ++i) {
        const int ks = u / (ntr * SPK), r = u % (ntr * SPK), tr = r / SPK, st = r % SPK;
        if (ks != cur) {  // (re)load x[:, chunk ks] once every consumer is done with the old chunk (first: producer)
            if (cur >= 0) asm volatile("bar.sync 1, %0;" :: "n"(NWC * 32) : "memory");
            if (t == 0 && cur >= 0) {
                bar_expect(xbar, M * KC * 2);
                for (int qq = 0; qq < M; ++qq) bulk(xs + qq * XS, x + (long)qq * k + (long)ks * KC, KC * 2, xbar);
            }
            bar_wait(xbar, xph);
            if (t == 0 && xph == 0) TR(2);
            if (t == 0 && xph == 1) TR(3);
            xph ^= 1;
            cur = ks;
        }
        if (u == u0 || st == 0) st0 = st;
        const int slot = i % NS;
        bar_wait(bars + 8 * slot, (i / NS) & 1);
        const unsigned sb = ring + slot * SB + w * AP * 512 + lane * 16, nb = ring + slot * SB + NWC * AP * 512 + w * AP * 256 + lane * 8;
#pragma unroll
        for (int j = 0; j < AP; ++j) {
            uint4 sv;
            uint2 nv;
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(sv.x), "=r"(sv.y), "=r"(sv.z), "=r"(sv.w) : "r"(sb + j * 512));
            asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(nv.x), "=r"(nv.y) : "r"(nb + j * 256));
            const unsigned kb = (st * AP + j) * 64;  // byte offset of the atom's k within the chunk
#ifdef G18_NOMMA
            acc[0][j & 3] += __uint_as_float(sv.x ^ sv.y ^ sv.z ^ sv.w ^ nv.x ^ nv.y); continue;
#endif
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                const uint4 af = dec8(s ? sv.z : sv.x, s ? sv.w : sv.y, s ? nv.y : nv.x);
#pragma unroll
                for (int jj = 0; jj + 1 < NT; jj += 2) {
                    unsigned b0, b1, b2, b3;
                    ldsm4(xbase + jj * 8 * XS + kb + s * 32, b0, b1, b2, b3);
                    mma(s ? acc2[jj] : acc[jj], af, b0, b1);
                    mma(s ? acc2[jj + 1] : acc[jj + 1], af, b2, b3);
                }
            }
        }
        __syncwarp();
        if (lane == 0) bar_arrive(bars + 8 * (NS + slot));
        if (u + 1 < u1 && st + 1 < SPK) continue;
#ifdef G18_NOFLUSH
        continue;
#endif
        // segment end: rows rg*16 + g (+8), tokens jj*8 + 2c (+1), k in [klo, khi)
        const int rg = tr * NWC + w, nat = (st - st0 + 1) * AP;
        const int klo = ks * KC + st0 * AP * 32, khi = ks * KC + (st + 1) * AP * 32;
        float v[NT][4];
#pragma unroll
        for (int jj = 0; jj < NT; ++jj)
#pragma unroll
            for (int e = 0; e < 4; ++e) { v[jj][e] = acc[jj][e] + acc2[jj][e]; acc[jj][e] = acc2[jj][e] = 0.f; }
#ifndef G18_NOESC
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const float4 hq = h ? hd1 : hd0;
#pragma unroll
            for (int e2 = 0; e2 < ESC; ++e2) {  // empty slot = (col 0, 2^-15): skipped (real escapes: |value| < 2^-15)
                const int src = (lane & ~3) | (e2 >> 1);
                const float hx = __shfl_sync(0xffffffffu, (e2 & 1) ? hq.z : hq.x, src);
                const float hy = __shfl_sync(0xffffffffu, (e2 & 1) ? hq.w : hq.y, src);
                const int col = __float_as_int(hx);
                if (hy != 0x1p-15f && col >= klo && col < khi) {  // two fmas: exact when the dot holds only this weight (probe rows)
                    const unsigned char* xc = smem + NS * SB + (col - ks * KC) * 2;
#pragma unroll
                    for (int jj = 0; jj < NT; ++jj)
#pragma unroll
                        for (int b = 0; b < 2; ++b) {
                            const float xv = __bfloat162float(*(const __nv_bfloat16*)(xc + (jj * 8 + 2 * c + b) * XS));
                            v[jj][2 * h + b] = fmaf(xv, hy, fmaf(xv, -0x1p-15f, v[jj][2 * h + b]));
                        }
                }
            }
        }
#endif
        if (u + 1 < u1) hload(u + 1, hd0, hd1);
        const int row0 = rg * 16 + g;
        if (nat == KA) {
#pragma unroll
            for (int jj = 0; jj < NT; ++jj)
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    if (F32) ws[(long)(jj * 8 + 2 * c + (e & 1)) * n + row0 + 8 * (e >> 1)] = v[jj][e] * scale; else
                    out[(long)(jj * 8 + 2 * c + (e & 1)) * n + row0 + 8 * (e >> 1)] = __float2bfloat16(v[jj][e] * scale);
            continue;
        }
        // hand the partials to the finisher: pbuf[pb][w * 16 + row16][tok]
        if (t == 0 && q < 8) TR(8 + q);
        const int pb = q & 1;
        if (q >= 2) bar_wait(pempty + 8 * pb, ((q >> 1) - 1) & 1);
        float* pw = pbuf + pb * PBF + (w * 16 + g) * M + 2 * c;
#pragma unroll
        for (int jj = 0; jj < NT; ++jj)
#pragma unroll
            for (int h = 0; h < 2; ++h) *(float2*)(pw + 8 * h * M + jj * 8) = make_float2(v[jj][2 * h], v[jj][2 * h + 1]);
        __syncwarp();
        if (lane == 0) bar_arrive(pfull + 8 * pb);
        if (t == 0 && q < 8) TR(16 + q);
        ++q;
    }
    if (t == 0) TR(4);
}
// PDL primary of an F32 gemm: lets it launch at once, zeroes its output (the gemm waits before its first red)
__global__ void zero(float4* __restrict__ p, int n4) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) p[i] = make_float4(0.f, 0.f, 0.f, 0.f);
}
}  // namespace g18
// ws: FP32 zeros >= n * m (self-cleaning), cnt: int32 zeros >= n / 16 (self-cleaning). grid: blocks (<= #SMs).
#ifdef G18_TRACE
torch::Tensor g18_trace() {
    auto o = torch::zeros({132, 64}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    cudaMemcpyFromSymbol(o.data_ptr(), g18::trace, sizeof(unsigned long long) * 132 * 64);
    return o;
}
#endif
// pdl: 0 plain, 1 PDL dependent of the previous kernel, 2 = FP32 output into wsf [m, n] (not self-cleaning; a zero
// kernel launched first is the PDL primary)
constexpr int ZB = 32;
void g18_gemm(torch::Tensor wp, torch::Tensor h, double scale, torch::Tensor x, torch::Tensor out,
              torch::Tensor wsf, torch::Tensor cnt, int64_t kc, int64_t ap, int64_t ns, int64_t grid, int64_t pdl) {
    const int m = x.size(0), k = x.size(1), n = out.size(1);
    const int sm = ns * g18::NWC * ap * 768 + m * (kc * 2 + 16) + 2 * g18::NWC * 16 * m * 4 + 8 * (2 * ns + 5);
    TORCH_CHECK(n % (16 * g18::NWC) == 0 && k % kc == 0 && sm <= 227 * 1024, "g18: shape");
#define G18_CASE(M_, KC_, AP_, NS_) \
    if (m == M_ && kc == KC_ && ap == AP_ && ns == NS_) { \
        auto kern = pdl == 2 ? g18::gemm<M_, KC_, AP_, NS_, 1, 1> : pdl ? g18::gemm<M_, KC_, AP_, NS_, 1, 0> : g18::gemm<M_, KC_, AP_, NS_, 0, 0>; \
        if (pdl == 2) g18::zero<<<ZB, 256, 0, c10::cuda::getCurrentCUDAStream()>>>((float4*)wsf.data_ptr(), m * n / 4); \
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, sm); \
        cudaLaunchConfig_t cfg = {}; \
        cfg.gridDim = dim3(grid); cfg.blockDim = dim3((g18::NWC + 2) * 32); cfg.dynamicSmemBytes = sm; \
        cfg.stream = c10::cuda::getCurrentCUDAStream(); \
        cudaLaunchAttribute attr[1]; \
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; attr[0].val.programmaticStreamSerializationAllowed = 1; \
        cfg.attrs = attr; cfg.numAttrs = pdl ? 1 : 0; \
        cudaLaunchKernelEx(&cfg, kern, (const unsigned char*)wp.data_ptr(), \
            (const float2*)h.data_ptr(), (float)scale, (const __nv_bfloat16*)x.data_ptr(), \
            (__nv_bfloat16*)out.data_ptr(), wsf.data_ptr<float>(), cnt.data_ptr<int>(), n, k); \
        C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }
    G18_CONFIGS
    TORCH_CHECK(false, "g18: no config");
}
