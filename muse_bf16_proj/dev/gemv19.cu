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
    asm volatile("{\n.reg .pred p;\nW19_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W19_%=;\n}"
                 :: "r"(b), "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk(unsigned dst, const void* src, int bytes, unsigned b) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(dst), "l"(src), "r"(bytes), "r"(b) : "memory");
}
// M tokens (16/32), KC k per chunk, AP atoms (32 k) per warp per stage, NS ring stages, IS stages per tail item.
// out FP32 [M, n] (zeroed by the primary), ctr int (zeroed by the primary). tail: stages left for dynamic items.
// smem: ring [NS][SB] | x [2][M][XS] | pbuf [2][128 rows][M] f32 | barriers | tag [NS] int | ptag [2] int
template <int M, int KC, int AP, int NS, int IS, int PRE>
__global__ void __launch_bounds__((NWC + 2) * 32, 1) gemm(const unsigned char* __restrict__ WP, const float2* __restrict__ H,
                                                         float scale, const __nv_bfloat16* __restrict__ x,
                                                         float* __restrict__ out, int* __restrict__ ctr, int n, int k, int tail, int npre) {
    constexpr int NT = M / 8, NA = KC / 32, SPK = NA / AP, SB = NWC * AP * 768, XS = KC * 2 + 16, PBF = NWC * 16 * M;
    static_assert(NA % AP == 0 && SPK % IS == 0, "config");
    extern __shared__ __align__(128) unsigned char smem[];
    const unsigned ring = (unsigned)__cvta_generic_to_shared(smem), xs = ring + NS * SB;
    float* pbuf = (float*)(smem + NS * SB + 2 * M * XS);
    const unsigned bars = (unsigned)__cvta_generic_to_shared(pbuf + 2 * PBF);
    int* tag = (int*)(pbuf + 2 * PBF) + 2 * (2 * NS + 8);  // after the barriers. per slot: u | xb << 30 | end << 31 (-1: no more work)
    int* ptag = tag + NS;                                 // per pbuf: tile (-1: consumers done)
    // full[NS], empty[NS], xfull[2], xempty[2], pfull[2], pempty[2]
    const unsigned xfull = bars + 16 * NS, xempty = xfull + 16, pfull = xfull + 32, pempty = xfull + 48;
    const int t = threadIdx.x, lane = t & 31, w = t >> 5, g = lane >> 2, c = lane & 3;
    const int nrg = n / 16, ntr = nrg / NWC, S = k / KC, KA = k / 32;
    const int U = S * ntr * SPK, US = U - tail;
    if (t == 0) {
        for (int i = 0; i < NS; ++i) { bar_init(bars + 8 * i, 1); bar_init(bars + 8 * (NS + i), NWC); }
        for (int i = 0; i < 2; ++i) {
            bar_init(xfull + 8 * i, 1); bar_init(xempty + 8 * i, NWC); bar_init(pfull + 8 * i, NWC); bar_init(pempty + 8 * i, 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (t == 0) TR(0);
#ifdef G19_TRACE
    if (t == 0) { unsigned sm_; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_)); trace[blockIdx.x * 64 + 10] = sm_; }
#endif
    if (w == NWC) {  // producer: one lane walks the static range, then the tail items
        if (lane == 0) {
            int ra = (int)((long)US * blockIdx.x / gridDim.x), rb = (int)((long)US * (blockIdx.x + 1) / gridDim.x);
            int xks[2] = {-1, -1}, xuse[2] = {0, 0}, xb = 1, i = 0;
            if (PRE && npre > 0 && ra < rb) {  // out zeroed long before: stream the first stages' W while the previous kernel runs
                const int ks0 = ra / (ntr * SPK);
                for (; i < npre && ra < rb && ra / (ntr * SPK) == ks0; ++i, ++ra) {
                    const int st = ra % SPK;
                    tag[i] = ra | (ra + 1 == rb || st + 1 == SPK ? 1 << 31 : 0);  // x buffer 0
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
                    const int ks = u / (ntr * SPK), st = u % SPK;
                    if (ks != xks[xb]) {  // switch x buffers, load chunk ks
                        xb ^= 1;
                        if (xuse[xb]) bar_wait(xempty + 8 * xb, (xuse[xb] - 1) & 1);
                        ++xuse[xb];
                        xks[xb] = ks;
                        bar_expect(xfull + 8 * xb, M * KC * 2);
                        for (int qq = 0; qq < M; ++qq) bulk(xs + xb * M * XS + qq * XS, x + (long)qq * k + (long)ks * KC, KC * 2, xfull + 8 * xb);
                    }
                    const int slot = i % NS;
                    if (i >= NS) bar_wait(bars + 8 * (NS + slot), ((i / NS) - 1) & 1);
                    const bool end = u + 1 == rb || st + 1 == SPK;
                    tag[slot] = u | xb << 30 | (end ? 1 << 31 : 0);
                    const unsigned full = bars + 8 * slot;
                    bar_expect(full, SB);
                    bulk(ring + slot * SB, WP + (long)u * SB, SB, full);
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
    if (w == NWC + 1) {  // finisher: pbuf [128 rows][M] -> out [M][n] by red.global.add.v4 (lane = token, 4 rows)
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
            const int pb = q & 1;
            bar_wait(pfull + 8 * pb, (q >> 1) & 1);
            const int tr = ptag[pb];
            if (tr < 0) break;
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
            __syncwarp();
            if (lane == 0) bar_arrive(pempty + 8 * pb);
        }
        return;
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    float acc[NT][4], acc2[NT][4];
#pragma unroll
    for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = acc2[j][0] = acc2[j][1] = acc2[j][2] = acc2[j][3] = 0.f;
    // ldmatrix row address: matrix q = lane / 8 (x4: n-tile pair member q / 2, k half q % 2), row = lane % 8
    const unsigned xrow = ((lane >> 4) * 8 + (lane & 7)) * XS + ((lane >> 3) & 1) * 16;
    int q = 0, cxb = -1, st0 = 0, prev_end = 1;
    unsigned xph[2] = {0, 0};
    float4 hd0, hd1;
    for (int i = 0;; ++i) {
        const int slot = i % NS;
        bar_wait(bars + 8 * slot, (i / NS) & 1);
        const int tg = tag[slot];
        if (tg == -1) break;
        const int u = tg & 0x3FFFFFFF, xb = (tg >> 30) & 1;
        const bool end = tg < 0;
        const int ks = u / (ntr * SPK), r = u % (ntr * SPK), tr = r / SPK, st = r % SPK;
        if (xb != cxb) {  // new x buffer: release the old one, wait for the new one
            if (cxb >= 0) { __syncwarp(); if (lane == 0) bar_arrive(xempty + 8 * cxb); }
            bar_wait(xfull + 8 * xb, xph[xb]);
            xph[xb] ^= 1;
            cxb = xb;
        }
        if (prev_end) {  // run start: its escape headers (rows g, g + 8 of the warp's row group; lane c: slots 2c, 2c + 1)
            st0 = st;
            const float2* hp0 = H + (long)((tr * NWC + w) * 16 + g) * ESC;
            hd0 = __ldg((const float4*)hp0 + c);
            hd1 = __ldg((const float4*)(hp0 + 8 * ESC) + c);
        }
        prev_end = end;
        const unsigned xbase = xs + xb * M * XS + xrow;
        const unsigned sb = ring + slot * SB + w * AP * 512 + lane * 16, nb = ring + slot * SB + NWC * AP * 512 + w * AP * 256 + lane * 8;
#pragma unroll
        for (int j = 0; j < AP; ++j) {
            uint4 sv;
            uint2 nv;
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(sv.x), "=r"(sv.y), "=r"(sv.z), "=r"(sv.w) : "r"(sb + j * 512));
            asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(nv.x), "=r"(nv.y) : "r"(nb + j * 256));
            const unsigned kb = (st * AP + j) * 64;  // byte offset of the atom's k within the chunk
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
        if (!end) continue;
        // run end: rows rg*16 + g (+8), tokens jj*8 + 2c (+1), k in [klo, khi)
        const int rg = tr * NWC + w, nat = (st - st0 + 1) * AP;
        const int klo = ks * KC + st0 * AP * 32, khi = ks * KC + (st + 1) * AP * 32;
        float v[NT][4];
#pragma unroll
        for (int jj = 0; jj < NT; ++jj)
#pragma unroll
            for (int e = 0; e < 4; ++e) { v[jj][e] = acc[jj][e] + acc2[jj][e]; acc[jj][e] = acc2[jj][e] = 0.f; }
        {
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
                        const unsigned char* xc = smem + NS * SB + xb * M * XS + (col - ks * KC) * 2;
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
        }
        const int row0 = rg * 16 + g;
        if (nat == KA && PRE != 2) {  // the whole dot: plain stores (PRE 2: out is being zeroed by other blocks)
#pragma unroll
            for (int jj = 0; jj < NT; ++jj)
#pragma unroll
                for (int e = 0; e < 4; ++e) out[(long)(jj * 8 + 2 * c + (e & 1)) * n + row0 + 8 * (e >> 1)] = v[jj][e] * scale;
            continue;
        }
        const int pb = q & 1;  // hand the partials to the finisher: pbuf[pb][w * 16 + row16][tok]
        if (q >= 2) bar_wait(pempty + 8 * pb, ((q >> 1) - 1) & 1);
        float* pw = pbuf + pb * PBF + (w * 16 + g) * M + 2 * c;
#pragma unroll
        for (int jj = 0; jj < NT; ++jj)
#pragma unroll
            for (int h = 0; h < 2; ++h) *(float2*)(pw + 8 * h * M + jj * 8) = make_float2(v[jj][2 * h], v[jj][2 * h + 1]);
        if (t == 0) ptag[pb] = tr;
        __syncwarp();
        if (lane == 0) bar_arrive(pfull + 8 * pb);
        ++q;
    }
    {  // done: tell the finisher
        const int pb = q & 1;
        if (q >= 2) bar_wait(pempty + 8 * pb, ((q >> 1) - 1) & 1);
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
// npre (modes 1, 2): weight stages streamed before griddepcontrol.wait (0: wait first, x first; use 0 when the
// previous kernel leaves nothing to overlap). mode 2: as 1, but the gemm zeroes out itself (grid barrier on the epoch counter ctr[1]: one int per call site
// and grid size, any initial multiple of the grid, e.g. 0; tail must be 0).
// out: FP32 [m, n] (overwritten), ctr: int32 [>= 1] scratch. grid: blocks (<= #SMs, 1 per SM). tail: dynamic stages.
void g19_gemm(torch::Tensor wp, torch::Tensor h, double scale, torch::Tensor x, torch::Tensor out, torch::Tensor ctr,
              int64_t kc, int64_t ap, int64_t ns, int64_t is, int64_t grid, int64_t tail, int64_t mode, int64_t npre) {
    const int m = x.size(0), k = x.size(1), n = out.size(1);
    const int sm = ns * g19::NWC * ap * 768 + 2 * m * (kc * 2 + 16) + 2 * g19::NWC * 16 * m * 4 + 4 * (ns + 2) + 8 * (2 * ns + 8) + 8;
    TORCH_CHECK(n % (16 * g19::NWC) == 0 && k % kc == 0 && sm <= 227 * 1024 && tail % is == 0 && (mode < 2 || tail == 0) && npre <= ns, "g19: shape");
    auto stream = c10::cuda::getCurrentCUDAStream();
    if (!mode) g19::zero<<<32, 256, 0, stream>>>((float4*)out.data_ptr(), m * n / 4, ctr.data_ptr<int>());
#ifdef G19_ONLY2
    TORCH_CHECK(mode == 2, "g19: built for mode 2 only");
#define G19_KERN(M_, KC_, AP_, NS_, IS_) g19::gemm<M_, KC_, AP_, NS_, IS_, 2>
#else
#define G19_KERN(M_, KC_, AP_, NS_, IS_) (mode == 2 ? g19::gemm<M_, KC_, AP_, NS_, IS_, 2> : mode ? g19::gemm<M_, KC_, AP_, NS_, IS_, 1> : g19::gemm<M_, KC_, AP_, NS_, IS_, 0>)
#endif
#define G19_CASE(M_, KC_, AP_, NS_, IS_) \
    if (m == M_ && kc == KC_ && ap == AP_ && ns == NS_ && is == IS_) { \
        auto kern = G19_KERN(M_, KC_, AP_, NS_, IS_); \
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, sm); \
        cudaLaunchConfig_t cfg = {}; \
        cfg.gridDim = dim3(grid); cfg.blockDim = dim3((g19::NWC + 2) * 32); cfg.dynamicSmemBytes = sm; cfg.stream = stream; \
        cudaLaunchAttribute attr[1]; \
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; attr[0].val.programmaticStreamSerializationAllowed = 1; \
        cfg.attrs = attr; cfg.numAttrs = 1; \
        cudaLaunchKernelEx(&cfg, kern, (const unsigned char*)wp.data_ptr(), (const float2*)h.data_ptr(), (float)scale, \
            (const __nv_bfloat16*)x.data_ptr(), out.data_ptr<float>(), ctr.data_ptr<int>(), n, k, (int)tail, (int)npre); \
        C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }
    G19_CONFIGS
    TORCH_CHECK(false, "g19: no config");
}
