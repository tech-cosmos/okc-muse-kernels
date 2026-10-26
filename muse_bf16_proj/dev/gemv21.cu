// gemv21: m = 1 GEMV over gemv13's lossless 12-bit weights (same packing: byte plane, nibble words, escape headers)
// with the weight stream in a shared-memory TMA ring instead of registers: one bulk copy per plane per row, so the
// bytes in flight per SM (NS rows x 12 KB) no longer cost registers. Few registers per thread leave room for another
// kernel's CTAs on the same SM (the layer's csl attention beside the qkv GEMV, QG).
// Block = NG consumer groups of WG warps + 1 producer warp. Stage = one row (bytes, nibbles, header). Rows by
// global ticket (dynamic: no slow-SM tail); this CTA's it-th row goes to stage it % NS, group it % NG. A consumer thread owns fixed 16-weight groups of
// the row (x in registers as FP32), warp sums -> red[s][warp]; the group's last warp (smem ticket) adds the WG
// partials in warp order (deterministic); the producer warp, before refilling a stage, adds that row's WG partials,
// applies the escapes (true / scale - 2^-15 per entry, as gemv13) and stores.
// qg != nullptr: after a row's store, red.release +1 on qg[row >= nq] (QG: a PDL secondary spins on these).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
namespace g21 {
constexpr int ESC = 8;
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
template <int P0>
__device__ __forceinline__ void dec4(unsigned sw, unsigned nw, float* f) {
    const unsigned q0 = (prmt(sw, 0x9180u) & 0x807f807fu) | expo<P0>(nw);
    const unsigned q1 = (prmt(sw, 0xB3A2u) & 0x807f807fu) | expo<P0 + 1>(nw);
    f[0] = __uint_as_float(q0 << 16); f[1] = __uint_as_float(q0 & 0xffff0000u);
    f[2] = __uint_as_float(q1 << 16); f[3] = __uint_as_float(q1 & 0xffff0000u);
}
__device__ __forceinline__ float dot16(const uint4& s, const uint2& nb, const float* x) {
    float f[16];
    dec4<0>(s.x, nb.x, f); dec4<2>(s.y, nb.x, f + 4); dec4<0>(s.z, nb.y, f + 8); dec4<2>(s.w, nb.y, f + 12);
    float a0 = 0.f, a1 = 0.f;
#pragma unroll
    for (int i = 0; i < 16; i += 2) { a0 = fmaf(f[i], x[i], a0); a1 = fmaf(f[i + 1], x[i + 1], a1); }
    return a0 + a1;
}
__device__ __forceinline__ void mbar_init(unsigned a, unsigned cnt) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(cnt) : "memory");
}
#ifndef G21_SUSP
#define G21_SUSP 0
#endif
__device__ __forceinline__ void mbar_wait(unsigned a, unsigned par) {
    if (G21_SUSP)  // suspend-time hint: the warp sleeps until the phase completes instead of spinning on issue slots
        asm volatile("{\n\t.reg .pred p;\n\tW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2;\n\t@!p bra W;\n\t}"
                     ::"r"(a), "r"(par), "r"(G21_SUSP) : "memory");
    else
        asm volatile("{\n\t.reg .pred p;\n\tW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t@!p bra W;\n\t}"
                     ::"r"(a), "r"(par) : "memory");
}
__device__ __forceinline__ void mbar_arrive(unsigned a) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(a) : "memory");
}
__device__ __forceinline__ void mbar_expect(unsigned a, unsigned bytes) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(a), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk(unsigned dst, const void* src, unsigned bytes, unsigned mb) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 ::"r"(dst), "l"(src), "r"(bytes), "r"(mb) : "memory");
}

template <int KK, int WG, int NG, int NS>
__global__ void __launch_bounds__((NG * WG + 1) * 32, 1) gemv(const unsigned char* __restrict__ S,
                                                              const unsigned char* __restrict__ N,
                                                              const float2* __restrict__ H,
                                                              const __nv_bfloat16* __restrict__ x,
                                                              __nv_bfloat16* __restrict__ out, int n, float scale,
                                                              unsigned* qg, int nq, unsigned long long* tk) {
    constexpr int GT = WG * 32, NJ = KK / 16 / GT, RB = KK + KK / 2 + ESC * 8, TQ = 4, DT = 3;
    static_assert(KK % (16 * GT) == 0 && NS % NG == 0, "shape");
    extern __shared__ __align__(128) unsigned char sm[];
    unsigned char* ring = sm;                                                  // [NS][RB]
    __nv_bfloat16* xb = reinterpret_cast<__nv_bfloat16*>(sm + NS * RB);       // [KK] (escapes)
    float* red = reinterpret_cast<float*>(sm + NS * RB + KK * 2);             // [NS][WG]
    int* tick = reinterpret_cast<int*>(red + NS * WG);                        // [NS]
    int* rowid = tick + NS;                                                    // [NS]
    unsigned long long* bars = reinterpret_cast<unsigned long long*>(sm + ((NS * RB + KK * 2 + NS * WG * 4 + NS * 8 + 7) & ~7));
    const unsigned full0 = (unsigned)__cvta_generic_to_shared(bars), empty0 = full0 + NS * 8;
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int grp = warp / WG, wi = warp % WG, gt = wi * 32 + lane;
    // Rows by ticket: the producer keeps TQ tickets in flight (one thread's atomics on one address return increasing
    // values); it stops at its first ticket >= n, holding TQ - 1 more (all >= n): one call takes exactly
    // n + TQ * gridDim.x tickets, so tk is never reset and ticket % (n + TQ * gridDim.x) is the row (or a stop).
    // Static rows first (row = blockIdx.x + it * G for it < r0), tickets only for the dynamic tail (the last ~DT
    // rows per CTA: one counter cannot hand out every row, same-address atomics serialize at ~0.5 us per round).
    const int G = gridDim.x, r0 = max(0, n / G - DT), nd = n - r0 * G;
    const unsigned long long NT = (unsigned long long)nd + TQ * G;
    unsigned long long tq[TQ];
    // lane 1 of the producer warp owns the tickets: lane 0's release arrives (TMA issue) and lane 2's release counter
    // bumps (finalize) must not wait for a pending ticket atomic. Slot u % TQ serves iteration u >= r0 and is fetched
    // at iteration u - TQ (before the loop if that is < 0).
    if (grp == NG && lane == 1) {
#pragma unroll
        for (int j = 0; j < TQ; ++j)
            if (r0 + j < TQ) tq[(r0 + j) % TQ] = atomicAdd(tk, 1ull);
    }
    // x first (consumers: their fixed 16-weight groups, to registers; everyone: the bf16 copy for the escapes)
    uint4 xv[NJ][2];
    if (grp < NG) {
#pragma unroll
        for (int q = 0; q < NJ; ++q) {
            const uint4* p = reinterpret_cast<const uint4*>(x + 16 * (gt + q * GT));
            xv[q][0] = __ldg(p); xv[q][1] = __ldg(p + 1);
        }
    }
    for (int i = threadIdx.x; i < KK / 8; i += blockDim.x)
        reinterpret_cast<uint4*>(xb)[i] = __ldg(reinterpret_cast<const uint4*>(x) + i);
    if (threadIdx.x == 0) {
        for (int s = 0; s < NS; ++s) { mbar_init(full0 + 8 * s, 1); mbar_init(empty0 + 8 * s, WG); tick[s] = 0; }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (grp == NG) {  // producer warp: refills stages; finalizes each row (sum of the WG partials, escapes, store)
        // just before its stage is refilled (consumers only compute and release)
        auto finalize = [&](int f) {
            const int s = f % NS;
            mbar_wait(empty0 + 8 * s, (f / NS) & 1);
#ifdef G21_TRACE
            if (lane == 0 && f < 32 && blockIdx.x < 4) ((long long*)qg)[(blockIdx.x * 32 + f + NS) * 4 + 2] = clock64();
#endif
            const int row = rowid[s];
            float r = lane < WG ? red[s * WG + lane] : 0.f;
#pragma unroll
            for (int o = 16; o; o >>= 1) r += __shfl_xor_sync(0xffffffffu, r, o);
            if (lane == 2) {
                const float2* he = reinterpret_cast<const float2*>(ring + s * RB + KK + KK / 2);
                for (int e = 0; e < ESC; ++e) {
                    const float2 h = he[e];
                    if (h.y == 0x1p-15f) break;  // slots fill in order; empty = 2^-15
                    const float xe = __bfloat162float(xb[__float_as_int(h.x)]);
                    r = fmaf(xe, h.y, fmaf(xe, -0x1p-15f, r));
                }
                out[row] = __float2bfloat16(r * scale);
#ifndef G21_TRACE
                if (qg) asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(qg + (row >= nq)) : "memory");
#endif
            }
            __syncwarp();
        };
        // ticket slot j of the queue is used at iterations it = j (mod TQ): unrolled so each slot keeps its own
        // register (a shifted queue would wait for every pending atomic). All of a call's tickets lie in one window
        // of NT: its base comes from the first ticket (one 64-bit division per CTA).
        unsigned long long tbase = ~0ull;
        int it = 0;
        bool done = false;
        while (!done) {
#pragma unroll
            for (int j = 0; j < TQ; ++j, ++it) {
                const int s = it % NS;
#ifdef G21_TRACE
                long long c0_ = clock64();
#endif
                if (it >= NS) finalize(it - NS);
#ifdef G21_TRACE
                long long c1_ = clock64();
                if (lane == 0 && it < 32 && blockIdx.x < 4) { ((long long*)qg)[(blockIdx.x * 32 + it) * 4] = c0_; ((long long*)qg)[(blockIdx.x * 32 + it) * 4 + 1] = c1_; }
#endif
                int row = blockIdx.x + it * G;
                if (lane == 1 && it >= r0) {
                    if (tbase == ~0ull) tbase = tq[j] - tq[j] % NT;
                    const int d = (int)(tq[j] - tbase);
                    row = d < nd ? r0 * G + d : n;
                }
                if (lane == 1 && row < n && it + TQ >= r0) tq[j] = atomicAdd(tk, 1ull);
                row = __shfl_sync(0xffffffffu, row, 1);
                if (row >= n) { done = true; break; }
                if (lane == 0) {
                    const unsigned d = (unsigned)__cvta_generic_to_shared(ring + s * RB), mb = full0 + 8 * s;
                    rowid[s] = row;
                    mbar_expect(mb, RB);
                    bulk(d, S + (long)row * KK, KK, mb);
                    bulk(d + KK, N + (long)row * (KK / 2), KK / 2, mb);
                    bulk(d + KK + KK / 2, H + (long)row * ESC, ESC * 8, mb);
                }
                __syncwarp();
#ifdef G21_TRACE
                if (lane == 0 && it < 32 && blockIdx.x < 4) ((long long*)qg)[(blockIdx.x * 32 + it) * 4 + 3] = clock64();
#endif
            }
        }
        for (int f = it >= NS ? it - NS + 1 : 0; f < it; ++f) finalize(f);  // (it - NS was finalized in the loop)
        if (lane == 0)
            for (int j = 0; j < NG; ++j) {  // one stop per group (those stages' last rows are finalized)
                const int s = (it + j) % NS;
                rowid[s] = -1;
                mbar_arrive(full0 + 8 * s);
            }
        return;
    }
    float xr[NJ][16];
#pragma unroll
    for (int q = 0; q < NJ; ++q) {
        const unsigned v[8] = {xv[q][0].x, xv[q][0].y, xv[q][0].z, xv[q][0].w, xv[q][1].x, xv[q][1].y, xv[q][1].z, xv[q][1].w};
#pragma unroll
        for (int j = 0; j < 8; ++j) { xr[q][2 * j] = __uint_as_float(v[j] << 16); xr[q][2 * j + 1] = __uint_as_float(v[j] & 0xffff0000u); }
    }
    for (int it = grp;; it += NG) {
        const int s = it % NS;
        mbar_wait(full0 + 8 * s, (it / NS) & 1);
        if (rowid[s] < 0) break;
        const unsigned char* st = ring + s * RB;
        float v = 0.f;
#ifndef G21_NOMATH
#pragma unroll
        for (int q = 0; q < NJ; ++q) {
            const int j = gt + q * GT;
            const uint4 b = *reinterpret_cast<const uint4*>(st + 16 * j);
            const uint2 c = *reinterpret_cast<const uint2*>(st + KK + 8 * j);
            v += dot16(b, c, xr[q]);
        }
#pragma unroll
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
#endif
        if (lane == 0) {
            red[s * WG + wi] = v;
            mbar_arrive(empty0 + 8 * s);  // release: orders the partial before the producer's finalize
        }
        __syncwarp();
    }
}
}  // namespace g21

void g21_gemv(torch::Tensor s, torch::Tensor nb, torch::Tensor h, torch::Tensor x, torch::Tensor out, double scale,
              int64_t wg, int64_t ng, int64_t ns, int64_t blocks, torch::Tensor qg, int64_t nq, torch::Tensor tk) {
    const int k = x.size(1), n = out.size(1);
    auto st = c10::cuda::getCurrentCUDAStream();
    TORCH_CHECK(x.size(0) == 1, "g21: m == 1");
    unsigned* q = qg.numel() ? (unsigned*)qg.data_ptr() : nullptr;
#define G21_CASE(K_, W_, G_, S_) \
    if (k == K_ && wg == W_ && ng == G_ && ns == S_) { \
        constexpr int RB = K_ + K_ / 2 + g21::ESC * 8; \
        const int smem = ((S_ * RB + K_ * 2 + S_ * W_ * 4 + S_ * 8 + 7) & ~7) + 2 * S_ * 8; \
        static bool set = false; \
        if (!set) { \
            cudaFuncSetAttribute(g21::gemv<K_, W_, G_, S_>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
            cudaFuncSetAttribute(g21::gemv<K_, W_, G_, S_>, cudaFuncAttributePreferredSharedMemoryCarveout, 100); \
            set = true; } \
        g21::gemv<K_, W_, G_, S_><<<blocks, (G_ * W_ + 1) * 32, smem, st>>>((const unsigned char*)s.data_ptr(), \
            (const unsigned char*)nb.data_ptr(), (const float2*)h.data_ptr(), (const __nv_bfloat16*)x.data_ptr(), \
            (__nv_bfloat16*)out.data_ptr(), n, (float)scale, q, (int)nq, (unsigned long long*)tk.data_ptr()); return; }
    G21_CONFIGS
    TORCH_CHECK(false, "g21: no config");
}
