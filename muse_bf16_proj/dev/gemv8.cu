// Skinny GEMM out[m, n] = x[m, k] @ W[n, k]^T for small m (gemv8): BF16 x, BF16 W (NA = 1) or the router's BF16
// hi + lo pair (NA = 2, one FP32 accumulator), FP32 accumulate, BF16 or FP32 out. Swap-AB mma.sync m16n8k16:
// A = 16 weight rows (16-B per lane from a per-warp cp.async ring in smem), B = 8 tokens of x via ldmatrix.
// Work unit = one block = (row group of NWR stripes, token group, K chunk KB = k / cs); many units, scheduled by the
// hardware (no clusters, no persistence), so SMs that finish early take more. W is repacked so every unit's
// weights are one contiguous run, units in launch order: [row group][K chunk][stripe][k-tile][NA][512 B].
// x[token group, K chunk] is staged in smem (only rows < m; others read a zero line). Split K: warps of a block
// reduce in smem; blocks add FP32 partials into a workspace (red.global.add) and the last block of a row group
// (counter) writes the output and re-zeroes workspace and counter: graph-replayable, no memset.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

namespace g8 {

__device__ __forceinline__ void mma(float* d, const uint4& a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint4 ldw(const uint4* p) {
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}

template <int MT, int NWR, int NWK, int U, int NA, bool F32OUT>
__global__ void __launch_bounds__(NWR * NWK * 32) gemv(const uint4* __restrict__ w, const __nv_bfloat16* __restrict__ x,
                                                       void* __restrict__ out, float* __restrict__ ws, int* __restrict__ cnt, int m, int n, int k, int cs) {
    constexpr int TOK = MT * 8, RB = NWR * 16, NT = NWR * NWK * 32;
    extern __shared__ __align__(16) unsigned char smem[];
    __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(smem);
    float* red = reinterpret_cast<float*>(smem);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, wr = warp % NWR, wk = warp / NWR;
    const int nrg = n / RB, tg = blockIdx.x / (nrg * cs), rg = (blockIdx.x / cs) % nrg, ks = blockIdx.x % cs;
    // x rows actually present in this token group (rows past m read a shared 16-B zero line instead)
    const int mr = min(TOK, m - tg * TOK);
    const int KB = k / cs, k0 = ks * KB, XS = KB + 8;  // smem row stride (elements): +16 B breaks bank conflicts
    const int XBYTES = max(mr * XS * 2 + 16, NWK * TOK * RB * 4);
    if (threadIdx.x == 0) *reinterpret_cast<uint4*>(smem + mr * XS * 2) = make_uint4(0, 0, 0, 0);
    for (int i = threadIdx.x; i < mr * KB / 8; i += NT) {
        const int r = i / (KB / 8), c = (i % (KB / 8)) * 8, tok = tg * TOK + r;
        const unsigned sa = (unsigned)__cvta_generic_to_shared(xs + r * XS + c);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(sa), "l"(x + (size_t)tok * k + k0 + c));
    }
    asm volatile("cp.async.commit_group;");
    const int KT = KB / 16 / NWK, kt16 = k >> 4, s = rg * NWR + wr;
    const int tl0 = wk * KT;  // first k-tile of this warp, local to the block's K range
    const uint4* wp = w + (((size_t)(rg * cs + ks) * NWR + wr) * (KB >> 4) + tl0) * NA * 32 + lane;
    constexpr int tstride = NA * 32;
    uint4* ring = reinterpret_cast<uint4*>(smem + XBYTES) + (size_t)warp * U * NA * 32 + lane;
#pragma unroll
    for (int u = 0; u < U; ++u) {
#pragma unroll
        for (int q = 0; q < NA; ++q)
            if (u < KT) {
                const unsigned sa = (unsigned)__cvta_generic_to_shared(ring + (u * NA + q) * 32);
                asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;" ::"r"(sa), "l"(wp + u * tstride + q * 32));
            }
        asm volatile("cp.async.commit_group;");
    }
    float acc[MT][4];
#pragma unroll
    for (int j = 0; j < MT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    asm volatile("cp.async.wait_group %0;" ::"n"(U));
    __syncthreads();
    // ldmatrix row address: lane l -> matrix l/8, row l%8; matrices (j, k lo), (j, k hi), (j+1, k lo), (j+1, k hi)
    const int lr = lane & 7, lm = lane >> 3;
    const unsigned xs0 = (unsigned)__cvta_generic_to_shared(xs), xzero = xs0 + 2u * (unsigned)(mr * XS);
    unsigned xb[(MT + 1) / 2];  // per ldmatrix: this lane's row address at the warp's first k-tile, or the zero line
    bool xvalid[(MT + 1) / 2];
#pragma unroll
    for (int jj = 0; jj < (MT + 1) / 2; ++jj) {
        const int row = jj * 16 + (lm >> 1) * 8 + lr;
        xvalid[jj] = row < mr;
        xb[jj] = xs0 + 2u * (unsigned)(row * XS + (lm & 1) * 8 + tl0 * 16);
    }
    for (int t = 0; t < KT; ++t) {
        const int slot = t % U;
        asm volatile("cp.async.wait_group %0;" ::"n"(U - 1));
        uint4 a[NA];
#pragma unroll
        for (int q = 0; q < NA; ++q) a[q] = ring[(slot * NA + q) * 32];
#pragma unroll
        for (int j = 0; j < MT; j += 2) {
            unsigned b0, b1, b2 = 0, b3 = 0;
            const unsigned addr = xvalid[j / 2] ? xb[j / 2] + 32u * t : xzero;
            if (j + 1 < MT)
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(addr));
            else
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(b0), "=r"(b1) : "r"(addr));
#pragma unroll
            for (int q = 0; q < NA; ++q) {
                mma(acc[j], a[q], b0, b1);
                if (j + 1 < MT) mma(acc[j + 1], a[q], b2, b3);
            }
        }
        if (t + U < KT) {
#pragma unroll
            for (int q = 0; q < NA; ++q) {
                const unsigned sa = (unsigned)__cvta_generic_to_shared(ring + (slot * NA + q) * 32);
                asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;" ::"r"(sa),
                             "l"(wp + (t + U) * tstride + q * 32));
            }
        }
        asm volatile("cp.async.commit_group;");
    }
    __syncthreads();  // x no longer needed: reuse smem for the reduction
    const int g = lane >> 2, tq = lane & 3;
#pragma unroll
    for (int j = 0; j < MT; ++j)
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int row = wr * 16 + g + ((r & 2) ? 8 : 0), tok = j * 8 + 2 * tq + (r & 1);
            red[(wk * TOK + tok) * RB + row] = acc[j][r];
        }
    __syncthreads();
    constexpr int E = TOK * RB;
    if (NWK > 1) {
        for (int e = threadIdx.x; e < E; e += NT) {
            float v = red[e];
#pragma unroll
            for (int q = 1; q < NWK; ++q) v += red[q * E + e];
            red[e] = v;
        }
    }
    auto store = [&](int e, float v) {
        const int tok = tg * TOK + e / RB, row = rg * RB + e % RB;
        if (tok < m) {
            if (F32OUT) reinterpret_cast<float*>(out)[(size_t)tok * n + row] = v;
            else reinterpret_cast<__nv_bfloat16*>(out)[(size_t)tok * n + row] = __float2bfloat16(v);
        }
    };
    if (cs == 1) {
        if (NWK > 1) __syncthreads();
        for (int e = threadIdx.x; e < E; e += NT) store(e, red[e]);
        return;
    }
    // split K: add this block's partials, the last block of the (row group, token group) finishes
    __shared__ int last;
    const int ci = tg * nrg + rg;
    for (int e = threadIdx.x; e < E; e += NT) {
        const int tok = tg * TOK + e / RB, row = rg * RB + e % RB;
        if (tok < m) atomicAdd(ws + (size_t)tok * n + row, red[e]);
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        const int prev = atomicAdd(cnt + ci, 1);
        last = prev == cs - 1;
        if (last) cnt[ci] = 0;
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    for (int e = threadIdx.x; e < E; e += NT) {
        const int tok = tg * TOK + e / RB, row = rg * RB + e % RB;
        if (tok < m) {
            float* p = ws + (size_t)tok * n + row;
            store(e, __ldcg(p));
            __stcg(p, 0.f);
        }
    }
}

template <int MT, int NWR, int NWK, int U, int NA, bool F32OUT>
int launch(const torch::Tensor& w, const torch::Tensor& x, torch::Tensor& out, torch::Tensor& ws, torch::Tensor& cnt, int n, int cs, int tgs, bool query) {
    const int m = (int)x.size(0), k = (int)x.size(1);
    constexpr int TOK = MT * 8, RB = NWR * 16, NT = NWR * NWK * 32;
    TORCH_CHECK(n % RB == 0 && k % (cs * 16 * NWK) == 0 && tgs * TOK >= m, "g8: bad shape/config");
    const int KB = k / cs;
    const int mr = m < TOK ? m : TOK;
    const size_t sx = (size_t)mr * (KB + 8) * 2 + 16, sr = (size_t)NWK * TOK * RB * 4;
    const size_t sm = (sx > sr ? sx : sr) + (size_t)NWR * NWK * U * NA * 512;
    auto kern = gemv<MT, NWR, NWK, U, NA, F32OUT>;
    static size_t set = 0;  // per instantiation: the largest dynamic smem allowed so far
    if (sm > set) {
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
        set = sm;
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(cs * (n / RB) * tgs);
    cfg.blockDim = dim3(NT);
    cfg.dynamicSmemBytes = sm;
    cfg.stream = c10::cuda::getCurrentCUDAStream();
    cfg.numAttrs = 0;
    if (query) {
        int nb = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, kern, NT, sm);
        return nb;
    }
    cudaLaunchKernelEx(&cfg, kern, reinterpret_cast<const uint4*>(w.data_ptr()),
                       reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), out.data_ptr(), ws.data_ptr<float>(), cnt.data_ptr<int>(), m, n, k, cs);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return 0;
}
}  // namespace g8

#define G8_CASE(MT, NWR, NWK, U, NA)                                                     \
    if (mt == MT && nwr == NWR && nwk == NWK && u == U && na == NA) {                    \
        if (f32) return g8::launch<MT, NWR, NWK, U, NA, true>(w, x, out, ws, cnt, n, cs, tgs, query);   \
        else return g8::launch<MT, NWR, NWK, U, NA, false>(w, x, out, ws, cnt, n, cs, tgs, query); \
                                                                                  \
    }

static int64_t g8_run(torch::Tensor w, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t n, int64_t mt, int64_t nwr,
                      int64_t nwk, int64_t u, int64_t na, int64_t cs, int64_t tgs, bool query) {
    const bool f32 = out.scalar_type() == torch::kFloat32;
    G8_CONFIGS
    TORCH_CHECK(false, "g8: no such config");
    return 0;
}
void g8_gemv(torch::Tensor w, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t n, int64_t mt, int64_t nwr, int64_t nwk,
             int64_t u, int64_t na, int64_t cs, int64_t tgs) {
    g8_run(w, x, out, ws, cnt, n, mt, nwr, nwk, u, na, cs, tgs, false);
}
int64_t g8_occ(torch::Tensor w, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t n, int64_t mt, int64_t nwr, int64_t nwk,
               int64_t u, int64_t na, int64_t cs, int64_t tgs) {
    return g8_run(w, x, out, ws, cnt, n, mt, nwr, nwk, u, na, cs, tgs, true);
}
