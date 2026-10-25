// gemv9: CUDA-core stream-K GEMV for small m (out[m, n] = x[m, k] @ W[n, k]^T), W plain row-major BF16.
// Tile = R rows x 256 k. Tiles are ordered (row group, k tile); warp w of the grid takes the contiguous tile
// range [w T / nw, (w+1) T / nw), so every SM reads the same number of bytes. Lane l owns k = kt*256 + l*8..+8.
// A register ring keeps U tiles of W in flight across row boundaries. At a row-group boundary the warp reduces
// its FP32 partials (shuffles); a row group done by one warp is written directly, else partials go to an FP32
// workspace with atomics and the warp completing the group's k tiles (counter) writes BF16 and re-zeroes.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
namespace g9 {
__device__ __forceinline__ uint4 ldw(const uint4* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
__device__ __forceinline__ float lo(unsigned u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float hi(unsigned u) { return __uint_as_float(u & 0xffff0000u); }
__device__ __forceinline__ void dot8(float& a, const uint4& w, const uint4& x) {
    a = fmaf(lo(w.x), lo(x.x), a); a = fmaf(hi(w.x), hi(x.x), a);
    a = fmaf(lo(w.y), lo(x.y), a); a = fmaf(hi(w.y), hi(x.y), a);
    a = fmaf(lo(w.z), lo(x.z), a); a = fmaf(hi(w.z), hi(x.z), a);
    a = fmaf(lo(w.w), lo(x.w), a); a = fmaf(hi(w.w), hi(x.w), a);
}
template <int M, int R, int U>
__global__ void __launch_bounds__(512) gemv(const uint4* __restrict__ W, const uint4* __restrict__ x,
                                            __nv_bfloat16* __restrict__ out, float* __restrict__ ws,
                                            int* __restrict__ cnt, int n, int k, int lkt) {
    const int lane = threadIdx.x & 31;
    const long gw = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const long nw = ((long)gridDim.x * blockDim.x) >> 5;
    const int KT = 1 << lkt, k8 = k >> 3;
    const long T = (long)(n / R) << lkt;
    long t0 = gw * T / nw, t1 = (gw + 1) * T / nw;
    if (t0 >= t1) t0 = t1 = T;  // (stays for the block barrier)
    uint4 buf[U][R];
    auto load = [&](int u, long t) {
        const long rg = t >> lkt; const int kt = (int)(t & (KT - 1));
        const uint4* p = W + (rg * R) * k8 + kt * 32 + lane;
#pragma unroll
        for (int r = 0; r < R; ++r) buf[u][r] = ldw(p + (long)r * k8);
    };
#pragma unroll
    for (int u = 0; u < U; ++u) load(u, t0 + u < t1 ? t0 + u : (t1 > 0 ? t1 - 1 : 0));
    extern __shared__ uint4 xs[];  // x[M][k], loaded after the first W loads are in flight
    for (int i = threadIdx.x; i < M * k8; i += blockDim.x) xs[i] = __ldg(x + i);
    __syncthreads();
    float acc[M][R];
#pragma unroll
    for (int i = 0; i < M; ++i)
#pragma unroll
        for (int r = 0; r < R; ++r) acc[i][r] = 0.f;
    long cur = t0 >> lkt; int ntl = 0;
    auto flush = [&]() {
#pragma unroll
        for (int i = 0; i < M; ++i)
#pragma unroll
            for (int r = 0; r < R; ++r) {
                float v = acc[i][r];
#pragma unroll
                for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
                acc[i][r] = v;
            }
        if (ntl == KT) {
#pragma unroll
            for (int i = 0; i < M; ++i)
#pragma unroll
                for (int r = 0; r < R; ++r)
                    if (lane == i * R + r) out[(long)i * n + cur * R + r] = __float2bfloat16(acc[i][r]);
        } else {
#pragma unroll
            for (int i = 0; i < M; ++i)
#pragma unroll
                for (int r = 0; r < R; ++r)
                    if (lane == i * R + r) atomicAdd(ws + (long)i * n + cur * R + r, acc[i][r]);
            __threadfence();
            int last = 0;
            if (lane == 0) last = atomicAdd(cnt + cur, ntl) + ntl == KT;
            last = __shfl_sync(0xffffffffu, last, 0);
            if (last) {
                __threadfence();
#pragma unroll
                for (int i = 0; i < M; ++i)
#pragma unroll
                    for (int r = 0; r < R; ++r)
                        if (lane == i * R + r) {
                            float* p = ws + (long)i * n + cur * R + r;
                            out[(long)i * n + cur * R + r] = __float2bfloat16(__ldcg(p));
                            *p = 0.f;
                        }
                if (lane == 0) cnt[cur] = 0;
            }
        }
#pragma unroll
        for (int i = 0; i < M; ++i)
#pragma unroll
            for (int r = 0; r < R; ++r) acc[i][r] = 0.f;
        ntl = 0;
    };
    if (t0 == T) return;
    for (long t = t0; t < t1; t += U) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const long tt = t + u;
            if (tt < t1) {
                const long rg = tt >> lkt;
                if (rg != cur) { flush(); cur = rg; }
                const int kt = (int)(tt & (KT - 1));
                uint4 xv[M];
#pragma unroll
                for (int i = 0; i < M; ++i) xv[i] = xs[i * k8 + kt * 32 + lane];
#pragma unroll
                for (int i = 0; i < M; ++i)
#pragma unroll
                    for (int r = 0; r < R; ++r) dot8(acc[i][r], buf[u][r], xv[i]);
                ++ntl;
                if (tt + U < t1) load(u, tt + U);
            }
        }
    }
    flush();
}
}  // namespace g9
void g9_gemv(torch::Tensor w, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt,
             int64_t r, int64_t u, int64_t blocks, int64_t threads) {
    const int m = x.size(0), k = x.size(1), n = out.size(1);
    int lkt = 0; while ((256 << lkt) < k) ++lkt;
    auto s = c10::cuda::getCurrentCUDAStream();
#define G9_CASE(M_, R_, U_) \
    if (m == M_ && r == R_ && u == U_) { if (m * k * 2 > 48 * 1024) cudaFuncSetAttribute(g9::gemv<M_, R_, U_>, \
        cudaFuncAttributeMaxDynamicSharedMemorySize, m * k * 2); \
        g9::gemv<M_, R_, U_><<<blocks, threads, m * k * 2, s>>>((const uint4*)w.data_ptr(), \
        (const uint4*)x.data_ptr(), (__nv_bfloat16*)out.data_ptr(), ws.data_ptr<float>(), cnt.data_ptr<int>(), n, k, lkt); return; }
    G9_CONFIGS
    TORCH_CHECK(false, "g9: no config");
}
