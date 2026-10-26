// gemv17: BF16 out[m, n] = x[m, K] @ W[n, K]^T for small m, router2's structure: block = R whole rows, T threads,
// thread t owns k = 8 (t + T j), j < J = K / (8 T): its x slice goes straight to registers, all R * J uint4 W loads
// are issued up front, FP32 FMA, warp shuffle + one smem pass. No split K, no workspace.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

namespace g17 {
__device__ __forceinline__ uint4 ldw(const uint4* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
__device__ __forceinline__ float lo(unsigned v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float hi(unsigned v) { return __uint_as_float(v & 0xffff0000u); }

template <int M, int R, int T, int K>
__global__ void __launch_bounds__(T) gemv(const uint4* __restrict__ w, const __nv_bfloat16* __restrict__ x,
                                          __nv_bfloat16* __restrict__ out, int m, int n) {
    constexpr int J = K / (8 * T), NW = T / 32;
    static_assert(K % (8 * T) == 0, "k split");
    __shared__ float red[NW][R * M];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5, r0 = blockIdx.x * R;
    uint4 xr[M][J];
#pragma unroll
    for (int i = 0; i < M; ++i)
#pragma unroll
        for (int j = 0; j < J; ++j)
            xr[i][j] = i < m ? __ldg(reinterpret_cast<const uint4*>(x + (size_t)i * K) + t + T * j) : make_uint4(0, 0, 0, 0);
    uint4 a[R][J];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int j = 0; j < J; ++j) a[r][j] = ldw(w + (size_t)(r0 + r) * (K / 8) + t + T * j);
    float acc[R][M];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int i = 0; i < M; ++i) acc[r][i] = 0.f;
#pragma unroll
    for (int j = 0; j < J; ++j)
#pragma unroll
        for (int i = 0; i < M; ++i) {
            const unsigned xv[4] = {xr[i][j].x, xr[i][j].y, xr[i][j].z, xr[i][j].w};
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const unsigned wv[4] = {a[r][j].x, a[r][j].y, a[r][j].z, a[r][j].w};
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    acc[r][i] = fmaf(lo(wv[q]), lo(xv[q]), acc[r][i]);
                    acc[r][i] = fmaf(hi(wv[q]), hi(xv[q]), acc[r][i]);
                }
            }
        }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int i = 0; i < M; ++i) {
            float v = acc[r][i];
#pragma unroll
            for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0) red[warp][r * M + i] = v;
        }
    __syncthreads();
    for (int e = t; e < R * M; e += T) {
        const int r = e / M, i = e % M;
        float s = 0.f;
#pragma unroll
        for (int q = 0; q < NW; ++q) s += red[q][e];
        if (i < m) out[(size_t)i * n + r0 + r] = __float2bfloat16(s);
    }
}
}  // namespace g17

#define G17_CASE(M, R, T, K)                                                                                        \
    if (mp == M && rr == R && tt == T && k == K) {                                                                  \
        g17::gemv<M, R, T, K><<<n / R, T, 0, c10::cuda::getCurrentCUDAStream()>>>(                                   \
            reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),     \
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), m, n);                                                \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                             \
        return;                                                                                                     \
    }

void g17_gemv(torch::Tensor w, torch::Tensor x, torch::Tensor out, int64_t mp, int64_t rr, int64_t tt) {
    const int m = (int)x.size(0), k = (int)x.size(1), n = (int)out.size(1);
    TORCH_CHECK(m <= mp && n % rr == 0, "g17: shape");
    G17_CONFIGS
    TORCH_CHECK(false, "g17: no such config");
}
