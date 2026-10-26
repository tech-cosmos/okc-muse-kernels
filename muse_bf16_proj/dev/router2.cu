// Router at m <= 8 in true FP32, no split K: block = R whole rows (k = 8192), T threads; thread t owns k positions
// 4 (t + T j), j < J = k / (4 T), so its x slice goes straight to registers (no smem staging) and all R * J float4 W
// loads are issued up front. Warp shuffle reduce, then one smem pass over the block's warps. No workspace/atomics.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

namespace pr2 {
template <int M, int R, int T, int K>
__global__ void __launch_bounds__(T) router(const float4* __restrict__ w, const __nv_bfloat16* __restrict__ x,
                                            float* __restrict__ out, int m, int n) {
    constexpr int J = K / (4 * T), NW = T / 32;
    static_assert(K % (4 * T) == 0, "k split");
    __shared__ float red[NW][R * M];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5, r0 = blockIdx.x * R;
    uint2 xr[M][J];  // 4 BF16 per (token, j)
#pragma unroll
    for (int i = 0; i < M; ++i)
#pragma unroll
        for (int j = 0; j < J; ++j)
            xr[i][j] = i < m ? __ldg(reinterpret_cast<const uint2*>(x + (size_t)i * K) + t + T * j) : make_uint2(0, 0);
    float4 a[R][J];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int j = 0; j < J; ++j) a[r][j] = __ldcs(w + (size_t)(r0 + r) * (K / 4) + t + T * j);
    float acc[R][M];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int i = 0; i < M; ++i) acc[r][i] = 0.f;
#pragma unroll
    for (int j = 0; j < J; ++j)
#pragma unroll
        for (int i = 0; i < M; ++i) {
            const float x0 = __uint_as_float(xr[i][j].x << 16), x1 = __uint_as_float(xr[i][j].x & 0xffff0000u);
            const float x2 = __uint_as_float(xr[i][j].y << 16), x3 = __uint_as_float(xr[i][j].y & 0xffff0000u);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                acc[r][i] = fmaf(a[r][j].x, x0, acc[r][i]);
                acc[r][i] = fmaf(a[r][j].y, x1, acc[r][i]);
                acc[r][i] = fmaf(a[r][j].z, x2, acc[r][i]);
                acc[r][i] = fmaf(a[r][j].w, x3, acc[r][i]);
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
    if (t < R * M) {
        const int r = t / M, i = t % M;
        float s = 0.f;
#pragma unroll
        for (int q = 0; q < NW; ++q) s += red[q][t];
        if (i < m) out[(size_t)i * n + r0 + r] = s;
    }
}
}  // namespace pr2

#define PR2_CASE(M, R, T)                                                                                           \
    if (mp == M && rr == R && tt == T) {                                                                            \
        pr2::router<M, R, T, 8192><<<n / R, T, 0, c10::cuda::getCurrentCUDAStream()>>>(                             \
            reinterpret_cast<const float4*>(w.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),    \
            out.data_ptr<float>(), m, n);                                                                           \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                             \
        return;                                                                                                     \
    }

void pj_router2(torch::Tensor w, torch::Tensor x, torch::Tensor out, int64_t mp, int64_t rr, int64_t tt) {
    const int m = (int)x.size(0), n = (int)w.size(0);
    TORCH_CHECK(m <= mp && w.size(1) == 8192, "pr2: shape");
    PR2_CONFIGS
    TORCH_CHECK(false, "pr2: no such config");
}
