// Router at m <= 8 in true FP32 (as SGLang's F.linear): out[m, 256] = x[m, k] (BF16 -> FP32) @ W[256, k]^T (FP32).
// Block = (RB rows, K chunk of KB): x[:, chunk] staged in smem as FP32; warp w takes rows r0 + w*RW .. +RW, each
// lane FP32 float4 loads of W at k = 4 (lane + 32 j) (all KB/128 loads in flight), FMA against smem x, warp
// shuffle reduce. Split K: FP32 red.global into a workspace, the last block of a row block writes the output and
// re-zeroes workspace + counter.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

namespace pr {
template <int M, int KB, int RW, int NWARP>
__global__ void __launch_bounds__(NWARP * 32) router(const float4* __restrict__ w, const __nv_bfloat16* __restrict__ x,
                                                     float* __restrict__ out, float* __restrict__ ws,
                                                     int* __restrict__ cnt, int m, int n, int k) {
    constexpr int J = KB / 128, RB = NWARP * RW;
    __shared__ __align__(16) float xs[M][KB];
    __shared__ int last;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int r0 = blockIdx.x * RB + warp * RW, k0 = blockIdx.y * KB;
    constexpr int XV = (M * KB / 8 + NWARP * 32 - 1) / (NWARP * 32);
    uint4 xr[XV];
#pragma unroll
    for (int q = 0; q < XV; ++q) {
        const int i = threadIdx.x + q * NWARP * 32, t = i / (KB / 8), c = (i % (KB / 8)) * 8;
        xr[q] = make_uint4(0, 0, 0, 0);
        if (i < M * KB / 8 && t < m) xr[q] = __ldcs(reinterpret_cast<const uint4*>(x + (size_t)t * k + k0 + c));
    }
    float4 a[RW][J];
#pragma unroll
    for (int r = 0; r < RW; ++r)
#pragma unroll
        for (int j = 0; j < J; ++j) a[r][j] = __ldcs(w + ((size_t)(r0 + r) * k + k0) / 4 + lane + 32 * j);
#pragma unroll
    for (int q = 0; q < XV; ++q) {
        const int i = threadIdx.x + q * NWARP * 32, t = i / (KB / 8), c = (i % (KB / 8)) * 8;
        if (i >= M * KB / 8) break;
        const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&xr[q]);
        float v[8];
#pragma unroll
        for (int e = 0; e < 4; ++e) { const float2 f = __bfloat1622float2(h[e]); v[2 * e] = f.x; v[2 * e + 1] = f.y; }
        *reinterpret_cast<float4*>(&xs[t][c]) = make_float4(v[0], v[1], v[2], v[3]);
        *reinterpret_cast<float4*>(&xs[t][c + 4]) = make_float4(v[4], v[5], v[6], v[7]);
    }
    __syncthreads();
    float acc[RW][M];
#pragma unroll
    for (int r = 0; r < RW; ++r)
#pragma unroll
        for (int t = 0; t < M; ++t) acc[r][t] = 0.f;
#pragma unroll
    for (int j = 0; j < J; ++j)
#pragma unroll
        for (int t = 0; t < M; ++t) {
            const float4 xv = *reinterpret_cast<const float4*>(&xs[t][4 * (lane + 32 * j)]);
#pragma unroll
            for (int r = 0; r < RW; ++r) {
                acc[r][t] = fmaf(a[r][j].x, xv.x, acc[r][t]);
                acc[r][t] = fmaf(a[r][j].y, xv.y, acc[r][t]);
                acc[r][t] = fmaf(a[r][j].z, xv.z, acc[r][t]);
                acc[r][t] = fmaf(a[r][j].w, xv.w, acc[r][t]);
            }
        }
#pragma unroll
    for (int r = 0; r < RW; ++r)
#pragma unroll
        for (int t = 0; t < M; ++t)
#pragma unroll
            for (int o = 16; o; o >>= 1) acc[r][t] += __shfl_xor_sync(0xffffffffu, acc[r][t], o);
    const bool split = gridDim.y > 1;
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < RW; ++r)
#pragma unroll
            for (int t = 0; t < M; ++t)
                if (t < m) {
                    if (split) atomicAdd(ws + (size_t)t * n + r0 + r, acc[r][t]);
                    else out[(size_t)t * n + r0 + r] = acc[r][t];
                }
    }
    if (!split) return;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        const int prev = atomicAdd(cnt + blockIdx.x, 1);
        last = prev == (int)gridDim.y - 1;
        if (last) cnt[blockIdx.x] = 0;
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    for (int i = threadIdx.x; i < m * RB; i += NWARP * 32) {
        const int t = i / RB, row = blockIdx.x * RB + i % RB;
        float* p = ws + (size_t)t * n + row;
        out[(size_t)t * n + row] = __ldcg(p);
        __stcg(p, 0.f);
    }
}
}  // namespace pr

#define PR_CASE(M, KB, RW, NWARP)                                                                                   \
    if (mp == M && kb == KB && rw == RW && nwarp == NWARP) {                                                        \
        dim3 grid(n / (RW * NWARP), k / KB);                                                                        \
        pr::router<M, KB, RW, NWARP><<<grid, NWARP * 32, 0, c10::cuda::getCurrentCUDAStream()>>>(                   \
            reinterpret_cast<const float4*>(w.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),    \
            out.data_ptr<float>(), ws.data_ptr<float>(), cnt.data_ptr<int>(), m, n, k);                             \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                             \
        return;                                                                                                     \
    }

void pj_router(torch::Tensor w, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t mp,
               int64_t kb, int64_t rw, int64_t nwarp) {
    const int m = (int)x.size(0), n = (int)w.size(0), k = (int)w.size(1);
    TORCH_CHECK(m <= mp, "pr: m");
    PR_CONFIGS
    TORCH_CHECK(false, "pr: no such config");
}
