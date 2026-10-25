# okc: problem=muse_bf16_proj
"""Dense projections of Muse Spark Open: out = x @ W[:n].T (BF16; the router FP32).

Router (FP32 weights, FP32 output): up to 16 tokens a native true-FP32 split-K kernel (FP32 weights, FP32 FMA,
FP32 split partials; SGLang runs F.linear in FP32 up to 8 tokens, BF16 hi/lo above), above that SGLang's two
BF16 hi/lo GEMMs with FP32 accumulation. BF16 projections: cuBLAS (as SGLang's bf16_gemm_dispatch on H200).
Native kernels are built once at import (load_inline); if the build fails everything falls back to torch.
"""

from typing import TypedDict

import torch
import torch.nn.functional as F


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [m, k]
    config: Config


_ROUTER_SRC = r'''// Router at m <= 8 in true FP32 (as SGLang's F.linear): out[m, 256] = x[m, k] (BF16 -> FP32) @ W[256, k]^T (FP32).
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
'''

# Router FP32 kernel configs (mp, kb, rw, nwarp), chosen on the H100 (dev/rt.py).
ROUTER_F32 = [(1, (1, 2048, 1, 8)), (8, (8, 512, 2, 8)), (16, (16, 512, 2, 8))]
_SRC = _ROUTER_SRC.replace("PR_CONFIGS", " ".join("PR_CASE(%d,%d,%d,%d)" % c for _, c in ROUTER_F32))
try:
    from torch.utils.cpp_extension import load_inline
    _mod = load_inline(
        name="muse_proj_v2",
        cpp_sources="void pj_router(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, "
                    "int64_t, int64_t, int64_t);",
        cuda_sources=_SRC, functions=["pj_router"],
        extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"], verbose=False)
except Exception as _e:  # noqa: BLE001
    print("muse_proj native build failed, using torch:", _e)
    _mod = None

ROUTER_FP32_MAX_TOKENS = 8


def prepare(weights):
    st = {name: w.contiguous() for name, w in weights.items()}
    r = weights["router"]
    hi = r.to(torch.bfloat16)
    lo = (r - hi.float()).to(torch.bfloat16)
    st["router_hi_t"] = hi.t().contiguous()
    st["router_lo_t"] = lo.t().contiguous()
    # Split-K workspace and per-row-block counters: always left zeroed by the kernels.
    st["ws"] = torch.zeros(32 * 8192, dtype=torch.float32, device=r.device)
    st["cnt"] = torch.zeros(1024, dtype=torch.int32, device=r.device)
    return st


def _router(x, st, m):
    if _mod is not None:
        for mmax, cfg in ROUTER_F32:
            if m <= mmax:
                out = torch.empty((m, 256), dtype=torch.float32, device=x.device)
                _mod.pj_router(st["router"], x, out, st["ws"], st["cnt"], *cfg)
                return out
    if m <= ROUTER_FP32_MAX_TOKENS:
        return F.linear(x.float(), st["router"])
    acc = torch.mm(x, st["router_hi_t"], out_dtype=torch.float32)
    acc += torch.mm(x, st["router_lo_t"], out_dtype=torch.float32)
    return acc


def kernel_fn(data, st):
    x, cfg = data["x"], data["config"]
    layer = cfg["layer"]
    if layer == "router":
        return _router(x, st, cfg["m"])
    return F.linear(x, st[layer][: cfg["n"]])
