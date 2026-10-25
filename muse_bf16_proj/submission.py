# okc: problem=muse_bf16_proj
"""Dense projections of Muse Spark Open: out = x @ W[:n].T (BF16; the router FP32).

Router (FP32 weights, FP32 output): up to 16 tokens a native true-FP32 split-K kernel (FP32 weights, FP32 FMA,
FP32 split partials; SGLang runs F.linear in FP32 up to 8 tokens), 17-32 tokens a native split-K BF16 hi + lo
tensor-core kernel with FP32 accumulation (SGLang's forward_split2 numerics), above that SGLang's two BF16 hi/lo
GEMMs with FP32 accumulation. BF16 projections: cuBLAS (as SGLang's bf16_gemm_dispatch on H200).
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
_GEMV_SRC = r'''// Split-K streaming skinny GEMM out[m, n] = x[m, k] @ W[n, k]^T, m <= 32, BF16 in, FP32 accumulate.
// Swap-AB mma.sync m16n8k16: A = a 16-row weight stripe, B = 8 tokens. W is repacked in prepare() into
// [stripe][k-tile][lane][8] (a 16x16 tile's A fragments = 512 contiguous bytes, 16 B per lane).
// Warp-level stream-K: the (stripe, k-tile) tiles are split evenly over all warps of the grid; each warp walks
// its contiguous run with U tiles in flight in registers (next batch loads while this one multiplies), x
// fragments straight from global (L1/L2). A stripe a warp covers whole is written directly; partial stripes are
// added into an FP32 workspace (red.global) and counted per stripe; the warp completing a stripe writes it and
// re-zeroes the workspace and counter, so the call is graph-replayable with no memset.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

namespace pj {
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
__device__ __forceinline__ unsigned ldx(const __nv_bfloat16* p) {
    return __ldg(reinterpret_cast<const unsigned*>(p));
}

template <int MT, int U, bool F32OUT>
__device__ __forceinline__ void flush(float (&acc)[MT][4], int stripe, bool whole, int ntiles, int kt16, int lane,
                                      int m, int n, void* out, float* ws, int* cnt) {
    const int g = lane >> 2, tq = lane & 3;
    if (whole) {
#pragma unroll
        for (int j = 0; j < MT; ++j)
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int row = stripe * 16 + g + ((r & 2) ? 8 : 0), tok = j * 8 + 2 * tq + (r & 1);
                if (tok < m) {
                    if (F32OUT) reinterpret_cast<float*>(out)[(size_t)tok * n + row] = acc[j][r];
                    else reinterpret_cast<__nv_bfloat16*>(out)[(size_t)tok * n + row] = __float2bfloat16(acc[j][r]);
                }
            }
        return;
    }
#ifdef PJ_NOFLUSH
    if (acc[0][0] != 12345.f) return;
#endif
#pragma unroll
    for (int j = 0; j < MT; ++j)
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int row = stripe * 16 + g + ((r & 2) ? 8 : 0), tok = j * 8 + 2 * tq + (r & 1);
            if (tok < m) atomicAdd(ws + (size_t)tok * n + row, acc[j][r]);
        }
    __threadfence();
    __syncwarp();
    int last = 0;
    if (lane == 0) {
        const int prev = atomicAdd(cnt + stripe, ntiles);
        last = prev + ntiles == kt16;
        if (last) cnt[stripe] = 0;
    }
    last = __shfl_sync(0xffffffffu, last, 0);
    if (!last) return;
    __threadfence();
    for (int i = lane; i < m * 16; i += 32) {
        const int tok = i >> 4, row = stripe * 16 + (i & 15);
        float* p = ws + (size_t)tok * n + row;
        const float v = __ldcg(p);
        __stcg(p, 0.f);
        if (F32OUT) reinterpret_cast<float*>(out)[(size_t)tok * n + row] = v;
        else reinterpret_cast<__nv_bfloat16*>(out)[(size_t)tok * n + row] = __float2bfloat16(v);
    }
}

template <int MT, int U, int WPB, bool F32OUT>
__global__ void __launch_bounds__(WPB * 32) gemv(const uint4* __restrict__ w, const __nv_bfloat16* __restrict__ x,
                                                 void* __restrict__ out, float* __restrict__ ws,
                                                 int* __restrict__ cnt, int m, int n, int k, int kx) {
    const int lane = threadIdx.x & 31;
    const int gw = blockIdx.x * WPB + (threadIdx.x >> 5), nwarps = gridDim.x * WPB;
    const int kt16 = k >> 4;
    const int T = (n >> 4) * kt16;
    const int t0 = (int)((long long)gw * T / nwarps), t1 = (int)((long long)(gw + 1) * T / nwarps);
    if (t0 >= t1) return;
    const int g = lane >> 2, tq = lane & 3;
    // x fragment base pointers per n8 tile (rows >= m read row 0 and are zeroed)
    const __nv_bfloat16* xr[MT];
    bool xv[MT];
#pragma unroll
    for (int j = 0; j < MT; ++j) {
        xv[j] = j * 8 + g < m;
        xr[j] = x + (size_t)(xv[j] ? j * 8 + g : 0) * kx + 2 * tq;
    }
    uint4 a[U], na[U];
    unsigned b[U][MT][2], nb[U][MT][2];
    auto load = [&](uint4 (&aa)[U], unsigned (&bb)[U][MT][2], int tb) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int t = min(tb + u, t1 - 1);
            aa[u] = ldw(w + (size_t)t * 32 + lane);
            const int kc = ((t % kt16) * 16) % kx;
#pragma unroll
            for (int j = 0; j < MT; ++j) {
                bb[u][j][0] = xv[j] ? ldx(xr[j] + kc) : 0u;
                bb[u][j][1] = xv[j] ? ldx(xr[j] + kc + 8) : 0u;
            }
        }
    };
    float acc[MT][4];
#pragma unroll
    for (int j = 0; j < MT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    int stripe = t0 / kt16, sbeg = t0;
    load(a, b, t0);
    for (int tb = t0; tb < t1; tb += U) {
        if (tb + U < t1) load(na, nb, tb + U);
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int t = tb + u;
            if (t < t1) {
                if (t - stripe * kt16 >= kt16) {  // crossed into the next stripe
                    flush<MT, U, F32OUT>(acc, stripe, sbeg == stripe * kt16, t - sbeg, kt16, lane, m, n, out, ws, cnt);
#pragma unroll
                    for (int j = 0; j < MT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
                    ++stripe;
                    sbeg = t;
                }
#pragma unroll
                for (int j = 0; j < MT; ++j) mma(acc[j], a[u], b[u][j][0], b[u][j][1]);
            }
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            a[u] = na[u];
#pragma unroll
            for (int j = 0; j < MT; ++j) { b[u][j][0] = nb[u][j][0]; b[u][j][1] = nb[u][j][1]; }
        }
    }
    flush<MT, U, F32OUT>(acc, stripe, sbeg == stripe * kt16 && t1 == (stripe + 1) * kt16, t1 - sbeg, kt16, lane, m, n,
                         out, ws, cnt);
}

template <int MT, int U, int WPB, bool F32OUT>
void launch(const torch::Tensor& w, const torch::Tensor& x, torch::Tensor& out, torch::Tensor& ws, torch::Tensor& cnt,
            int n, int k, int kx, int blocks) {
    const int m = (int)x.size(0);
    TORCH_CHECK(m <= MT * 8 && n % 16 == 0 && k % 16 == 0 && kx % 16 == 0, "pj: bad shape");
    gemv<MT, U, WPB, F32OUT><<<blocks, WPB * 32, 0, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        out.data_ptr(), ws.data_ptr<float>(), cnt.data_ptr<int>(), m, n, k, kx);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace pj

#define PJ_CASE(MT, U, WPB)                                                                  \
    if (mt == MT && u == U && wpb == WPB) {                                                  \
        if (f32) pj::launch<MT, U, WPB, true>(w, x, out, ws, cnt, n, k, kx, blocks);         \
        else pj::launch<MT, U, WPB, false>(w, x, out, ws, cnt, n, k, kx, blocks);            \
        return;                                                                              \
    }

void pj_gemv(torch::Tensor w, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t n,
             int64_t k, int64_t kx, int64_t mt, int64_t u, int64_t wpb, int64_t blocks) {
    const bool f32 = out.scalar_type() == torch::kFloat32;
    PJ_CONFIGS
    TORCH_CHECK(false, "pj: no such config");
}
'''

# Router FP32 kernel configs (mp, kb, rw, nwarp), chosen on the H100 (dev/rt.py).
ROUTER_F32 = [(1, (1, 2048, 1, 8)), (8, (8, 512, 2, 8)), (16, (16, 512, 2, 8))]
# Split-K hi/lo tensor-core GEMV config (mt, u, wpb, blocks) for the router at 17..32 tokens.
ROUTER_HILO = (4, 4, 8, 132)
_SRC = ("#include <torch/extension.h>\n" + _ROUTER_SRC.replace(
    "PR_CONFIGS", " ".join("PR_CASE(%d,%d,%d,%d)" % c for _, c in ROUTER_F32)) + "\n" +
    _GEMV_SRC.replace("PJ_CONFIGS", "PJ_CASE(%d,%d,%d)" % ROUTER_HILO[:3]))
try:
    from torch.utils.cpp_extension import load_inline
    _mod = load_inline(
        name="muse_proj_v1",
        cpp_sources="void pj_router(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, "
                    "int64_t, int64_t, int64_t); void pj_gemv(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                    "torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);",
        cuda_sources=_SRC, functions=["pj_router", "pj_gemv"],
        extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"], verbose=False)
except Exception as _e:  # noqa: BLE001
    print("muse_proj native build failed, using torch:", _e)
    _mod = None

ROUTER_FP32_MAX_TOKENS = 8


def _pack(w):
    """[n, k] -> [n/16][k/16][lane 32][8]: a 16x16 tile's mma.sync A fragments as 512 contiguous bytes."""
    n, k = w.shape
    return w.view(n // 16, 2, 8, k // 16, 2, 4, 2).permute(0, 3, 2, 5, 4, 1, 6).contiguous()


def prepare(weights):
    st = {name: w.contiguous() for name, w in weights.items()}
    r = weights["router"]
    hi = r.to(torch.bfloat16)
    lo = (r - hi.float()).to(torch.bfloat16)
    st["router_hi_t"] = hi.t().contiguous()
    st["router_lo_t"] = lo.t().contiguous()
    st["router_hilo_p"] = _pack(torch.cat([hi, lo], 1).contiguous())  # K = 2 x 8192: x @ hi^T + x @ lo^T
    # Split-K workspace and per-row-block counters: always left zeroed by the kernels.
    st["ws"] = torch.zeros(32 * 8192, dtype=torch.float32, device=r.device)
    st["cnt"] = torch.zeros(1024, dtype=torch.int32, device=r.device)
    return st


def _router(x, st, m):
    if _mod is not None and m <= 32:
        out = torch.empty((m, 256), dtype=torch.float32, device=x.device)
        for mmax, cfg in ROUTER_F32:
            if m <= mmax:
                _mod.pj_router(st["router"], x, out, st["ws"], st["cnt"], *cfg)
                return out
        _mod.pj_gemv(st["router_hilo_p"], x, out, st["ws"], st["cnt"], 256, 2 * 8192, 8192, *ROUTER_HILO)
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
