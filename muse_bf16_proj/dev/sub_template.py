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


_ROUTER_SRC = r'''@@ROUTER@@'''
_GEMV_SRC = r'''@@GEMV@@'''

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
