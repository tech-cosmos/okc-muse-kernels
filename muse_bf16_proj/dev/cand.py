# okc: problem=muse_bf16_proj
"""dev candidate: pj_gemv for m <= 32 (incl. router hi/lo), cuBLAS above."""
import hashlib, os
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

_SRC = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemv4.cu")).read()
CONFIGS = [tuple(int(v) for v in c.split(",")) for c in os.environ.get(
    "PJ_CONFIGS", "1,8,8;2,8,8;4,8,8;1,4,8;2,4,8;4,4,8;1,16,8").split(";")]
_defs = " ".join(f"PJ_CASE({a},{b},{c})" for a, b, c in CONFIGS)
_mod = load_inline(name="pj_dev_" + hashlib.md5((_defs + os.environ.get("PJ_DEFS", "") + _SRC).encode()).hexdigest()[:10], cpp_sources="void pj_gemv(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);",
                   cuda_sources=os.environ.get("PJ_DEFS", "") + _SRC.replace("PJ_CONFIGS", _defs), functions=["pj_gemv"],
                   extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"], verbose=False)

ROUTER_FP32_MAX_TOKENS = 8


def _pack(w):
    n, k = w.shape
    return w.view(n // 16, 2, 8, k // 16, 2, 4, 2).permute(0, 3, 2, 5, 4, 1, 6).contiguous()


def prepare(weights):
    st = {name: w.contiguous() for name, w in weights.items()}
    r = weights["router"]
    hi = r.to(torch.bfloat16)
    lo = (r - hi.float()).to(torch.bfloat16)
    st["router_hi_t"] = hi.t().contiguous()
    st["router_lo_t"] = lo.t().contiguous()
    st["p"] = {name: _pack(weights[name]) for name in ("qkv_gate", "o", "pooler", "unpooler")}
    st["p"]["router"] = _pack(torch.cat([hi, lo], 1).contiguous())
    st["ws"] = torch.zeros(32 * 8192, dtype=torch.float32, device="cuda")
    st["cnt"] = torch.zeros(1024, dtype=torch.int32, device="cuda")
    return st


# (m, layer) -> (mt, sp, c, nw, ns, splits); default by m
import json
PLAN = {tuple(json.loads(k_)): tuple(v) for k_, v in json.loads(os.environ.get("PJ_PLAN", "{}")).items()}


def plan(m, layer, n, k):
    if (m, layer) in PLAN:
        return PLAN[(m, layer)]
    mt = 1 if m <= 8 else 2 if m <= 16 else 4
    sp = 1
    c = 32
    return (mt, 8, 8, 132 * 2)


def kernel_fn(data, st):
    x, cfg = data["x"], data["config"]
    layer, m, n, k = cfg["layer"], cfg["m"], cfg["n"], cfg["k"]
    if m <= 32 and not (layer == "router" and m <= ROUTER_FP32_MAX_TOKENS):
        f32 = layer == "router"
        out = torch.empty((m, n), dtype=torch.float32 if f32 else torch.bfloat16, device=x.device)
        kk = 2 * k if f32 else k
        _mod.pj_gemv(st["p"][layer], x, out, st["ws"], st["cnt"], n, kk, k, *plan(m, layer, n, kk))
        return out
    if layer != "router":
        return F.linear(x, st[layer][:n])
    if m <= ROUTER_FP32_MAX_TOKENS:
        return F.linear(x.float(), st["router"])
    acc = torch.mm(x, st["router_hi_t"], out_dtype=torch.float32)
    acc += torch.mm(x, st["router_lo_t"], out_dtype=torch.float32)
    return acc
