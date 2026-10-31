"""n16: router3 bounds. DEFS variants: base | R3_NOZERO | R3_NORED | both.  python r3b.py m cfg..."""
import os, sys, torch, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from torch.utils.cpp_extension import load_inline
import bench, reference
src0 = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "r3v.cu")).read()
CF0 = {16: (16, 64, 1, 4, 256, 4), 32: (16, 64, 1, 4, 256, 4), 8: (16, 64, 1, 4, 256, 4), 128: (128, 64, 4, 4, 256, 4), 256: (256, 32, 4, 2, 512, 4)}
import ast
CF0 = {int(k): ast.literal_eval(v) for k, v in (a.split(":") for a in os.environ["CFS"].split(";"))} if "CFS" in os.environ else CF0
import ast
CFL = [(int(k), ast.literal_eval(v)) for k, v in (a.split(":") for a in os.environ["CFS"].split(";"))] if "CFS" in os.environ else list(CF0.items())
CF = dict(CFL)
mods = {}
for d in os.environ.get("VARS", ";R3_NOZERO").split(";"):
    src = src0.replace("PR3_CONFIGS", " ".join("PR3_CASE(%d,%d,%d,%d,%d,%d)" % c for c in set(c for _, c in CFL)))
    mods[d] = load_inline(name="pj_r3v_" + hashlib.md5((src + d).encode()).hexdigest()[:10],
        cpp_sources="void pj_router3(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);",
        cuda_sources=src, functions=["pj_router3"], extra_cuda_cflags=[*["-D" + x for x in d.split(",") if x], "-O3", "-gencode=arch=compute_90a,code=sm_90a"])
W = reference.generate_weights(reference.WEIGHTS_SEED)["router"].cuda()
hi = W.to(torch.bfloat16); lo = (W - hi.float()).to(torch.bfloat16)
whl = torch.stack([hi, lo]).contiguous()
e = bench.empty_cost()
for m, cf in CFL:
    x = torch.randn(m, 8192, device="cuda").to(torch.bfloat16)
    out = torch.zeros(m, 256, device="cuda")
    ref = x.double() @ W.double().t()
    r = []
    for d, mod in mods.items():
        out.fill_(7); mod.pj_router3(whl, x, out, *cf); torch.cuda.synchronize()
        err = ((out.double() - ref).norm() / ref.norm()).item()
        t = bench.time_graph(lambda: mod.pj_router3(whl, x, out, *cf), reps=200) - e
        r.append(f"{d or 'base'} {t:.2f} (err {err:.1e})")
    print(f"m{m} {cf}: " + " | ".join(r), flush=True)
