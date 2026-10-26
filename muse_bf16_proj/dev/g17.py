"""Sweep dev/gemv17.cu vs the submission on small BF16 shapes.
  python dev/g17.py "1,8,512;8,4,512" unp1k:1 o:1 unp1k:8 o:8      configs (M, R, T); shapes layer:m"""
import os, sys, torch, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.cpp_extension import load_inline
import bench, reference

cfgs = [tuple(map(int, c.split(","))) for c in sys.argv[1].split(";")]
shapes = [(a.split(":")[0], int(a.split(":")[1])) for a in sys.argv[2:]]
L = {"qkv": ("qkv_gate", 8192), "o": ("o", 1024), "pool": ("pooler", 8192), "unp1k": ("unpooler", 4096), "unp": ("unpooler", 4096)}
ks = sorted({L[s][1] for s, _ in shapes})
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemv17.cu")).read()
src = src.replace("G17_CONFIGS", " ".join("G17_CASE(%d,%d,%d,%d)" % (c + (k,)) for c in cfgs for k in ks if k % (8 * c[2]) == 0))
mod = load_inline(name="pj_g17_" + hashlib.md5(src.encode()).hexdigest()[:10],
                  cpp_sources="void g17_gemv(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t);",
                  cuda_sources=src, functions=["g17_gemv"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])
w, st = bench.state()
e = bench.empty_cost()
for nm, m in shapes:
    layer, k = L[nm]
    data = reference.generate_input(m, layer, seed=11 + m)
    x, n = data["x"], data["config"]["n"]
    W = w[layer][:n].contiguous()
    ref = (x.double() @ W.double().t())
    t0 = bench.time_graph(lambda: bench.submission.kernel_fn(data, st)) - e
    print(f"{nm} m={m} n={n}: current {t0:.2f} us", flush=True)
    for c in cfgs:
        if c[0] < m or k % (8 * c[2]) or n % c[1]:
            continue
        out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
        mod.g17_gemv(W, x, out, *c)
        err = ((out.double() - ref).norm() / ref.norm()).item()
        t = bench.time_graph(lambda: mod.g17_gemv(W, x, out, *c)) - e
        print(f"  g17 {c}: {t:.2f} us err {err:.2e}", flush=True)
