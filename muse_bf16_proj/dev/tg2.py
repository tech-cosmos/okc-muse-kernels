"""H100 A/B of flashinfer 0.7.0 tinygemm_bf16 (dev/tinygemm2.cu port) vs the submission at m<=8 (BF16 out).
  python dev/tg2.py qkv:1 o:1 pool:1 unp1k:1 qkv:8 o:8 pool:8 unp1k:8"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.cpp_extension import load_inline
import bench, reference

L = {"qkv": "qkv_gate", "o": "o", "pool": "pooler", "unp1k": "unpooler", "unp": "unpooler"}
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tinygemm2.cu")).read()
mod = load_inline(name="pj_tg2", cpp_sources="void tg2(torch::Tensor, torch::Tensor, torch::Tensor, bool);",
                  cuda_sources=src, functions=["tg2"], extra_ldflags=["-lcuda"],
                  extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])
w, st = bench.state()
e = bench.empty_cost()
for a in sys.argv[1:]:
    nm, m = a.split(":"); m = int(m); layer = L[nm]
    data = reference.generate_input(m, layer, seed=11 + m)
    x, n = data["x"], data["config"]["n"]
    W = w[layer][:n].contiguous()
    ref = x.double() @ W.double().t()
    t0 = bench.time_graph(lambda: bench.submission.kernel_fn(data, st)) - e
    out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    mod.tg2(x, W, out, False); torch.cuda.synchronize()
    err = ((out.double() - ref).norm() / ref.norm()).item()
    t = bench.time_graph(lambda: mod.tg2(x, W, out, False)) - e
    print(f"{nm} m={m} n={n}: current {t0:.2f} us  tinygemm2 {t:.2f} us err {err:.2e}", flush=True)
