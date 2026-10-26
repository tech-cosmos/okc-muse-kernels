"""Sweep dev/router2.cu (m <= 8 FP32 router, no split K) vs the submission's pj_router.
  python dev/r2.py "1,2,1024;1,4,512;8,2,1024" [m ...]   configs (M, R, T)"""
import os, sys, torch, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.cpp_extension import load_inline
import bench, reference

cfgs = [tuple(map(int, c.split(","))) for c in sys.argv[1].split(";")]
ms = [int(a) for a in sys.argv[2:]] or [1, 8]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "router2.cu")).read()
src = src.replace("PR2_CONFIGS", " ".join("PR2_CASE(%d,%d,%d)" % c for c in cfgs))
mod = load_inline(name="pj_r2_" + hashlib.md5(src.encode()).hexdigest()[:10],
                  cpp_sources="void pj_router2(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t);",
                  cuda_sources=src, functions=["pj_router2"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])
w, st = bench.state()
W = st["router"]
e = bench.empty_cost()
for m in ms:
    data = reference.generate_input(m, "router", seed=5 + m)
    x = data["x"]
    ref = (x.double() @ W.double().t())
    out0 = bench.submission.kernel_fn(data, st)
    t0 = bench.time_graph(lambda: bench.submission.kernel_fn(data, st)) - e
    print(f"m={m} current {t0:.2f} us  err {((out0.double() - ref).norm() / ref.norm()).item():.2e}", flush=True)
    for c in cfgs:
        if c[0] < m or (8192 // (4 * c[2])) * 0 != 0:
            continue
        out = torch.empty(m, 256, dtype=torch.float32, device="cuda")
        mod.pj_router2(W, x, out, *c)
        err = ((out.double() - ref).norm() / ref.norm()).item()
        t = bench.time_graph(lambda: mod.pj_router2(W, x, out, *c)) - e
        print(f"m={m} r2 {c}: {t:.2f} us  err {err:.2e}", flush=True)
