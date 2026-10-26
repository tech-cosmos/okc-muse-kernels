"""Sweep dev/router3.cu (router m >= 9: one-kernel hi/lo split-K tensor-core GEMM) vs the submission.
  python dev/r3.py "128,64,4,2,256,4;..." [m ...]   configs (MT, NT, WM, WN, KC, NS)"""
import os, sys, torch, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.cpp_extension import load_inline
import bench, reference

cfgs = [tuple(map(int, c.split(","))) for c in sys.argv[1].split(";")]
ms = [int(a) for a in sys.argv[2:]] or [16, 32, 128, 256]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "router3.cu")).read()
src = src.replace("PR3_CONFIGS", " ".join("PR3_CASE(%d,%d,%d,%d,%d,%d)" % c for c in cfgs))
mod = load_inline(name="pj_r3_" + hashlib.md5((src + os.environ.get("DEFS", "")).encode()).hexdigest()[:10],
                  cpp_sources="void pj_router3(torch::Tensor, torch::Tensor, torch::Tensor, "
                              "int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);",
                  cuda_sources=src, functions=["pj_router3"], extra_cuda_cflags=[*["-D" + d for d in os.environ.get("DEFS", "").split(",") if d], "-O3", "-gencode=arch=compute_90a,code=sm_90a"])
w, st = bench.state()
W = st["router"]
hi = W.to(torch.bfloat16)
lo = (W - hi.float()).to(torch.bfloat16)
whl = torch.stack([hi, lo]).contiguous()
ws = torch.zeros(256 * 256, dtype=torch.float32, device="cuda")
cnt = torch.zeros(1024, dtype=torch.int32, device="cuda")
e = bench.empty_cost()
for m in ms:
    data = reference.generate_input(m, "router", seed=5 + m)
    x = data["x"]
    ref = (x.double() @ W.double().t())
    out0 = bench.submission.kernel_fn(data, st)
    t0 = bench.time_graph(lambda: bench.submission.kernel_fn(data, st)) - e
    print(f"m={m} current {t0:.2f} us  err {((out0.double() - ref).norm() / ref.norm()).item():.2e}", flush=True)
    for c in cfgs:
        out = torch.empty(m, 256, dtype=torch.float32, device="cuda")
        for _ in range(3):
            out.fill_(7)
            mod.pj_router3(whl, x, out, *c)
        err = ((out.double() - ref).norm() / ref.norm()).item()
        rowerr = ((out.double() - ref).norm(dim=1) / ref.norm(dim=1).clamp_min(1e-30)).max().item()
        t = bench.time_graph(lambda: mod.pj_router3(whl, x, out, *c)) - e
        z = ws.abs().max().item(), cnt.abs().max().item()
        print(f"m={m} r3 {c}: {t:.2f} us  err {err:.2e} rowmax {rowerr:.2e}", flush=True)
