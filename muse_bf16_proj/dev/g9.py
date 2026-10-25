# gemv9 sweep: python dev/g9.py "M,R,U;..." [blocks,threads ...]   (H100, graph replay, cold L2)
import os, sys, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.cpp_extension import load_inline
import bench, hashlib
cfgs = [tuple(map(int, c.split(","))) for c in sys.argv[1].split(";")]
grids = [tuple(map(int, g.split(","))) for g in (sys.argv[2:] or ["132,512"])]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemv9.cu")).read()
src = src.replace("G9_CONFIGS", " ".join("G9_CASE(%d,%d,%d)" % c for c in cfgs))
name = "pj_g9_" + hashlib.md5(src.encode()).hexdigest()[:10]
mod = load_inline(name=name, cpp_sources="void g9_gemv(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t);",
                  cuda_sources=src, functions=["g9_gemv"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"])
torch.manual_seed(0)
SH = [("qkv", 2304, 8192), ("o", 8192, 1024), ("pool", 4096, 8192), ("unp1k", 1024, 4096), ("unp", 8192, 4096)]
only = os.environ.get("SH")
ws = torch.zeros(32 * 8192, device="cuda"); cnt = torch.zeros(8192, dtype=torch.int32, device="cuda")
e = bench.empty_cost()
for nm, n, k in SH:
    if only and nm not in only.split(","): continue
    W = (torch.randn(n, k, device="cuda") / k ** 0.5).bfloat16()
    for M in sorted({c[0] for c in cfgs}):
        x = torch.randn(M, k, device="cuda").bfloat16()
        ref = x.double() @ W.double().T
        tc = bench.time_graph(lambda: F.linear(x, W)) - e
        line = f"{nm:5s} m{M}: cublas {tc:6.2f}"
        for (m_, r, u) in cfgs:
            if m_ != M or n % r: continue
            for b, t in grids:
                out = torch.empty(M, n, dtype=torch.bfloat16, device="cuda")
                mod.g9_gemv(W, x, out, ws, cnt, r, u, b, t); torch.cuda.synchronize()
                err = ((out.double() - ref).norm() / ref.norm()).item()
                assert err < 5e-3 and ws.abs().max().item() == 0 and cnt.abs().max().item() == 0, (err, nm, m_, r, u)
                dt = bench.time_graph(lambda: mod.g9_gemv(W, x, out, ws, cnt, r, u, b, t)) - e
                line += f" | R{r}U{u} {b}x{t} {dt:6.2f}"
        print(line, flush=True)
    del W
