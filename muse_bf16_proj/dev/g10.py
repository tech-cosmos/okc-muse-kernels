# gemv10 sweep: python dev/g10.py "M,U,NV,MINB,PF;..." blocks [blocks ...]   (H100, graph replay, cold L2)
# env SH=qkv,o,... selects shapes. Compiles every config for every K that fits (U*2048 % K == 0).
import os, sys, torch, torch.nn.functional as F, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.cpp_extension import load_inline
import bench
cfgs = [tuple(map(int, c.split(","))) for c in sys.argv[1].split(";")]
grids = [int(g) for g in (sys.argv[2:] or ["132"])]
SH = [("qkv", 2304, 8192), ("o", 8192, 1024), ("pool", 4096, 8192), ("unp1k", 1024, 4096), ("unp", 8192, 4096)]
only = os.environ.get("SH")
SH = [s for s in SH if not only or s[0] in only.split(",")]
ks = sorted({k for _, _, k in SH})
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemv10.cu")).read()
src = src.replace("G10_CONFIGS", " ".join("G10_CASE(%d,%d,%d,%d,%d,%d)" % (m, k, u, v, b, p) for (m, u, v, b, p) in cfgs for k in ks if (u * 2048) % k == 0))
mod = load_inline(name="pj_g10_" + hashlib.md5(src.encode()).hexdigest()[:10], cpp_sources="void g10_gemv(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t);",
                  cuda_sources=src, functions=["g10_gemv"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"])
torch.manual_seed(0)
e = bench.empty_cost()
for nm, n, k in SH:
    W = (torch.randn(n, k, device="cuda") / k ** 0.5).bfloat16()
    for M in sorted({c[0] for c in cfgs}):
        x = torch.randn(M, k, device="cuda").bfloat16()
        ref = x.double() @ W.double().T
        tc = bench.time_graph(lambda: F.linear(x, W)) - e
        line = f"{nm:5s} m{M}: cublas {tc:6.2f}"
        for (m_, u, nv, mb, pf) in cfgs:
            if m_ != M or (u * 2048) % k: continue
            for b in grids:
                out = torch.empty(M, n, dtype=torch.bfloat16, device="cuda")
                mod.g10_gemv(W, x, out, u, nv, mb, pf, b); torch.cuda.synchronize()
                err = ((out.double() - ref).norm() / ref.norm()).item()
                assert err < 5e-3, (err, nm, m_, u, nv, mb, pf, b)
                dt = bench.time_graph(lambda: mod.g10_gemv(W, x, out, u, nv, mb, pf, b)) - e
                line += f" | U{u}V{nv}B{mb}P{pf} g{b} {dt:6.2f}"
        print(line, flush=True)
    del W
