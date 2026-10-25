"""python dev/rt.py 'm,...' '[[mp,kb,rw,nwarp], ...]': time + check the FP32 router kernel."""
import hashlib, json, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.cpp_extension import load_inline
import bench, reference
plans = json.loads(sys.argv[2])
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "router.cu")).read().replace(
    "PR_CONFIGS", " ".join("PR_CASE(%d,%d,%d,%d)" % tuple(p) for p in plans))
mod = load_inline(name="pr_" + hashlib.md5(src.encode()).hexdigest()[:10], cpp_sources="void pj_router(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t);",
                  cuda_sources=src, functions=["pj_router"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])
W = reference.generate_weights(reference.WEIGHTS_SEED)
wr = W["router"].contiguous()
ws = torch.zeros(32 * 8192, device="cuda"); cnt = torch.zeros(1024, dtype=torch.int32, device="cuda")
for m in map(int, sys.argv[1].split(",")):
    d = reference.generate_input(m, "router", seed=99)
    for p in plans:
        if p[0] < m: continue
        def f():
            out = torch.empty((m, 256), device="cuda")
            mod.pj_router(wr, d["x"], out, ws, cnt, *p)
            return out
        ok, msg = reference.check_implementation(d, f(), W)
        t = bench.time_graph(f) - bench.empty_cost()
        print(m, p, f"{t:6.2f} us", ok, msg[:70], flush=True)
