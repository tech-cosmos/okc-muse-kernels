"""n7s5: all cuBLASLt heuristic algos (any algo id) for the FR GEMM, FP32 out, H100 cold-L2 graph us."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv = [sys.argv[0], "0"] + sys.argv[1:]
import frlt  # builds mod, W (loop over m=0 is skipped below)
from frlt import mod, W, N, K, ws, wsb, bench
for m in [int(a) for a in sys.argv[2].split(",")]:
    h = torch.randn(m, K, device="cuda").bfloat16()
    ref = torch.mm(h, W.t(), out_dtype=torch.float32)
    out = torch.empty(m, N, dtype=torch.float32, device="cuda")
    al = mod.lt_algos(m, N, K, wsb, 64)
    res = []
    for i in range(al.shape[0]):
        a = al[i].clone()
        if mod.lt_mm(W, h, out, a, ws) != 0: continue
        torch.cuda.synchronize()
        if not ((out - ref).norm() / ref.norm()).item() < 1e-5: continue
        res.append((bench.time_graph(lambda: mod.lt_mm(W, h, out, a, ws), reps=50) - bench.empty_cost(), i, mod.lt_desc(a).tolist()))
    tc = bench.time_graph(lambda: torch.mm(h, W.t(), out_dtype=torch.float32)) - bench.empty_cost()
    res.sort()
    print(f"T={m} torch.mm {tc:.2f} | heur {len(res)}/{al.shape[0]}:", " ".join(f"#{i}:{t:.2f}{d[:2]}{d[3:]}" for t, i, d in res[:8]), flush=True)
