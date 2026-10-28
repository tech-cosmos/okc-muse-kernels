import sys, torch
sys.path.insert(0, ".")
import bench
sub = bench.load_sub("submission.py")
w, st = bench.weights_state(sub)
h = (torch.randn(128, 8192, device="cuda") * 0.5).bfloat16(); h32 = h.float()
ls = torch.zeros(1, dtype=torch.int32, device="cuda")
e0 = bench.empty_cost()
t = bench.time_graph(lambda: sub._route(st, h, h32, ls, 128)) - e0
tp = bench.time_graph(lambda: torch.mm(h, st["pool_router"].t(), out_dtype=torch.float32)) - e0
tq = bench.time_graph(lambda: torch.mm(h, w["pooler"].t())) - e0
print(f"T=128 router(pr3)+routing {t:.2f} us | FR torch.mm {tp:.2f} | pooler-only cuBLAS {tq:.2f}")
