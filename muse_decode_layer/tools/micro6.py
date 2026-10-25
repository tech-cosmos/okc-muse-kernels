import sys, torch
sys.path.insert(0, ".")
import bench
sub = bench.load_sub("sub_cur.py"); w, st = bench.weights_state(sub); e0 = bench.empty_cost()
for T in (1, 8, 32, 256):
    y = torch.randn(T, 8, 4096, device="cuda").to(torch.bfloat16); wt = torch.rand(T, 8, device="cuda")
    l = torch.randint(-1, 32, (T, 8), device="cuda", dtype=torch.int32); o = torch.empty(T, 4096, device="cuda", dtype=torch.bfloat16)
    f = lambda: sub._post_expert_wsum[(T,)](y, wt, l, st["post_expert_w"], o, D=4096, TOPK=8, EPS=1e-8, num_warps=16)
    print(T, round(bench.time_graph(f) - e0, 2))
