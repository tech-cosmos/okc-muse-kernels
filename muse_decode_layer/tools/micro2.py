import sys, torch, triton
sys.path.insert(0, ".")
import bench
sub = bench.load_sub(sys.argv[1] if len(sys.argv) > 1 else "sub_cur.py")
w, st = bench.weights_state(sub)
dev = "cuda"
e0 = bench.empty_cost()
x = torch.zeros(1, device=dev)
print("one tiny kernel", bench.time_graph(lambda: x.add_(1)) - e0)
for T in (1, 8, 16, 64, 256):
    h = torch.randn(T, 8192, device=dev).to(torch.bfloat16); h32 = h.float()
    bt, be, bk, ks, warps = sub._router_plan(T)
    part = torch.empty((ks, T, 256), dtype=torch.float32, device=dev)
    if T <= 8:
        f1 = lambda: sub._router_fma[(256 // be, ks)](h32, st["router"], part, T, K=8192, E=256, BT=bt, BE=be, BK=bk, KS=ks, num_warps=warps)
    else:
        f1 = lambda: sub._router_split[(256 // be, ks, triton.cdiv(T, bt))](h, st["router_hi"], st["router_lo"], part, T, K=8192, E=256, BT=bt, BE=be, BK=bk, KS=ks, HILO=True, num_warps=warps)
    ids = torch.empty((T, 8), dtype=torch.int32, device=dev); lids = ids.clone(); wts = torch.empty((T, 8), device=dev)
    ls = torch.zeros(1, dtype=torch.int32, device=dev)
    f2 = lambda: sub._routing[(T,)](part, st["router_bias"], ls, ids, lids, wts, T, E=256, KS=ks, TOPK=8, LOCAL=32, MULT=0.1767766952966369, REPS=1e-15, num_warps=NW)
    fl = lambda: torch.nn.functional.linear(h32 if T <= 8 else h, st["router"] if T <= 8 else st["router_hi"])
    for NW in (1, 2, 4):
        print(NW, round(bench.time_graph(f2)-e0, 2))
    print(f"T={T} split {bench.time_graph(f1)-e0:6.2f} routing {bench.time_graph(f2)-e0:6.2f} cublas {bench.time_graph(fl)-e0:6.2f}", flush=True)
