import sys, torch, triton
sys.path.insert(0, ".")
import bench
sub = bench.load_sub("sub_v2.py")
w, st = bench.weights_state(sub)
e0 = bench.empty_cost()
T = 1
h32 = torch.randn(T, 8192, device="cuda")
for bt, be, bk, ks, nw in [(1,1,2048,4,4),(1,2,1024,8,4),(1,2,2048,4,4),(1,4,1024,8,4),(1,4,512,32,4),(1,4,1024,16,8),(1,8,512,16,4),(1,8,1024,8,8),(1,2,2048,2,4),(1,1,4096,2,4),(8,2,1024,8,4),(8,1,2048,4,4),(4,2,1024,8,4)]:
    part = torch.empty((ks, T, 256), dtype=torch.float32, device="cuda")
    f1 = lambda: sub._router_fma[(256 // be, ks)](h32, st["router"], part, T, K=8192, E=256, BT=bt, BE=be, BK=bk, KS=ks, num_warps=nw)
    print(bt, be, bk, ks, nw, round(bench.time_graph(f1)-e0, 2), flush=True)
