import sys, torch
sys.path.insert(0, ".")
import bench
sub = bench.load_sub("sub_v3.py")
w, st = bench.weights_state(sub)
e0 = bench.empty_cost()
al, be = st["res1"]
for T in (1, 16):
    a = torch.randn(T, 8192, device="cuda").to(torch.bfloat16); s = torch.randn(T, 8192, device="cuda")
    s1, h, h32 = torch.empty_like(s), torch.empty_like(a), torch.empty_like(s)
    for nw in (4, 8, 16, 32):
        f = lambda: sub._boundary[(T,)](a, s, al, be, al, st["pre_ffn"], s1, h, h32, NB=8192, N=8192, BGAIN=False, H32=False, EPS_POST=1e-8, EPS=1e-5, num_warps=nw)
        print(T, "single nw", nw, round(bench.time_graph(f)-e0, 2), flush=True)
    for nc in (2, 4, 16):
        for nw in (4, 8):
            ws, cnt = sub._ws(f"x{nc}{nw}", T, "cuda")
            f = lambda: sub._boundary_mc[(nc, T)](a, s, al, be, al, st["pre_ffn"], s1, h, h32, ws, cnt, NB=8192, N=8192, NC=nc, BGAIN=False, H32=False, EPS_POST=1e-8, EPS=1e-5, num_warps=nw)
            print(T, "mc", nc, nw, round(bench.time_graph(f)-e0, 2), flush=True)
