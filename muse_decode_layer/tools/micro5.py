"""o GEMM (cuBLAS) followed by boundary 1, with and without PDL."""
import sys, torch
sys.path.insert(0, ".")
import bench
sub = bench.load_sub("sub_v3.py")
w, st = bench.weights_state(sub)
e0 = bench.empty_cost()
al, be = st["res1"]
for T in (1, 16):
    o = torch.randn(T, 1024, device="cuda").to(torch.bfloat16); s = torch.randn(T, 8192, device="cuda")
    s1, h, h32 = torch.empty_like(s), torch.empty(T, 8192, device="cuda", dtype=torch.bfloat16), torch.empty_like(s)
    for pdl in (False, True):
        def f():
            a = torch.nn.functional.linear(o, st["dense"]["o"])
            sub._boundary[(T,)](a, s, al, be, al, st["pre_ffn"], s1, h, h32, NB=8192, N=8192, BGAIN=False, H32=False, EPS_POST=1e-8, EPS=1e-5, PDL=pdl, num_warps=16, launch_pdl=pdl)
        print(T, "pdl", pdl, round(bench.time_graph(f)-e0, 2), flush=True)
    print(T, "gemm alone", round(bench.time_graph(lambda: torch.nn.functional.linear(o, st["dense"]["o"]))-e0, 2))
