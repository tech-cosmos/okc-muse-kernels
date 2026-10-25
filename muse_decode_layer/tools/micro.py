"""Glue kernels alone: CUDA-graph replay, cold L2, median (us, minus the empty replay)."""
import sys, torch
sys.path.insert(0, ".")
import bench
sub = bench.load_sub(sys.argv[1] if len(sys.argv) > 1 else "sub_v2.py")
w, st = bench.weights_state(sub)
dev = "cuda"
e0 = bench.empty_cost()
for T in (1, 8, 16, 32, 64, 128, 256):
    a = torch.randn(T, 8192, device=dev).to(torch.bfloat16)
    s = torch.randn(T, 8192, device=dev)
    s1, h, h32 = torch.empty_like(s), torch.empty_like(a), torch.empty_like(s)
    al, be = st["res1"]
    fb = lambda: sub._boundary[(T,)](a, s, al, be, al, st["pre_ffn"], s1, h, h32, NB=8192, N=8192, BGAIN=False,
                                     H32=T <= 8, EPS_POST=1e-8, EPS=1e-5, num_warps=16)
    ls = torch.zeros(1, dtype=torch.int32, device=dev)
    fr = lambda: sub._route(st, h, h32, ls, T)
    p = torch.randn(T, 4096, device=dev).to(torch.bfloat16); po = torch.empty_like(p)
    fn = lambda: sub._rmsnorm_gain[(T,)](p, st["pre_expert"], po, N=4096, EPS=1e-5, num_warps=8)
    fl = lambda: torch.nn.functional.linear(h, st["router"].to(torch.float32)) if False else None
    print(f"T={T:4d} boundary {bench.time_graph(fb)-e0:6.2f}  route {bench.time_graph(fr)-e0:6.2f}  "
          f"prenorm {bench.time_graph(fn)-e0:6.2f}", flush=True)
