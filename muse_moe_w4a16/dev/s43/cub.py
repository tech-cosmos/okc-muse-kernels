import sys, torch
sys.path.insert(0, "/home/daytona/r-layer/muse_decode_layer")
import bench
e0 = bench.empty_cost()
for nm, N, K in [("unp", 8192, 4096), ("pool", 4096, 8192), ("qkv", 2304, 8192)]:
    w = (torch.randn(N, K, device="cuda") * 0.02).bfloat16()
    for T in (128, 256):
        x = torch.randn(T, K, device="cuda").bfloat16()
        t = bench.time_graph(lambda: torch.mm(x, w.t())) - e0
        print(f"{nm} T{T}: cuBLAS {t:.2f} us  {N*K*2/1e6/t:.2f} TB/s  {2*T*N*K/t/1e6:.0f} TF/s", flush=True)
