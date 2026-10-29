# n7s20: H100 bound for the T=256 unpool zero-local-token skip: cuBLAS y[M,8192] @ W[8192,4096]^T... as torch.mm,
# M = 256 (today) vs kept counts 200..212 (v32 layer routings keep 201-212 of 256), cold L2, CUDA-graph replay, median.
import torch, fcntl, statistics
torch.manual_seed(0)
W = torch.randn(8192, 4096, device="cuda", dtype=torch.bfloat16) * 0.02   # unpooler [8192, 4096]
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
def t(M, reps=60):
    x = torch.randn(M, 8192, device="cuda", dtype=torch.bfloat16)
    o = torch.empty(M, 4096, device="cuda", dtype=torch.bfloat16)
    for _ in range(3): torch.mm(x, W, out=o)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): torch.mm(x, W, out=o)
    ts = []
    with open("/tmp/okc-gpu.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for _ in range(reps):
            flush.zero_(); a = torch.cuda.Event(True); b = torch.cuda.Event(True)
            a.record(); g.replay(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b) * 1e3)
        fcntl.flock(lk, fcntl.LOCK_UN)
    return statistics.median(ts)
for rnd in range(2):
    print(" ".join(f"M{M}:{t(M):.1f}" for M in (256, 240, 224, 212, 208, 201, 192)))
