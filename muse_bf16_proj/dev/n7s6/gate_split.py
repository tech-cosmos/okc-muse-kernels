# n7s6: cost of splitting the layer's T=128/256 qkv (N) and o (K) cuBLAS GEMMs into 2 KV-head-group halves.
# CUDA-graph replay, cold L2 (256 MB flush outside the timed region via events), median of 50. Holds the GPU lock only around timing.
import torch, fcntl, statistics
torch.manual_seed(0)
dev = "cuda"
flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
def timeit(fn, reps=50):
    g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn(); torch.cuda.synchronize()
        with torch.cuda.graph(g): fn()
    ts = []
    with open("/tmp/okc-gpu.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for _ in range(reps):
            flush.zero_(); a = torch.cuda.Event(True); b = torch.cuda.Event(True)
            a.record(); g.replay(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b) * 1e3)
    return statistics.median(ts) - globals().get('base', 0.0)
tiny = torch.zeros(1, device=dev)
base = timeit(lambda: tiny.add_(1))
for M in (128, 256):
    x = torch.randn(M, 8192, device=dev, dtype=torch.bfloat16)
    W = torch.randn(2304, 8192, device=dev, dtype=torch.bfloat16)
    out = torch.empty(M, 2304, device=dev, dtype=torch.bfloat16)
    Wa = W[:1280].contiguous(); Wg = W[1280:].contiguous()
    full = timeit(lambda: torch.mm(x, W.t(), out=out))
    a = timeit(lambda: out[:, :1280].copy_(torch.mm(x, Wa.t())))
    a_only = timeit(lambda: torch.mm(x, Wa.t()))
    g = timeit(lambda: torch.mm(x, Wg.t()))
    print(f"M{M}: qkv 2304 {full:.1f} | qkv' 1280 {a_only:.1f} (+copy into slice {a:.1f}) | gate 1024 {g:.1f} us")
