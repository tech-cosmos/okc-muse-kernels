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
print(f'empty-graph (1 tiny kernel) {base:.1f} us; numbers below are minus this')
for M in (128, 256):
    x = torch.randn(M, 8192, device=dev, dtype=torch.bfloat16)
    Wq = torch.randn(2304, 8192, device=dev, dtype=torch.bfloat16)
    full = timeit(lambda: torch.mm(x, Wq.t()))
    h0 = Wq[:1152].contiguous(); h1 = Wq[1152:].contiguous()
    half = timeit(lambda: torch.mm(x, h0.t()))
    both = timeit(lambda: (torch.mm(x, h0.t()), torch.mm(x, h1.t())))
    a = torch.randn(M, 1024, device=dev, dtype=torch.bfloat16)
    Wo = torch.randn(8192, 1024, device=dev, dtype=torch.bfloat16)
    ofull = timeit(lambda: torch.mm(a, Wo.t()))
    a0 = a[:, :512].contiguous(); w0 = Wo[:, :512].contiguous()
    ohalf = timeit(lambda: torch.mm(a0, w0.t()))
    out = torch.empty(M, 8192, device=dev, dtype=torch.float32)
    print(f"M{M}: qkv full {full:.1f} half {half:.1f} two-halves-serial {both:.1f} | o full {ofull:.1f} half-K {ohalf:.1f} us")
    del x, Wq, h0, h1, a, Wo, a0, w0, out
