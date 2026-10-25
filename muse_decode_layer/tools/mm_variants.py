"""cuBLAS variants for the layer's GEMMs at small m: F.linear (bf16 out) vs mm out_dtype=float32, vs addmm."""
import fcntl, torch, torch.nn.functional as F
dev = "cuda"
flush = torch.empty(512 << 20, dtype=torch.uint8, device=dev)
def timeit(fn, n=40):
    g = torch.cuda.CUDAGraph(); fn(); torch.cuda.synchronize()
    with torch.cuda.graph(g):
        fn()
    ts = []
    with open("/tmp/okc-gpu.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for _ in range(n):
            flush.zero_(); s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record(); g.replay(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e) * 1000)
    ts.sort(); return ts[n // 2]
empty = timeit(lambda: None)
for name, (N, K) in dict(qkv=(2304, 8192), o=(8192, 1024), pooler=(4096, 8192), unpool_s=(1024, 4096), unpool=(8192, 4096)).items():
    w = torch.randn(N, K, device=dev).bfloat16()
    for m in (1, 8, 16, 32):
        x = torch.randn(m, K, device=dev).bfloat16()
        a = timeit(lambda: F.linear(x, w)) - empty
        try:
            b = timeit(lambda: torch.mm(x, w.T, out_dtype=torch.float32)) - empty
        except Exception as ex:
            b = float("nan"); print(ex)
        print(f"{name} m{m}: bf16 {a:.1f}  fp32-out {b:.1f}")
