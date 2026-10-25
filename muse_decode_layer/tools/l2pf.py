"""Does an L2 prefetch of a weight make the following GEMM faster? (H100, graph replay, cold L2 flush between)."""
import fcntl, torch, triton, triton.language as tl, torch.nn.functional as F

@triton.jit
def _pf(w_ptr, n, BLOCK: tl.constexpr, ITERS: tl.constexpr):
    pid = tl.program_id(0)
    acc = tl.zeros([BLOCK], dtype=tl.int32)
    for i in range(ITERS):
        offs = (pid * ITERS + i) * BLOCK + tl.arange(0, BLOCK)
        acc += tl.load(w_ptr + offs, mask=offs < n, other=0, eviction_policy="evict_last")
    s = tl.sum(acc)
    if s == 0x7FFFFFF1:  # never: keeps the loads alive
        tl.store(w_ptr, s)

def prefetch(w, grid=528, BLOCK=1024):
    wi = w.view(torch.int32).view(-1)
    n = wi.numel()
    iters = triton.cdiv(n, grid * BLOCK)
    _pf[(grid,)](wi, n, BLOCK=BLOCK, ITERS=iters, num_warps=4)

dev = "cuda"
flush = torch.empty(512 << 20, dtype=torch.uint8, device=dev)
def timeit(fn, n=50):
    g = torch.cuda.CUDAGraph()
    fn(); torch.cuda.synchronize()
    with torch.cuda.graph(g):
        fn()
    ts = []
    with open("/tmp/okc-gpu.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for _ in range(n):
            flush.zero_()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record(); g.replay(); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e) * 1000)
    ts.sort(); return ts[n // 2]

for name, (N, K) in dict(o=(8192, 1024), qkv=(2304, 8192), unpool=(1024, 4096), pooler=(4096, 8192)).items():
    w = torch.randn(N, K, device=dev).bfloat16()
    x = torch.randn(1, K, device=dev).bfloat16()
    cold = timeit(lambda: F.linear(x, w))
    pf = timeit(lambda: prefetch(w))
    both = timeit(lambda: (prefetch(w), F.linear(x, w)))
    warm = timeit(lambda: (F.linear(x, w), F.linear(x, w))) - cold
    print(f"{name}: {N*K*2/1e6:.1f} MB  gemm cold {cold:.1f}  prefetch {pf:.1f}  pf+gemm {both:.1f} -> gemm after pf {both-pf:.1f}  warm {warm:.1f}")
