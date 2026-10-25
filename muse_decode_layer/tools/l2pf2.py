"""Bulk L2 prefetch (cp.async.bulk.prefetch.L2) from a few CTAs: duration and its effect on the following GEMM."""
import fcntl, sys, torch, triton, triton.language as tl, torch.nn.functional as F

@triton.jit
def _bpf(w_ptr, nbytes, CHUNK: tl.constexpr, PER: tl.constexpr):
    # each lane prefetches PER chunks of CHUNK bytes
    pid = tl.program_id(0)
    lane = tl.arange(0, 32)
    for i in range(PER):
        off = ((pid * PER + i) * 32 + lane).to(tl.int64) * CHUNK
        ok = off < nbytes
        p = w_ptr + tl.where(ok, off, 0)
        tl.inline_asm_elementwise(
            "{ .reg .pred q; setp.ne.b32 q, $2, 0; @q cp.async.bulk.prefetch.L2.global [$1], $3; mov.b32 $0, 0; }",
            "=r,l,r,r", [p, ok.to(tl.int32), tl.full([32], CHUNK, tl.int32)], dtype=tl.int32, is_pure=False, pack=1)

def prefetch(w, grid, chunk):
    nb = w.numel() * w.element_size()
    per = triton.cdiv(nb, grid * 32 * chunk)
    _bpf[(grid,)](w.view(torch.uint8).view(-1), nb, CHUNK=chunk, PER=per, num_warps=1)

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

empty = timeit(lambda: None)
for name, (N, K) in dict(o=(8192, 1024), unpool=(1024, 4096)).items():
    w = torch.randn(N, K, device=dev).bfloat16()
    x = torch.randn(1, K, device=dev).bfloat16()
    cold = timeit(lambda: F.linear(x, w)) - empty
    print(f"{name}: gemm cold {cold:.1f}")
    for grid in (1, 4, 16, 64):
        for chunk in (4096, 16384, 65536):
            pf = timeit(lambda: prefetch(w, grid, chunk)) - empty
            both = timeit(lambda: (prefetch(w, grid, chunk), F.linear(x, w))) - empty
            sl = timeit(lambda: (prefetch(w, grid, chunk), torch.cuda._sleep(20000), F.linear(x, w))) - empty
            print(f"  grid {grid} chunk {chunk}: pf {pf:.1f}  pf+gemm {both:.1f}  pf+sleep+gemm {sl:.1f}")
    print("  sleep+gemm", timeit(lambda: (torch.cuda._sleep(20000), F.linear(x, w))) - empty)
