"""L2 bulk-prefetch issue cost: kernel time of P programs x 512 threads each prefetching ACT chunks of CH bytes,
then the time of a 64 MB cuBLAS GEMV (pooler, m=1) right after (graph, cold L2)."""
import sys, fcntl, statistics, torch, triton, triton.language as tl, torch.nn.functional as F


@triton.jit
def pf(ptr, ch, CHE: tl.constexpr, NT: tl.constexpr):
    i = tl.program_id(0) * NT + tl.arange(0, NT)
    tl.inline_asm_elementwise("cp.async.bulk.prefetch.L2.global [$1], $2; mov.u32 $0, 0;", "=r,l,r",
                              [ptr + i.to(tl.int64) * CHE, tl.full((NT,), 0, tl.int32) + ch], dtype=tl.int32,
                              is_pure=False, pack=1)


dev = "cuda"
w = torch.randn(4096, 8192, device=dev).bfloat16(); x = torch.randn(1, 8192, device=dev).bfloat16()
flush = torch.empty(512 << 20, dtype=torch.uint8, device=dev); sink = torch.zeros((), dtype=torch.int64, device=dev)
def t(fns, n=40):
    g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for f in fns: f()
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            for f in fns: f()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); ts = []
    with open("/tmp/okc-gpu.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for _ in range(n):
            torch.sum(flush.view(torch.int64), 0, out=sink); e0.record(); g.replay(); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1) * 1e3)
    return statistics.median(ts)
gemv = lambda: F.linear(x, w)
print("gemv alone", round(t([gemv]), 2))
for mb, ch, nw in [(16, 16384, 4), (16, 4096, 4), (16, 1024, 16), (16, 2048, 8), (24, 4096, 4), (32, 4096, 4), (16, 8192, 1), (16, 2048, 1)]:
    nt = nw * 32; P = (mb << 20) // (ch * nt)
    k = lambda: pf[(P,)](w, ch, CHE=ch // 2, NT=nt, num_warps=nw)
    print(f"{mb}MB ch{ch} warps{nw} P{P}: pf alone {t([k]):6.2f}  pf+gemv {t([k, gemv]):6.2f}")


@triton.jit
def spin_pf(ptr, ch, NS, T, CHE: tl.constexpr, NT: tl.constexpr, PF: tl.constexpr):
    """Program 0..T-1 spin NS ns (a latency-bound glue kernel); the rest prefetch."""
    if tl.program_id(0) >= T:
        if PF:
            i = (tl.program_id(0) - T) * NT + tl.arange(0, NT)
            tl.inline_asm_elementwise("cp.async.bulk.prefetch.L2.global [$1], $2; mov.u32 $0, 0;", "=r,l,r",
                                      [ptr + i.to(tl.int64) * CHE, tl.full((NT,), 0, tl.int32) + ch], dtype=tl.int32,
                                      is_pure=False, pack=1)
        return
    t0 = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1)
    t1 = t0
    while t1 - t0 < NS:
        t1 = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1)


print("--- spin 4 us kernel (1 CTA, 16 warps) [+ prefetch CTAs] then the 64 MB GEMV")
sp = lambda: spin_pf[(1,)](w, 0, 4000, 1, CHE=1, NT=512, PF=False, num_warps=16)
print("spin+gemv", round(t([sp, gemv]), 2), " spin alone", round(t([sp]), 2))
for mb, ch in [(8, 1024), (16, 1024), (16, 512), (24, 1024), (32, 1024), (48, 1024)]:
    P = (mb << 20) // (ch * 512)
    k = lambda: spin_pf[(1 + P,)](w, ch, 4000, 1, CHE=ch // 2, NT=512, PF=True, num_warps=16)
    print(f"{mb}MB ch{ch} P{P}: spin+pf {t([k]):6.2f}  spin+pf+gemv {t([k, gemv]):6.2f}")
