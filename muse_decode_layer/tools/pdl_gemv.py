"""Prototype: o-proj GEMV (W [8192,1024] bf16) launched with PDL that loads its W tile into registers BEFORE
griddepcontrol.wait, so the weight read overlaps a latency-bound producer (attention at decode 1)."""
import fcntl, sys, torch, triton, triton.language as tl, torch.nn.functional as F
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

@triton.jit
def _producer(src, x_ptr, K: tl.constexpr, STEPS: tl.constexpr, TRIGGER: tl.constexpr):
    # 32 CTAs: a dependent chain of loads (latency-bound, like attention at decode 1), then writes x
    if TRIGGER:
        gdc_launch_dependents()
    pid = tl.program_id(0)
    idx = pid * 64 + tl.arange(0, 64)
    for i in range(STEPS):
        v = tl.load(src + idx)
        idx = (v + idx * 7 + i * 4099) & 0xFFFFF
    offs = tl.arange(0, K // 32) + pid * (K // 32)
    tl.store(x_ptr + offs, (idx.to(tl.float32).sum() * 0.0 + offs.to(tl.float32) * 0.001).to(tl.bfloat16))

@triton.jit
def _gemv(x_ptr, w_ptr, out_ptr, M, N: tl.constexpr, K: tl.constexpr, BN: tl.constexpr, PDL: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BN + tl.arange(0, BN)
    k = tl.arange(0, K)
    w = tl.load(w_ptr + rows[:, None] * K + k[None, :])
    if PDL:
        gdc_wait()
    for m in range(M):
        x = tl.load(x_ptr + m * K + k).to(tl.float32)
        acc = tl.sum(w.to(tl.float32) * x[None, :], axis=1)
        tl.store(out_ptr + m * N + rows, acc.to(tl.bfloat16))

dev = "cuda"
flush = torch.empty(512 << 20, dtype=torch.uint8, device=dev)
def timeit(fn, n=60):
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

N, K = 8192, 1024
w = torch.randn(N, K, device=dev).bfloat16()
src = torch.randint(0, 1 << 20, (1 << 20,), device=dev, dtype=torch.int32)
steps = int(sys.argv[1]) if len(sys.argv) > 1 else 20
if len(sys.argv) > 2 and sys.argv[2] == "tl":
    import json
    from torch.profiler import profile, ProfilerActivity
    M, BN = 1, 32
    x = torch.zeros(M, K, device=dev, dtype=torch.bfloat16); out = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
    def f():
        _producer[(32,)](src, x[0], K=K, STEPS=steps, TRIGGER=True, num_warps=4)
        _gemv[(N // BN,)](x, w, out, M, N=N, K=K, BN=BN, PDL=True, num_warps=8, launch_pdl=True)
    g = torch.cuda.CUDAGraph(); f(); torch.cuda.synchronize()
    with torch.cuda.graph(g):
        f()
    for mode in ("eager", "graph"):
        flush.zero_(); torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            (f() if mode == "eager" else g.replay()); torch.cuda.synchronize()
        p.export_chrome_trace("/tmp/pt.json")
        ev = sorted([e for e in json.load(open("/tmp/pt.json"))["traceEvents"] if e.get("cat") == "kernel"], key=lambda e: e["ts"])
        t0 = ev[0]["ts"]
        print(mode, [(e["name"][:10], round(e["ts"] - t0, 1), round(e["dur"], 1)) for e in ev])
    sys.exit()
for M in (1, 8):
    x = torch.zeros(M, K, device=dev, dtype=torch.bfloat16)
    out = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
    def prod(trig):
        for m in range(M):
            _producer[(32,)](src, x[m], K=K, STEPS=steps, TRIGGER=trig, num_warps=4)
    empty = timeit(lambda: None)
    p = timeit(lambda: prod(False)) - empty
    cub = timeit(lambda: (prod(False), F.linear(x, w))) - empty
    for BN, nw in ((32, 8), (64, 16), (16, 4)):
        g0 = timeit(lambda: _gemv[(N // BN,)](x, w, out, M, N=N, K=K, BN=BN, PDL=False, num_warps=nw)) - empty
        seq = timeit(lambda: (prod(False), _gemv[(N // BN,)](x, w, out, M, N=N, K=K, BN=BN, PDL=False, num_warps=nw))) - empty
        pdl = timeit(lambda: (prod(True), _gemv[(N // BN,)](x, w, out, M, N=N, K=K, BN=BN, PDL=True, num_warps=nw, launch_pdl=True))) - empty
        ref = F.linear(x, w)
        prod(True); _gemv[(N // BN,)](x, w, out, M, N=N, K=K, BN=BN, PDL=True, num_warps=nw, launch_pdl=True)
        err = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
        print(f"M{M} BN{BN} w{nw}: producer {p:.1f}  +cuBLAS {cub:.1f}  gemv alone {g0:.1f}  +gemv {seq:.1f}  +gemv PDL {pdl:.1f}  err {err:.1e}")
