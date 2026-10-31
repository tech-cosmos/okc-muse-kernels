# n30 oracle 2: same as qkv_splitk.py but TMA tensor descriptors + optional warp_specialize (Triton 3.8), FP32 slabs.
import os, fcntl, statistics, torch, triton, triton.language as tl
from torch.profiler import profile, ProfilerActivity
N, K = int(os.environ.get("N", 2304)), int(os.environ.get("K", 8192))
dev = "cuda"
triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device=dev))

@triton.jit
def sk_tma(x_ptr, w_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, S: tl.constexpr,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, NS: tl.constexpr, WS: tl.constexpr):
    pid_n = tl.program_id(0); pid_m = tl.program_id(1); s = tl.program_id(2)
    xd = tl.make_tensor_descriptor(x_ptr, shape=[M, K], strides=[K, 1], block_shape=[BM, BK])
    wd = tl.make_tensor_descriptor(w_ptr, shape=[N, K], strides=[K, 1], block_shape=[BN, BK])
    od = tl.make_tensor_descriptor(o_ptr, shape=[S * M, N], strides=[N, 1], block_shape=[BM, BN])
    nch = K // BK
    c0 = (s * nch) // S; c1 = ((s + 1) * nch) // S
    acc = tl.zeros((BM, BN), tl.float32)
    for c in tl.range(c0, c1, num_stages=NS, warp_specialize=WS):
        a = xd.load([pid_m * BM, c * BK])
        b = wd.load([pid_n * BN, c * BK])
        acc = tl.dot(a, b.T, acc)
    od.store([s * M + pid_m * BM, pid_n * BN], acc)

flush = torch.empty(200 << 20, dtype=torch.uint8, device=dev)
w = (torch.randn(N, K, device=dev) * 0.02).bfloat16()
for M in (128, 256):
    x = torch.randn(M, K, device=dev).bfloat16()
    ref = (x.float() @ w.float().T)
    ws = torch.empty(16 * M * N, device=dev)
    cands = [("cublas", lambda: torch.nn.functional.linear(x, w))]
    for (S, BM, BN, BK, nw, ns, wsp) in [(7, 128, 128, 64, 4, 4, False), (7, 128, 128, 64, 4, 4, True),
                                        (7, 128, 128, 128, 4, 3, True), (7, 128, 128, 128, 8, 3, False),
                                        (4, 128, 128, 128, 4, 3, True), (3, 128, 128, 128, 4, 3, True),
                                        (14, 128, 64, 128, 4, 4, True), (7, 128, 64, 128, 4, 4, True),
                                        (6, 128, 256, 64, 8, 3, True), (3, 256, 128, 64, 8, 3, True)]:
        if M % BM: continue
        def f(S=S, BM=BM, BN=BN, BK=BK, nw=nw, ns=ns, wsp=wsp):
            sk_tma[(N // BN, M // BM, S)](x, w, ws, M, N, K, S, BM, BN, BK, ns, wsp, num_warps=nw)
            return ws[:S * M * N].view(S, M, N).sum(0)
        try:
            out = f(); torch.cuda.synchronize()
        except Exception as e:
            print("skip", S, BM, BN, BK, wsp, e.__class__.__name__, str(e)[:200]); continue
        err = ((out - ref).norm() / ref.norm()).item()
        cands.append((f"S{S} {BM}x{BN}x{BK} w{nw} s{ns} ws{int(wsp)} ctas={N//BN*(M//BM)*S} err={err:.1e}", f))
    with open("/tmp/okc-gpu.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for name, f in cands:
            with profile(activities=[ProfilerActivity.CUDA]) as p:
                for _ in range(40):
                    flush.zero_(); f()
                torch.cuda.synchronize()
            gemm = [e.device_time for e in p.events() if "sk_tma" in e.name or "gemm" in e.name.lower() or "nvjet" in e.name]
            red = [e.device_time for e in p.events() if "reduce" in e.name.lower()]
            g = statistics.median(gemm) if gemm else 0; r = statistics.median(red) if red else 0
            print(f"M={M} {name}: gemm {g:.1f} + sum {r:.1f} = {g+r:.1f} us", flush=True)
        fcntl.flock(lk, fcntl.LOCK_UN)
