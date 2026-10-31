# n30 oracle: qkv (N=2304, K=8192) at M=128/256: cuBLAS F.linear vs Triton split-K (FP32 slabs [S,M,N]) + slab sum.
# Cold L2 (200 MB flush), CUPTI kernel times via torch.profiler, median over reps; GPU lock around timed loops.
import os, fcntl, statistics, torch, triton, triton.language as tl
from torch.profiler import profile, ProfilerActivity
N, K = int(os.environ.get("N", 2304)), int(os.environ.get("K", 8192))
dev = "cuda"

@triton.jit
def sk_gemm(x_ptr, w_ptr, o_ptr, M, N: tl.constexpr, K: tl.constexpr, S: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_n = tl.program_id(0); pid_m = tl.program_id(1); s = tl.program_id(2)
    nch = K // BK
    c0 = (s * nch) // S; c1 = ((s + 1) * nch) // S
    rm = pid_m * BM + tl.arange(0, BM); rn = pid_n * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for c in range(c0, c1):
        k = c * BK + rk
        a = tl.load(x_ptr + rm[:, None] * K + k[None, :])
        b = tl.load(w_ptr + rn[:, None] * K + k[None, :])
        acc = tl.dot(a, tl.trans(b), acc)
    tl.store(o_ptr + s * M * N + rm[:, None] * N + rn[None, :], acc)

def run_sk(x, w, ws, S, BM, BN, BK, nw, ns):
    M = x.shape[0]
    sk_gemm[(N // BN, M // BM, S)](x, w, ws, M, N, K, S, BM, BN, BK, num_warps=nw, num_stages=ns)
    return ws[:S].sum(0)

flush = torch.empty(200 << 20, dtype=torch.uint8, device=dev)
w = (torch.randn(N, K, device=dev) * 0.02).bfloat16()
res = {}
for M in (128, 256):
    x = torch.randn(M, K, device=dev).bfloat16()
    ref = (x.float() @ w.float().T)
    ws = torch.empty(16, M, N, device=dev)
    cands = [("cublas", lambda: torch.nn.functional.linear(x, w))]
    for (S, BM, BN, BK, nw, ns) in [(7, 128, 128, 64, 8, 4), (4, 128, 128, 64, 8, 4), (8, 128, 128, 64, 8, 3),
                                   (3, 128, 128, 64, 8, 4), (7, 64, 128, 64, 4, 4), (4, 64, 128, 64, 4, 4),
                                   (2, 128, 128, 64, 8, 4), (3, 128, 256, 64, 8, 3), (6, 128, 256, 64, 8, 3)]:
        if M % BM: continue
        f = (lambda S=S, BM=BM, BN=BN, BK=BK, nw=nw, ns=ns: run_sk(x, w, ws, S, BM, BN, BK, nw, ns))
        try:
            out = f(); torch.cuda.synchronize()
        except Exception as e:
            print("skip", S, BM, BN, e.__class__.__name__); continue
        err = ((out - ref).norm() / ref.norm()).item()
        cands.append((f"S{S} {BM}x{BN}x{BK} w{nw} s{ns} ctas={N//BN*(M//BM)*S} err={err:.1e}", f))
    with open("/tmp/okc-gpu.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for name, f in cands:
            with profile(activities=[ProfilerActivity.CUDA]) as p:
                for _ in range(40):
                    flush.zero_(); f()
                torch.cuda.synchronize()
            evs = [e for e in p.events() if "zero" not in e.name and "fill" not in e.name.lower() and "elementwise_kernel" not in e.name or "reduce" in e.name]
            gemm = [e.device_time for e in p.events() if "sk_gemm" in e.name or ("gemm" in e.name.lower() or "nvjet" in e.name or "sm90" in e.name)]
            red = [e.device_time for e in p.events() if "reduce" in e.name.lower()]
            g = statistics.median(gemm) if gemm else 0; r = statistics.median(red) if red else 0
            print(f"M={M} {name}: gemm {g:.1f} + sum {r:.1f} = {g+r:.1f} us", flush=True)
        fcntl.flock(lk, fcntl.LOCK_UN)
