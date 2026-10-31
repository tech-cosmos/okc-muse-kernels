# n31 oracle: qkv (N=2304, K=8192) at M=128/256 with a TileLang (wgmma+TMA pipelined) split-K GEMM -> FP32 slabs [S,M,N]
# (no reduce) vs cuBLAS F.linear. Run with /usr/local/bin/python3.14 (has tilelang; torch 2.11, kernel-speed oracle only).
# Cold L2 (200 MB flush), CUDA-event-free CUPTI timing via torch.profiler, median of 40; GPU lock around timed loops.
import os, sys, fcntl, statistics, torch
import tilelang, tilelang.language as T
from torch.profiler import profile, ProfilerActivity
N, K = int(os.environ.get("N", 2304)), int(os.environ.get("K", 8192))

def make(M, S, BM, BN, BK, stages, threads):
    KS = K // S
    @T.prim_func
    def sk(A: T.Tensor((M, K), "bfloat16"), B: T.Tensor((N, K), "bfloat16"), C: T.Tensor((S, M, N), "float32")):
        with T.Kernel(N // BN, M // BM, S, threads=threads) as (bx, by, bz):
            As = T.alloc_shared((BM, BK), "bfloat16")
            Bs = T.alloc_shared((BN, BK), "bfloat16")
            Cl = T.alloc_fragment((BM, BN), "float32")
            T.clear(Cl)
            for k in T.Pipelined(KS // BK, num_stages=stages):
                T.copy(A[by * BM, bz * KS + k * BK], As)
                T.copy(B[bx * BN, bz * KS + k * BK], Bs)
                T.gemm(As, Bs, Cl, transpose_B=True)
            T.copy(Cl, C[bz, by * BM, bx * BN])
    return tilelang.compile(sk, out_idx=[2], target="cuda")

flush = torch.empty(200 << 20, dtype=torch.uint8, device="cuda")
w = (torch.randn(N, K, device="cuda") * 0.02).bfloat16()
cfgs = [eval(c) for c in os.environ.get("CFGS", "").split(";") if c] or [
    (8, 128, 128, 64, 4, 256), (4, 128, 128, 64, 4, 256), (8, 64, 128, 64, 4, 128), (4, 128, 128, 128, 3, 256),
    (8, 128, 128, 128, 3, 256), (2, 128, 256, 64, 3, 256), (4, 128, 256, 64, 3, 256), (16, 64, 128, 64, 4, 128),
    (4, 64, 128, 128, 4, 128), (8, 64, 64, 128, 4, 128)]
for M in [int(m) for m in os.environ.get("MS", "128,256").split(",")]:
    x = torch.randn(M, K, device="cuda").bfloat16()
    ref = x.float() @ w.float().T
    cands = [("cublas", lambda: torch.nn.functional.linear(x, w))]
    for (S, BM, BN, BK, st, th) in cfgs:
        if M % BM or N % BN or (K // S) % BK: continue
        try:
            kern = make(M, S, BM, BN, BK, st, th)
            out = kern(x, w); torch.cuda.synchronize()
        except Exception as e:
            print("skip", (S, BM, BN, BK, st, th), str(e).splitlines()[-1][:120], flush=True); continue
        err = ((out.sum(0) - ref).norm() / ref.norm()).item()
        cands.append((f"S{S} {BM}x{BN}x{BK} st{st} th{th} ctas={N//BN*(M//BM)*S} err={err:.1e}", lambda kern=kern: kern(x, w)))
    with open("/tmp/okc-gpu.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for name, f in cands:
            f(); torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as p:
                for _ in range(40):
                    flush.zero_(); f()
                torch.cuda.synchronize()
            ts = [e.device_time for e in p.events() if e.device_time > 0 and "zero" not in e.name.lower() and "fill" not in e.name.lower()
                  and not (e.name.startswith("void at::") and "elementwise" in e.name)]
            names = set(e.name[:40] for e in p.events() if e.device_time > 0)
            # gemm kernels (cuBLAS may add a splitK reduce kernel: sum per iteration)
            tot = sum(ts) / 40
            print(f"M={M} {name}: {tot:.1f} us/iter  kernels={sorted(n for n in names if 'zero' not in n.lower() and 'fill' not in n.lower())[:3]}", flush=True)
        fcntl.flock(lk, fcntl.LOCK_UN)
