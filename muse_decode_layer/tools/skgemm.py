"""Triton split-K skinny GEMM (T = 16/32 tokens) with a last-CTA reduce, vs F.linear, graph + cold L2 (H100).
  python tools/skgemm.py            # sweep configs on the layer's 4 projections at m 16 and 32"""
import sys, fcntl, statistics, itertools, torch, torch.nn.functional as F
import triton, triton.language as tl


@triton.jit
def _sk_gemm(x_ptr, w_ptr, out_ptr, ws_ptr, cnt_ptr, N, K: tl.constexpr, M: tl.constexpr, BN: tl.constexpr,
             BK: tl.constexpr, SPLIT: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    KC: tl.constexpr = K // SPLIT
    offs_m = tl.arange(0, M)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    acc = tl.zeros((M, BN), dtype=tl.float32)
    k0 = pid_k * KC
    for kk in tl.range(0, KC, BK):
        x = tl.load(x_ptr + offs_m[:, None] * K + (k0 + kk + offs_k)[None, :])
        w = tl.load(w_ptr + offs_n[:, None] * K + (k0 + kk + offs_k)[None, :])
        acc = tl.dot(x, w.T, acc)
    if SPLIT == 1:
        tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.bfloat16))
    else:
        tl.store(ws_ptr + (pid_k * M + offs_m[:, None]) * N + offs_n[None, :], acc)
        tl.debug_barrier()
        old = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel")
        if old == SPLIT - 1:
            tot = tl.zeros((M, BN), dtype=tl.float32)
            for s in range(SPLIT):
                tot += tl.load(ws_ptr + (s * M + offs_m[:, None]) * N + offs_n[None, :], cache_modifier=".cg")
            tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], tot.to(tl.bfloat16))
            tl.atomic_xchg(cnt_ptr + pid_n, 0)


def sk_linear(x, w, out, ws, cnt, BN, BK, SPLIT, warps, stages):
    M, K = x.shape
    N = w.shape[0]
    _sk_gemm[(N // BN, SPLIT)](x, w, out, ws, cnt, N, K=K, M=M, BN=BN, BK=BK, SPLIT=SPLIT, num_warps=warps,
                               num_stages=stages)
    return out


if __name__ == "__main__":
    dev = "cuda"
    flush = torch.empty(512 << 20, dtype=torch.uint8, device=dev)
    sink = torch.zeros((), dtype=torch.int64, device=dev)

    def t(fn, n=40):
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            fn(); torch.cuda.synchronize()
            with torch.cuda.graph(g):
                fn()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        ts = []
        with open("/tmp/okc-gpu.lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            for _ in range(n):
                torch.sum(flush.view(torch.int64), 0, out=sink)
                e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
                ts.append(e0.elapsed_time(e1) * 1e3)
        return statistics.median(ts)

    empty = t(lambda: None)
    shapes = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpool": (8192, 4096)}
    only = sys.argv[1:] or list(shapes)
    ws = torch.empty(64 * 32 * 8192, dtype=torch.float32, device=dev)
    cnt = torch.zeros(1024, dtype=torch.int32, device=dev)
    for name in only:
        n, k = shapes[name]
        w = (torch.randn(n, k, device=dev) * 0.02).bfloat16()
        for m in (16, 32):
            x = torch.randn(m, k, device=dev).bfloat16()
            ref = F.linear(x, w)
            tc = t(lambda: F.linear(x, w)) - empty
            res = []
            for BN, BK, SPLIT, warps, stages in itertools.product((32, 64, 128), (64, 128, 256), (1, 2, 4, 8, 16),
                                                                  (4,), (3, 4, 6)):
                if n % BN or k % (SPLIT * BK) or (n // BN) * SPLIT < 100 or (n // BN) * SPLIT > 2200:
                    continue
                out = torch.empty(m, n, dtype=torch.bfloat16, device=dev)
                try:
                    sk_linear(x, w, out, ws, cnt, BN, BK, SPLIT, warps, stages)
                except Exception as e:  # smem overflow etc.
                    continue
                err = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
                if err > 1e-2:
                    print("BAD", name, m, BN, BK, SPLIT, err)
                    continue
                res.append((t(lambda: sk_linear(x, w, out, ws, cnt, BN, BK, SPLIT, warps, stages), 20) - empty,
                            (BN, BK, SPLIT, warps, stages)))
            res.sort()
            print(f"{name:9s} m{m}: cublas {tc:6.2f}  best", [(round(a, 2), c) for a, c in res[:4]], flush=True)
