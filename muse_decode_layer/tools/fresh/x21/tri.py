# x21: can a Triton wgmma GEMM beat nvjet (torch.matmul) on the layer's T=128/256 dense shapes? H100, cold-ish L2, CUDA graph.
import torch, triton, triton.language as tl, fcntl, itertools, sys

@triton.jit
def mm(x_ptr, w_ptr, o_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SK: tl.constexpr, GM: tl.constexpr):
    # out[M,N] (+)= x[M,K] . w[N,K]^T ; SK > 1: FP32 atomics into a zeroed fp32 out
    pid = tl.program_id(0); ks = tl.program_id(1)
    nm = tl.cdiv(M, BM); nn = tl.cdiv(N, BN)
    gsz = GM * nn; g = pid // gsz; fm = g * GM; gs = min(nm - fm, GM)
    pm = fm + (pid % gsz) % gs; pn = (pid % gsz) // gs
    kper = K // SK; k0 = ks * kper
    xd = tl.make_tensor_descriptor(x_ptr, shape=[M, K], strides=[K, 1], block_shape=[BM, BK])
    wd = tl.make_tensor_descriptor(w_ptr, shape=[N, K], strides=[K, 1], block_shape=[BN, BK])
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in tl.range(k0, k0 + kper, BK, num_stages=0 if False else None):
        a = xd.load([pm * BM, k]); b = wd.load([pn * BN, k])
        acc = tl.dot(a, b.T, acc)
    rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN)
    if SK == 1:
        tl.store(o_ptr + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16))
    else:
        tl.atomic_add(o_ptr + rm[:, None] * N + rn[None, :], acc, sem="relaxed")

def alloc(size, align, stream): return torch.empty(size, dtype=torch.int8, device="cuda")
triton.set_allocator(alloc)

def gt(fn, flush, reps=30):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn(); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): fn()
    ts = []
    for _ in range(reps):
        fl.copy_(flush.sum(dtype=torch.int32)); e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1) * 1e3)
    ts.sort(); return ts[len(ts) // 2]

shapes = [(256, 4096, 8192), (256, 8192, 4096), (256, 2304, 8192), (128, 2304, 8192), (128, 4096, 8192)]
flush = torch.ones(256 << 20, dtype=torch.int8, device="cuda"); fl = torch.zeros((), dtype=torch.int32, device="cuda")
lk = open("/tmp/okc-gpu.lock", "w")
for M, N, K in shapes:
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    ref = (x.float() @ w.float().T)
    o = torch.empty(M, N, device="cuda", dtype=torch.bfloat16); o32 = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    fcntl.flock(lk, fcntl.LOCK_EX); tb = gt(lambda: torch.matmul(x, w.T, out=o), flush); fcntl.flock(lk, fcntl.LOCK_UN)
    res = []
    for BM, BN, BK, SK, nw, ns in itertools.product([64, 128], [64, 128, 256], [64, 128], [1, 2, 4], [4, 8], [3, 4, 6]):
        if BM * BN * 4 // (nw * 32) > 256: continue
        if (BM * BK + BN * BK) * 2 * ns > 220 * 1024: continue
        if M % BM or N % BN or (K // SK) % BK: continue
        grid = (M // BM * (N // BN), SK)
        if grid[0] * SK > 264: continue
        out = o if SK == 1 else o32
        def f(out=out):
            if SK > 1: out.zero_()
            mm[grid](x, w, out, M, N, K, BM, BN, BK, SK, 8, num_warps=nw, num_stages=ns)
        try:
            f(); torch.cuda.synchronize()
        except Exception as ex:
            continue
        err = ((out.float() - ref).norm() / ref.norm()).item()
        if err > 1e-2: print("BAD", BM, BN, BK, SK, err); continue
        fcntl.flock(lk, fcntl.LOCK_EX); t = gt(f, flush, 15); fcntl.flock(lk, fcntl.LOCK_UN)
        res.append((t, BM, BN, BK, SK, nw, ns))
    res.sort()
    print(f"M{M} N{N} K{K}: nvjet {tb:.1f} us; triton best:", [f"{r[0]:.1f} BM{r[1]} BN{r[2]} BK{r[3]} SK{r[4]} w{r[5]} s{r[6]}" for r in res[:4]], flush=True)
