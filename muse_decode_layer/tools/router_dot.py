"""FP32 router at T<=8 as tl.dot(input_precision='ieee') (FP32 FMA), split-K partials, vs _router_fma (H100, cold L2)."""
import sys, fcntl, statistics, itertools, importlib.util, torch, triton, triton.language as tl
spec = importlib.util.spec_from_file_location("sub", sys.argv[1]); sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)


@triton.jit
def _router_dot(h_ptr, w_ptr, part_ptr, T, K: tl.constexpr, E: tl.constexpr, BT: tl.constexpr, BE: tl.constexpr,
                BK: tl.constexpr, KS: tl.constexpr):
    pe = tl.program_id(0)
    ks = tl.program_id(1)
    e = pe * BE + tl.arange(0, BE)
    t = tl.arange(0, BT)
    t_ok = t < T
    acc = tl.zeros((BT, BE), dtype=tl.float32)
    k_per = K // KS
    for k0 in tl.range(ks * k_per, ks * k_per + k_per, BK):
        k = k0 + tl.arange(0, BK)
        w = tl.load(w_ptr + e[None, :] * K + k[:, None])
        h = tl.load(h_ptr + t[:, None] * K + k[None, :], mask=t_ok[:, None], other=0.0)
        acc = tl.dot(h, w, acc, input_precision="ieee")
    tl.store(part_ptr + (ks * T + t[:, None]) * E + e[None, :], acc, mask=t_ok[:, None])


dev = "cuda"
flush = torch.empty(512 << 20, dtype=torch.uint8, device=dev); sink = torch.zeros((), dtype=torch.int64, device=dev)
def t(fn, n=30):
    g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn(); torch.cuda.synchronize()
        with torch.cuda.graph(g): fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); ts = []
    with open("/tmp/okc-gpu.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for _ in range(n):
            torch.sum(flush.view(torch.int64), 0, out=sink); e0.record(); g.replay(); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1) * 1e3)
    return statistics.median(ts)
empty = t(lambda: None)
W = torch.randn(256, 8192, device=dev) * 0.02
for T in (1, 8):
    h = torch.randn(T, 8192, device=dev)
    ref = (h.double() @ W.double().T).float()
    part = torch.empty((8, T, 256), device=dev)
    base = t(lambda: sub._router_fma[(128, 8)](h, W, part, T, K=8192, E=256, BT=triton.next_power_of_2(T), BE=2, BK=1024, KS=8, num_warps=4)) - empty
    res = []
    for BE, BK, KS, warps, st in itertools.product((16, 32, 64), (32, 64, 128), (4, 8, 16, 32), (2, 4), (2, 3, 4)):
        if 8192 % (KS * BK) or (256 // BE) * KS < 64:
            continue
        part = torch.empty((KS, T, 256), device=dev)
        fn = lambda: _router_dot[(256 // BE, KS)](h, W, part, T, K=8192, E=256, BT=16, BE=BE, BK=BK, KS=KS, num_warps=warps, num_stages=st)
        try:
            fn(); torch.cuda.synchronize()
        except Exception as ex:
            continue
        err = (part.sum(0) - ref).abs().max().item()
        if err > 1e-4:
            print("BAD", BE, BK, KS, warps, err); continue
        res.append((t(fn) - empty, (BE, BK, KS, warps, st)))
    res.sort()
    print(f"T={T}: fma {base:.2f}  dot best {[(round(a, 2), c) for a, c in res[:6]]}", flush=True)
