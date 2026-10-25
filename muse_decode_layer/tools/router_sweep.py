"""_router_fma (FP32 router) config sweep at T=1/8 + routing, standalone graph, cold L2 (H100)."""
import sys, fcntl, statistics, itertools, importlib.util, torch
spec = importlib.util.spec_from_file_location("sub", sys.argv[1]); sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
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
    res = []
    for BE, BK, KS, warps in itertools.product((2, 4, 8, 16), (64, 128, 256, 512, 1024), (4, 8, 16, 32), (2, 4, 8)):
        BT = triton_np2 = 1 << max(0, T - 1).bit_length()
        if BT * BE * BK // (warps * 32) > 128 or 8192 % (KS * BK):
            continue
        part = torch.empty((KS, T, 256), device=dev)
        fn = lambda: sub._router_fma[(256 // BE, KS)](h, W, part, T, K=8192, E=256, BT=BT, BE=BE, BK=BK, KS=KS, num_warps=warps)
        fn(); torch.cuda.synchronize()
        err = (part.sum(0) - ref).abs().max().item()
        if err > 1e-3:
            print("BAD", BE, BK, KS, warps, err); continue
        res.append((t(fn) - empty, (BE, BK, KS, warps)))
    res.sort()
    cur = [r for r in res if r[1] == ((2, 1024, 8, 4))]
    print(f"T={T}: current {cur}  best {[(round(a, 2), c) for a, c in res[:6]]}", flush=True)
