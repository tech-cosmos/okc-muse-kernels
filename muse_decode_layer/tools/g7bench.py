"""gemv7 (r-proj 955bbc6 muse_bf16_proj/submission.py) vs F.linear on the layer's BF16 weights, graph + cold L2."""
import sys, fcntl, statistics, importlib.util, torch, torch.nn.functional as F
spec = importlib.util.spec_from_file_location("pj", sys.argv[1]); pj = importlib.util.module_from_spec(spec); spec.loader.exec_module(pj)
torch.manual_seed(0)
dev = "cuda"
shapes = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (1024, 4096)}
flush = torch.empty(512 << 20, dtype=torch.uint8, device=dev); sink = torch.zeros((), dtype=torch.int64, device=dev)
def t(fn, n=60):
    g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn(); torch.cuda.synchronize()
        with torch.cuda.graph(g): fn()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); ts = []
    with open("/tmp/okc-gpu.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for _ in range(n):
            torch.sum(flush.view(torch.int64), 0, out=sink); e0.record(); g.replay(); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1) * 1e3)
    return statistics.median(ts)
cfgs = dict((k, v[0][1]) for k, v in pj.BF16_GEMV.items())
for name, (n, k) in shapes.items():
    w = (torch.randn(n, k, device=dev) * 0.02).bfloat16(); wp = pj._pack(w).contiguous()
    for m in map(int, sys.argv[2:]):
        x = torch.randn(m, k, device=dev).bfloat16(); out = torch.empty(m, n, dtype=torch.bfloat16, device=dev)
        ref = F.linear(x, w)
        c = cfgs[name]; pj._mod.g7_gemv(wp, x, out, *c)
        err = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
        tc = t(lambda: F.linear(x, w)); tg = t(lambda: pj._mod.g7_gemv(wp, x, out, *c))
        print(f"{name:9s} m{m}: cublas {tc:6.2f}  g7 {tg:6.2f} us  err {err:.1e}")
