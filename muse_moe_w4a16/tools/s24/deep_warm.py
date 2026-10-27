# s24: consumer-bound proxy for moe_twq: ONE local expert (L2-resident, 27 MB), T tokens all routed to it, warm L2,
# 5 kernel_fn calls per graph. main vs sub_deep (TQ_DEEP 1). python tools/s24/deep_warm.py
import sys, importlib, torch, fcntl
sys.path.insert(0, '.'); sys.path.insert(0, 'tools')
import layer_ab as L
A = L.submission; B = importlib.import_module("sub_deep"); st = L.st
def mkd(T, ntok):
    loc = torch.full((T, 8), -1, dtype=torch.int8); loc[:ntok, 0] = 3
    return L.mk(loc, 1)
def tw(m, data, q):
    m.TWQ = 0; m.TWQ_M = {data["config"]["bs"]: q}; m.HOTQ_M = {}; m.RG = 0; m.TWQ_NE = 0
    fn = lambda: [m.kernel_fn(data, st) for _ in range(5)]
    g = torch.cuda.CUDAGraph(); fn(); torch.cuda.synchronize()
    with torch.cuda.graph(g): fn()
    ts = []
    with open('/tmp/okc-gpu.lock', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        g.replay()
        for _ in range(20):
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1e3 / 5)
    ts.sort(); return ts[len(ts) // 2]
for T, ntok in ((40, 8), (40, 16), (40, 24), (40, 32), (64, 64)):
    d = mkd(T, ntok)
    r = {"main": [], "deep": []}
    for _ in range(3):
        for nm, m in (("main", A), ("deep", B)): r[nm].append(tw(m, d, 4))
    print(f"T{T} tokens {ntok}: " + "  ".join(f"{k} " + "/".join(f"{x:.1f}" for x in v) for k, v in r.items()), flush=True)
