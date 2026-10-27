# s24 TQ_DEEP: moe_twq consumer with 4 A sets / 3 wgmma groups in flight (sub_deep.py = submission.py with TQ_DEEP 1)
# vs main: bitwise on layer v16/v32 + random bs33-257 (twq q4 / s66h32ne), then H100 timing cold + warm L2 (consumer proxy).
import sys, importlib, torch, fcntl
sys.path.insert(0, '.'); sys.path.insert(0, 'tools')
import layer_ab as L, reference
A = L.submission; B = importlib.import_module("sub_deep"); st = L.st
QS = [4, 536953348 | 1 << 21]
def run(m, data, q):
    m.TWQ = 0; m.TWQ_M = {data["config"]["bs"]: q}; m.HOTQ_M = {}; m.RG = 0
    g = torch.cuda.CUDAGraph(); m.kernel_fn(data, st); torch.cuda.synchronize()
    with torch.cuda.graph(g): o = m.kernel_fn(data, st)
    g.replay(); torch.cuda.synchronize(); return o.clone()
cases = [(f"rand bs{bs} s{sd} {dist}", lambda bs=bs, sd=sd, dist=dist: reference.generate_input(sharding="ep8", bs=bs, seed=sd, **({"dist": dist} if dist else {})))
         for bs in (33, 128, 256, 257) for sd in (1,) for dist in (None, "skewed")]
cases += [(f"layer {l} s{sd}", lambda loc=loc, sd=sd: L.mk(loc, sd)) for (l, sd), loc in sorted(L.R.items()) if l in ("verify 16", "verify 32")]
bad = n = 0
for name, mkd in cases:
    data = mkd()
    for q in QS:
        n += 1
        if not torch.equal(run(A, data, q), run(B, data, q)): bad += 1; print("DIFF", name, q, flush=True)
print(f"deep_t bitwise: {bad}/{n} differ", flush=True)
def tw(fn, warm):
    g = torch.cuda.CUDAGraph(); fn(); torch.cuda.synchronize()
    with torch.cuda.graph(g): fn()
    ts = []
    with open('/tmp/okc-gpu.lock', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for _ in range(30):
            if not warm: torch.sum(L.flush.view(torch.int64), 0, out=L.sink)
            else: g.replay()
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1e3)
    ts.sort(); return ts[len(ts) // 2]
for load in ("verify 16", "verify 32"):
    for q in QS:
        for warm in (0, 1):
            r = {}
            for rnd in range(2):
                for nm, m in (("main", A), ("deep", B)):
                    m.TWQ = 0; m.HOTQ_M = {}; m.RG = 0
                    ts = []
                    for (l, sd), loc in sorted(L.R.items()):
                        if l != load: continue
                        data = L.mk(loc, sd); m.TWQ_M = {data["config"]["bs"]: q}
                        ts.append(tw(lambda: m.kernel_fn(data, st), warm))
                    r.setdefault(nm, []).append(sum(ts) / len(ts))
            print(f"{load} q {q} {'warm' if warm else 'cold'}: " + "  ".join(f"{k} " + "/".join(f"{x:.1f}" for x in v) for k, v in r.items()), flush=True)
