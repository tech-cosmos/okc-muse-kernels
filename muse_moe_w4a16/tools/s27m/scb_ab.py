# s27: SCB (main) vs the previous commit (/tmp/s27/sub_old.py = git show HEAD~:submission.py) on the layer's routings
# (decode + verify, tools/data/routes.pt) and random big-M: bitwise check, cold-L2 time and warm (5 calls/graph) time.
# PYTHONPATH=/tmp/s27 python tools/s27m/scb_ab.py [rounds]
import sys, torch, importlib
sys.path.insert(0, '.'); sys.path.insert(0, 'tools')
import layer_ab as L
S = L.submission; st = L.st
O = importlib.import_module("sub_old")
libs = {"new": S._lib(), "old": O._lib()}
mods = {"new": S, "old": O}
sys.path.insert(0, 'tools/n22')
rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 2
def warm(fn, reps=40):
    import fcntl
    g = torch.cuda.CUDAGraph(); fn(); torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for _ in range(5): fn()
    ts = []
    with open('/tmp/okc-gpu.lock', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for _ in range(reps):
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1e3 / 5)
    ts.sort(); return ts[len(ts) // 2]
def run(q, d):
    st["lib"] = libs[q]; return mods[q].kernel_fn(d, st)
loads = ["decode 1", "decode 16", "decode 32", "verify 1", "verify 16", "verify 32"]
for load in loads:
    c = {"old": [], "new": []}; w = {"old": [], "new": []}; bad = 0; nd = 0
    for (l, sd), loc in sorted(L.R.items()):
        if l != load: continue
        d = L.mk(loc, sd)
        o0 = run("old", d).clone(); o0b = run("old", d).clone(); o1 = run("new", d).clone()
        nd += int(not torch.equal(o0, o0b)); bad += int(not torch.equal(o0, o1))
        for _ in range(rounds):
            for q in ("old", "new"):
                c[q].append(L.tg(lambda: run(q, d))); w[q].append(warm(lambda: run(q, d)))
    m = lambda v: sum(v) / len(v)
    print(f"{load}: not-bitwise {bad} (old vs old {nd}) | cold old {m(c['old']):.1f} new {m(c['new']):.1f} | warm old {m(w['old']):.1f} new {m(w['new']):.1f}", flush=True)
