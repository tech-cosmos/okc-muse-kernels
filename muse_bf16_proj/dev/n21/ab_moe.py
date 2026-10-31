"""proj n21: moe-board HOT_SL A/B. Run with cwd = r-moe/muse_moe_w4a16 (its bench/reference):
python ab_moe.py <base dir> <sl dir> (copies kept in /tmp/n21keep: base = r-moe 0d36b3a submission.py, sl3 = + hotsl_moe_board.diff) bs seed0|s1,s2,.. nseeds rounds. Prints max local nt, bitwise, interleaved replay medians (us)."""
import sys, statistics, torch, importlib.util
sys.path.insert(0, ".")
import bench, reference
def load(n, p):
    spec = importlib.util.spec_from_file_location(n, p + "/submission.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
A_, B_ = load("suba", sys.argv[1]), load("subb", sys.argv[2])
bs, nrnd = int(sys.argv[3]), int(sys.argv[6])
SEEDS = [int(x) for x in sys.argv[4].split(",")] if "," in sys.argv[4] else list(range(int(sys.argv[4]), int(sys.argv[4]) + int(sys.argv[5])))
w = reference.generate_weights(reference.WEIGHTS_SEED)
sta = A_.prepare(w); stb = B_.prepare(w)
rows = []
for sd in SEEDS:
    d = reference.generate_input(bs=bs, sharding="ep8", seed=sd)
    ids = d["topk_ids"].flatten(); cnt = torch.bincount(ids[ids < 32], minlength=32); mx = int(cnt.max())
    oa = A_.kernel_fn(d, sta).clone(); ob = B_.kernel_fn(d, stb).clone(); torch.cuda.synchronize()
    eq = torch.equal(oa, ob)
    ok = reference.check_implementation(d, ob, w) if hasattr(reference, "check_implementation") else None
    a, b = [], []
    for r in range(nrnd):
        for k in ((0, 1) if r % 2 == 0 else (1, 0)):
            t = bench.time_graph(lambda: (A_.kernel_fn(d, sta) if k == 0 else B_.kernel_fn(d, stb)), 30)
            (a if k == 0 else b).append(t)
    ta, tb = statistics.median(a), statistics.median(b)
    rows.append((mx, ta, tb))
    print(f"seed {sd} maxnt {mx:3d} n>64 {int((cnt > 64).sum())} bitwise {eq} chk {ok if ok is None else bool(ok[0]) if isinstance(ok, tuple) else ok}: base {ta:7.2f} sl {tb:7.2f} d {tb - ta:+6.2f}", flush=True)
for lab, f in (("maxnt>128", lambda r: r[0] > 128), ("maxnt<=128", lambda r: r[0] <= 128)):
    rr = [r for r in rows if f(r)]
    if rr: print(f"bs {bs} {lab} n={len(rr)} base {statistics.mean(r[1] for r in rr):7.2f} sl {statistics.mean(r[2] for r in rr):7.2f} d {statistics.mean(r[2] - r[1] for r in rr):+6.2f} us")
