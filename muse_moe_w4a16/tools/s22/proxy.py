# H200 proxy: MOE_GRID=96 python proxy.py "verify 16,verify 32" 'name=python-dict-of-globals' ...
# times experts on the layer's routings (layer_ab), interleaving variants, ROUNDS rounds; prints mean over seeds per round
import sys, os
sys.path.insert(0, '/home/daytona/r-moe/muse_moe_w4a16/tools')
import layer_ab as L
sub = L.submission
loads = sys.argv[1].split(",")
vs = []
for a in sys.argv[2:]:
    n, d = a.split("=", 1); vs.append((n, eval(d)))
R = int(os.environ.get("ROUNDS", "2"))
for load in loads:
    res = {n: [] for n, _ in vs}
    items = [(sd, loc) for (l, sd), loc in sorted(L.R.items()) if l == load]
    for r in range(R):
        for n, d in vs:
            old = {k: getattr(sub, k) for k in d}
            for k, v in d.items(): setattr(sub, k, v)
            ts = []
            for sd, loc in items:
                data = L.mk(loc, sd); ts.append(L.tg(lambda: sub.kernel_fn(data, L.st)))
            for k, v in old.items(): setattr(sub, k, v)
            res[n].append(sum(ts) / len(ts))
    print(f"G{os.environ.get('MOE_GRID','-')} {load}: " + "  ".join(f"{n} " + "/".join(f"{x:.1f}" for x in res[n]) for n, _ in vs), flush=True)
