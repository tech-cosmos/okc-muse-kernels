"""proj s99: in-layer sweep of g30k's grid G (TUNE-off submission copy): python gsweep.py <sub> <T> <rounds> G1,G2,.. ;
0 = g30k off; every G must be in G30K_CFG's grids for T. Run from a r-layer muse_decode_layer/ dir (bench.py)."""
import sys, statistics
sys.path.insert(0, ".")
import bench, torch
sub = bench.load_sub(sys.argv[1]); T = int(sys.argv[2]); R = int(sys.argv[3]); GS = [int(g) for g in sys.argv[4].split(",")]
w, st = bench.weights_state(sub)
ent = st["g30k"][("pooler", T)]  # {G: entry} (prepared per G30K_CFG grids)
names = [(f"{l} {c['layer']} s{sd}", c, sd) for l, c in bench.SCORED if l == f"decode {T}" for sd in (99, 7)]
data = {nm: bench.make_input(c, w, sd) for nm, c, sd in names}
e0 = bench.empty_cost()
res = {}
for r in range(R):
    for G in (GS if r % 2 == 0 else GS[::-1]):
        sub.G30K_DEFAULT = {("pooler", T): G}
        for nm, c, sd in names:
            res.setdefault((nm, G), []).append(bench.time_graph(lambda: sub.kernel_fn(data[nm], st)) - e0)
for nm, c, sd in names:
    print(f"{nm:22s}", "  ".join(f"G{G} {statistics.median(res[(nm, G)]):7.2f}" for G in GS), flush=True)
