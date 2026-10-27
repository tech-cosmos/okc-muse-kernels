"""ab_files with TUNE disabled (both files at TUNE_BASE defaults): python ab2.py <filter> <rounds> A.py B.py; env SEEDS."""
import sys, os, statistics, importlib.util, torch
sys.path.insert(0, ".")
import bench, reference
flt = sys.argv[1]; rounds = int(sys.argv[2]); files = sys.argv[3:]
SEEDS = [int(x) for x in os.environ.get("SEEDS", "99").split(",")]
w = reference.generate_weights(reference.WEIGHTS_SEED)
subs = []
for i, f in enumerate(files):
    spec = importlib.util.spec_from_file_location(f"sub{i}", f)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    m.TUNE = {}
    subs.append((m, m.prepare(w)))
names = [(f"{l} {c['layer']} s{sd}", c, sd) for l, c in bench.SCORED if flt in f"{l} {c['layer']}" for sd in SEEDS]
data = {n: bench.make_input(c, w, sd) for n, c, sd in names}
# correctness: outputs bitwise equal between files
for n, c, sd in names:
    outs = [m.kernel_fn(data[n], st) for m, st in subs]
    o = [x if torch.is_tensor(x) else x[0] for x in outs]
    print(n, "bitwise eq:", all(torch.equal(o[0], z) for z in o[1:]))
res = {(n, i): [] for n, _, _ in names for i in range(len(files))}
e0 = bench.empty_cost()
for r in range(rounds):
    order = range(len(subs)) if r % 2 == 0 else reversed(range(len(subs)))
    for i in order:
        m, st = subs[i]
        for n, _, _ in names:
            res[(n, i)].append(bench.time_graph(lambda: m.kernel_fn(data[n], st)) - e0)
for n, _, _ in names:
    print(f"{n:24s}", "  ".join(f"[{i}] {statistics.median(res[(n, i)]):7.2f} ({min(res[(n, i)]):7.2f})" for i in range(len(files))))
