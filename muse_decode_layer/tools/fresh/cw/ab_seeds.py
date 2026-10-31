"""fresh x12: tools/ab_files.py over several input seeds (env SEEDS=99,7,..., mean of per-seed medians). Interleaved A/B of whole submission files in ONE process: python tools/ab_files.py <filter> <rounds> <file A> <file B> ...
Each file gets its own module + prepare() state (same weights); every scored shape matching <filter> is captured and
timed (bench.time_graph: cold L2, median of 40 replays) per round, files interleaved. Prints per-shape medians."""
import sys, statistics, importlib.util, torch
sys.path.insert(0, ".")
import bench, reference
flt = sys.argv[1]; rounds = int(sys.argv[2]); files = sys.argv[3:]
w = reference.generate_weights(reference.WEIGHTS_SEED)
subs = []
for i, f in enumerate(files):
    spec = importlib.util.spec_from_file_location(f"sub{i}", f)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    subs.append((m, m.prepare(w)))
names = [(f"{l} {c['layer']}", c) for l, c in bench.SCORED if flt in f"{l} {c['layer']}"]
SEEDS = [int(x) for x in __import__("os").environ.get("SEEDS", "99").split(",")]
data = {(n, sd): bench.make_input(c, w, sd) for n, c in names for sd in SEEDS}
res = {(n, i, sd): [] for n, _ in names for i in range(len(files)) for sd in SEEDS}
e0 = bench.empty_cost()
for r in range(rounds):
    for i, (m, st) in enumerate(subs):
        for n, _ in names:
            [res[(n, i, sd)].append(bench.time_graph(lambda: m.kernel_fn(data[(n, sd)], st)) - e0) for sd in SEEDS]
mm = lambda n, i: statistics.mean(statistics.median(res[(n, i, sd)]) for sd in SEEDS)
for n, _ in names:
    print(f"{n:18s}", "  ".join(f"[{i}] {mm(n, i):7.2f}" for i in range(len(files))))
