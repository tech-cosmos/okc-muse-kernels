"""Interleaved A/B of module-flag settings in ONE process/state: python tools/ab_flag.py <file> <filter> <rounds> "<stmt A>" "<stmt B>" ...
Each stmt is exec'd in the submission module's namespace (e.g. "G19_ON = set()"), then every scored shape matching
<filter> is captured and timed (bench.time_graph: cold L2, median of 40 replays). Prints per-shape medians over rounds."""
import sys, statistics
sys.path.insert(0, ".")
import bench
sub = bench.load_sub(sys.argv[1]); flt = sys.argv[2]; rounds = int(sys.argv[3]); stmts = sys.argv[4:]
w, st = bench.weights_state(sub); sub._ST = st; print({k: v for k, v in st["tune"].items() if k[0] in (8, 16, 32)})
names = [(f"{l} {c['layer']}", c) for l, c in bench.SCORED if flt in f"{l} {c['layer']}"]
import os
SEEDS = [int(x) for x in os.environ.get("SEEDS", "99").split(",")]  # s58: several input seeds (routing) per shape
names = [(f"{n} s{sd}" if len(SEEDS) > 1 else n, c, sd) for n, c in names for sd in SEEDS]
data = {n: bench.make_input(c, w, sd) for n, c, sd in names}
names = [(n, c) for n, c, sd in names]
res = {(n, i): [] for n, _ in names for i in range(len(stmts))}

e0 = bench.empty_cost()
for r in range(rounds):
    for i, s in enumerate(stmts):
        exec(s, vars(sub))
        for n, _ in names:
            res[(n, i)].append(bench.time_graph(lambda: sub.kernel_fn(data[n], st)) - e0)
for n, _ in names:
    print(f"{n:18s}", "  ".join(f"[{i}] {statistics.median(res[(n, i)]):7.2f} ({min(res[(n, i)]):7.2f})" for i in range(len(stmts))))
