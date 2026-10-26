"""Interleaved A/B of module-flag settings in ONE process/state: python tools/ab_flag.py <file> <filter> <rounds> "<stmt A>" "<stmt B>" ...
Each stmt is exec'd in the submission module's namespace (e.g. "G19_ON = set()"), then every scored shape matching
<filter> is captured and timed (bench.time_graph: cold L2, median of 40 replays). Prints per-shape medians over rounds."""
import sys, statistics
sys.path.insert(0, ".")
import bench
sub = bench.load_sub(sys.argv[1]); flt = sys.argv[2]; rounds = int(sys.argv[3]); stmts = sys.argv[4:]
w, st = bench.weights_state(sub)
names = [(f"{l} {c['layer']}", c) for l, c in bench.SCORED if flt in f"{l} {c['layer']}"]
res = {(n, i): [] for n, _ in names for i in range(len(stmts))}
data = {n: bench.make_input(c, w, 99) for n, c in names}
e0 = bench.empty_cost()
for r in range(rounds):
    for i, s in enumerate(stmts):
        exec(s, vars(sub))
        for n, _ in names:
            res[(n, i)].append(bench.time_graph(lambda: sub.kernel_fn(data[n], st)) - e0)
for n, _ in names:
    print(f"{n:18s}", "  ".join(f"[{i}] {statistics.median(res[(n, i)]):7.2f} ({min(res[(n, i)]):7.2f})" for i in range(len(stmts))))
