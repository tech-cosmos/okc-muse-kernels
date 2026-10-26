import sys, statistics
sys.path.insert(0, ".")
import bench
sub = bench.load_sub(sys.argv[1]); flt = sys.argv[2]; rounds = int(sys.argv[3]); vals = [int(v) for v in sys.argv[4:]]
w, st = bench.weights_state(sub)
print("tune", {k: v for k, v in st.get("tune", {}).items() if k[0] >= 128})
names = [(f"{l} {c['layer']}", c) for l, c in bench.SCORED if flt in f"{l} {c['layer']}"]
res = {(n, i): [] for n, _ in names for i in range(len(vals))}
data = {n: bench.make_input(c, w, 99) for n, c in names}
e0 = bench.empty_cost()
for r in range(rounds):
    for i, v in enumerate(vals):
        for k, d in st.setdefault("tune", {}).items():
            if k[0] >= 128: d.pop("twf_split", None); d.update({"twf_split": v} if v else {})
        for m in (128, 256):
            for s in (True, False):
                st["tune"].setdefault((m, s), {}).update({"twf_split": v} if v else {})
        for n, _ in names:
            res[(n, i)].append(bench.time_graph(lambda: sub.kernel_fn(data[n], st)) - e0)
for n, _ in names:
    print(f"{n:18s}", "  ".join(f"[{vals[i]}] {statistics.median(res[(n, i)]):7.2f} ({min(res[(n, i)]):7.2f})" for i in range(len(vals))))
