"""Time n2u variants at decode 1 (sliding/full): off, R=8/4, PDL on/off."""
import sys
sys.path.insert(0, ".")
import bench
sub = bench.load_sub(sys.argv[1] if len(sys.argv) > 1 else "submission.py")
w, s = bench.weights_state(sub)
data = {f"{l} {c['layer']}": bench.make_input(c, w, 99) for l, c in bench.SCORED if l == "decode 1"}
V = [None, False, True]
res = {}
for rnd in range(4):
    for v in V:
        sub.N2U_TOKENS = () if v is None else (1,)
        sub.N2U_BOUNDARY = bool(v)
        for name, d in data.items():
            res.setdefault((name, v), []).append(bench.time_graph(lambda: sub.kernel_fn(d, s)) - bench.empty_cost())
for (name, v), x in sorted(res.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
    print(f"{name:16s} {str(v):8s} " + " ".join(f"{t:7.2f}" for t in x) + f"  min {min(x):7.2f}")
