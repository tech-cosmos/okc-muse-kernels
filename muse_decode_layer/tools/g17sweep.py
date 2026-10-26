"""Sweep gemv17 configs for the unpool slice inside the layer (T=1 / T=8 shapes)."""
import sys
sys.path.insert(0, ".")
import bench
sub = bench.load_sub(sys.argv[1] if len(sys.argv) > 1 else "submission.py")
C1 = [(1, 4, 512), (1, 4, 256), (1, 2, 512), (1, 8, 256), (1, 2, 256), (1, 8, 512)]
C8 = [(8, 4, 256), (8, 4, 128), (8, 2, 256), (8, 2, 512), (8, 8, 128)]
sub.G17_CFG = {**{-i - 1: c for i, c in enumerate(C1 + C8)}, 1: C1[0], 8: C8[0]}
w, s = bench.weights_state(sub)
shapes = [(l, c) for l, c in bench.SCORED if l in ("decode 1", "verify 1")]
data = {f"{l} {c['layer']}": bench.make_input(c, w, 99) for l, c in shapes}
res = {}
for rnd in range(3):
    for t, cs in ((1, C1), (8, C8)):
        for c in [None] + cs:
            if c is None:
                sub.G17_KEYS = ()
            else:
                sub.G17_KEYS = ("unpool_slice",)
                sub.G17_CFG[t] = c
            for name, d in data.items():
                if (t == 1) != name.startswith("decode"):
                    continue
                x = bench.time_graph(lambda: sub.kernel_fn(d, s)) - bench.empty_cost()
                res.setdefault((name, c), []).append(x)
for (name, c), v in sorted(res.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
    print(f"{name:16s} {str(c):14s} " + " ".join(f"{x:7.2f}" for x in v) + f"   min {min(v):7.2f}")
