"""o26: routing-variance audit. python tools/o26/rv.py <bs> <nseeds> [dist]
Per seed: distinct local experts, max tokens/expert, our graph time (cold L2). Prints a linear fit
t = a + b*experts and the residuals, to find routing-triggered slow modes (held-out reruns)."""
import sys, os, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch, bench, reference, submission
bs, n = int(sys.argv[1]), int(sys.argv[2]); dist = sys.argv[3] if len(sys.argv) > 3 else "normal"
_, st = bench.state()
rows = []
for sd in range(5000, 5000 + n):
    data = reference.generate_input(bs=bs, sharding="ep8", seed=sd, dist=dist)
    ids = data["topk_ids"]; loc = ids[(ids >= 0) & (ids < 32)]
    cnt = torch.bincount(loc.long(), minlength=32)
    ne, mx = int((cnt > 0).sum()), int(cnt.max())
    t = bench.time_graph(lambda: submission.kernel_fn(data, st), 20) - bench.empty_cost()
    g = int(((cnt + 7) // 8 - (cnt > 0).long()).sum())
    rows.append((sd, ne, mx, t, g))
xs = [r[1] for r in rows]; ys = [r[3] for r in rows]
mx_, my_ = statistics.mean(xs), statistics.mean(ys)
vx = sum((x - mx_) ** 2 for x in xs)
b = sum((x - mx_) * (y - my_) for x, y in zip(xs, ys)) / vx if vx else 0.0
a = my_ - b * mx_
res = [y - (a + b * x) for x, y in zip(xs, ys)]
sd_ = statistics.pstdev(res)
print(f"bs {bs} {dist} n {n}: fit t = {a:.1f} + {b:.2f}*experts  resid sd {sd_:.2f} us; t med {statistics.median(ys):.1f} min {min(ys):.1f} max {max(ys):.1f}")
import numpy as np
A = np.array([[1, r[1], r[4]] for r in rows], float); Y = np.array(ys)
c, *_ = np.linalg.lstsq(A, Y, rcond=None); r2 = Y - A @ c
print(f"  2-var fit t = {c[0]:.1f} + {c[1]:.2f}*experts + {c[2]:.2f}*extra8groups  resid sd {r2.std():.2f}")
for r, e in sorted(zip(rows, r2), key=lambda z: -z[1])[:6]:
    print(f"  seed {r[0]} experts {r[1]} maxtok {r[2]} xg {r[4]} t {r[3]:.1f} resid {e:+.1f}")
