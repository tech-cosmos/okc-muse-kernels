"""proj n36: T=32 variance fit from route_var.py ROW lines: us ~ D + R + n16 (experts with > 16 tokens) + max."""
import sys, collections, numpy as np
rows = collections.defaultdict(list)
for l in open(sys.argv[1]):
    if l.startswith("ROW"):
        kv = dict(x.split("=", 1) for x in l.split()[1:]); rows[kv["shape"]].append(kv)
for sh, rs in rows.items():
    us = np.array([float(r["us"]) for r in rs]); D = np.array([int(r["D"]) for r in rs]); R = np.array([int(r["R"]) for r in rs])
    cnt = [list(map(int, r["cnt"].split(","))) for r in rs]
    n16 = np.array([sum(c > 16 for c in x) for x in cnt]); mx = np.array([x[0] for x in cnt])
    for name, cols in [("D", [D]), ("D+R", [D, R]), ("D+R+n16", [D, R, n16]), ("D+R+n16+max", [D, R, n16, mx])]:
        X = np.stack([np.ones_like(us)] + cols, 1); c, *_ = np.linalg.lstsq(X, us, rcond=None); res = us - X @ c
        print(f"{sh} {name:12s} coef {' '.join(f'{v:+.2f}' for v in c)} resid sd {res.std():.2f} R2 {1 - res.var() / us.var():.2f}")
    X = np.stack([np.ones_like(us), D, R], 1); c, *_ = np.linalg.lstsq(X, us, rcond=None); res = us - X @ c
    byr = collections.defaultdict(list)
    for r, e in zip(rs, res): byr[r["rs"]].append(e)
    m = sorted((np.mean(v), k) for k, v in byr.items())
    print(f"  n={len(us)} us {us.mean():.1f} [{us.min():.1f},{us.max():.1f}] D {D.min()}-{D.max()} n16 {n16.min()}-{n16.max()} max {mx.min()}-{mx.max()}")
    print("  router resid means (D+R fit): " + " ".join(f"{k}:{v:+.1f}" for v, k in m) + f"  sd {np.std([a for a, _ in m]):.2f}")
    for i in np.argsort(-np.abs(res))[:3]:
        print(f"  outlier rs {rs[i]['rs']} in {rs[i]['in']} us {us[i]:.1f} resid {res[i]:+.1f} D {D[i]} R {R[i]} n16 {n16[i]} cnt {rs[i]['cnt']}")
