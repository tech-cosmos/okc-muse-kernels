"""Fit us ~ D + nh + nb (+R) per shape from route_var.py ROW lines; router-level residual means."""
import sys, re, collections, numpy as np
rows = collections.defaultdict(list)
for f in sys.argv[1:]:
    for l in open(f):
        if not l.startswith("ROW"): continue
        kv = dict(x.split("=", 1) for x in l.split()[1:])
        rows[kv["shape"]].append(kv)
for sh, rs in rows.items():
    us = np.array([float(r["us"]) for r in rs])
    D = np.array([int(r["D"]) for r in rs]); R = np.array([int(r["R"]) for r in rs])
    nh = np.array([int(r["nh"]) for r in rs]); nb = np.array([int(r["nb"]) for r in rs])
    mx = np.array([int(r["max"]) for r in rs])
    X = np.stack([np.ones_like(us), D, nh, nb, R], 1)
    coef, *_ = np.linalg.lstsq(X, us, rcond=None)
    res = us - X @ coef
    print(f"{sh}: n={len(us)} us {us.mean():.1f} sd {us.std():.1f} [{us.min():.1f},{us.max():.1f}]  D {D.min()}-{D.max()} nh {nh.min()}-{nh.max()} nb {nb.min()}-{nb.max()} max {mx.min()}-{mx.max()}")
    print(f"   fit: {coef[0]:.1f} + {coef[1]:.2f}*D + {coef[2]:.2f}*nh + {coef[3]:.2f}*nb + {coef[4]:.3f}*R ; resid sd {res.std():.2f}  (R2 {1 - res.var() / us.var():.2f})")
    byr = collections.defaultdict(list)
    for r, e in zip(rs, res): byr[r["rs"]].append(e)
    m = sorted((np.mean(v), k) for k, v in byr.items())
    print(f"   router resid means: min {m[0][0]:+.1f} (rs {m[0][1]}) max {m[-1][0]:+.1f} (rs {m[-1][1]}); sd of means {np.std([a for a, _ in m]):.2f}")
    big = np.argsort(-np.abs(res))[:3]
    for i in big:
        print(f"   outlier rs {rs[i]['rs']} in {rs[i]['in']} us {us[i]:.1f} resid {res[i]:+.1f} D {D[i]} nh {nh[i]} nb {nb[i]} max {mx[i]} cnt {rs[i]['cnt'][:60]}")
