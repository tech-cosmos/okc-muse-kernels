"""Summarize heldout.py output: per shape, main us at the scored seed vs the other seeds (mean over routers), and each
variant's mean / min / max delta vs main over all (router, seed) inputs. python tools/n11/summ.py <out.txt>"""
import re, sys, statistics as S
rows = {}
for l in open(sys.argv[1]):
    if not l.startswith("R "):
        continue
    p = [x.strip() for x in l.split("|")]
    shape, seed = p[1], int(p[2].split()[1])
    v = dict(zip(p[-1].split()[::2], map(float, p[-1].split()[1::2])))
    rows.setdefault(shape, []).append((p[0], seed, v))
W = {"sliding": 46, "full": 16}
for shape, rs in rows.items():
    sc = [v["main"] for _, s, v in rs if s < 1000]; ot = [v["main"] for _, s, v in rs if s >= 1000]
    out = f"{shape:17s} n{len(rs):3d} scored {S.mean(sc):6.1f} other {S.mean(ot):6.1f} ({(S.mean(ot) / S.mean(sc) - 1) * 100:+.1f}%) range {min(sc + ot):.0f}-{max(sc + ot):.0f}"
    for vn in [k for k in rs[0][2] if k != "main"]:
        d = [v[vn] - v["main"] for _, _, v in rs if vn in v]
        out += f" | {vn} {S.mean(d):+.1f} [{min(d):+.1f},{max(d):+.1f}] {sum(x < 0 for x in d)}/{len(d)}<0"
    print(out)
