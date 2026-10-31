"""fresh x39: per-CTA view of one kernel instance from lstamp output: entry / first pre / first release / exit per CTA,
quantiles over CTAs (median over replays). python tools/fresh/x39/cta.py <pt> <tag substring> <min entry us> [max entry us]"""
import sys, torch, statistics, collections
D = torch.load(sys.argv[1]); tags = D["tags"]; sub = sys.argv[2]; lo = float(sys.argv[3]); hi = float(sys.argv[4]) if len(sys.argv) > 4 else 1e9
TG = [i for i, t in enumerate(tags) if sub in t]
Q = (0, .1, .25, .5, .75, .9, 1)
acc = collections.defaultdict(list)
for r in D["rows"]:
    t0 = r[r[:, 1] > 0, 1].min(); meta = r[:, 0]
    m = torch.zeros(len(r), dtype=torch.bool)
    for T in TG: m |= (meta >> 48) == T
    rr = r[m]; e = (rr[:, 1] - t0).double() / 1e3
    sel = (e >= lo) & (e < hi); rr = rr[sel]
    cta = ((rr[:, 0] >> 16) & 0xffffffff).tolist()
    by = collections.defaultdict(list)
    for c, row in zip(cta, rr): by[c].append(row)
    per = []
    for c, L in by.items():
        L = torch.stack(L); f = lambda k: ((L[L[:, k] > 0, k] - t0).double() / 1e3)
        en, pr, wt, ex = f(1), f(4), f(2), f(3)
        per.append((en.min().item(), pr.min().item() if len(pr) else float('nan'), wt.min().item() if len(wt) else float('nan'), ex.max().item()))
    per.sort()
    for j, nm in enumerate(("entry", "pre", "release", "exit")):
        v = sorted(p[j] for p in per)
        acc[nm].append([v[int(q * (len(v) - 1))] for q in Q])
    late = [p for p in per if p[0] >= per[int(0.9 * (len(per) - 1))][0]]
    acc["late-exit"].append([statistics.median(p[3] for p in late)] * len(Q))
    acc["n"].append([len(per)] * len(Q))
print("quantiles over CTAs", Q)
for nm, L in acc.items():
    print(f"{nm:10s}", [round(statistics.median(x[i] for x in L), 1) for i in range(len(Q))])
