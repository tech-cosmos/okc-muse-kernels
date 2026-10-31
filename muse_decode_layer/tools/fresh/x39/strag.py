"""fresh x39: straggler CTAs per kernel instance: for each (tag, instance by entry window) the CTAs whose exit is latest,
how often each CTA is in the top-3 over replays, and median exit of max / p90 / median CTA.
python tools/fresh/x39/strag.py <pt> <tag substring> <lo> <hi>"""
import sys, torch, statistics, collections
D = torch.load(sys.argv[1]); tags = D["tags"]; sub = sys.argv[2]; lo, hi = float(sys.argv[3]), float(sys.argv[4])
TG = [i for i, t in enumerate(tags) if sub in t]
top = collections.Counter(); mx, p90, med, gap = [], [], [], []
smx = collections.defaultdict(list)
for r in D["rows"]:
    t0 = r[r[:, 1] > 0, 1].min(); meta = r[:, 0]
    m = torch.zeros(len(r), dtype=torch.bool)
    for T in TG: m |= (meta >> 48) == T
    rr = r[m]; e = (rr[:, 1] - t0).double() / 1e3; rr = rr[(e >= lo) & (e < hi)]
    ex = collections.defaultdict(float); en = {}
    for row in rr:
        c = int((row[0] >> 16) & 0xffffffff); x = (row[3] - t0).item() / 1e3; ex[c] = max(ex[c], x)
        en[c] = min(en.get(c, 1e9), (row[1] - t0).item() / 1e3)
    v = sorted(ex.items(), key=lambda kv: -kv[1])
    for c, _ in v[:3]: top[c] += 1
    xs = sorted(ex.values())
    mx.append(xs[-1]); p90.append(xs[int(.9 * (len(xs) - 1))]); med.append(xs[len(xs) // 2]); gap.append(xs[-1] - xs[-2])
    for c, x in ex.items(): smx[c].append(x - en[c])
print(f"exit max {statistics.median(mx):.1f} p90 {statistics.median(p90):.1f} med {statistics.median(med):.1f}; max - 2nd {statistics.median(gap):.2f}")
print("top-3 counts:", top.most_common(8))
dur = sorted(((statistics.median(v), c) for c, v in smx.items()), reverse=True)
print("longest CTA (exit - entry) median:", [(c, round(d, 1)) for d, c in dur[:6]], "median CTA", round(dur[len(dur) // 2][0], 1))
