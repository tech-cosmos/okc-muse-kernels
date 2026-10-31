"""fresh x39: analyse lstamp.py output: per kernel instance (tag split by entry gaps > GAP us), medians over replays of
first entry / wait release (min, med, max) / exit (med, max), relative to the replay's first stamp.
python tools/fresh/x39/lan.py /tmp/x39_decode_16_sliding.pt [GAP]"""
import sys, statistics, torch
D = torch.load(sys.argv[1]); GAP = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
tags, rows = D["tags"], D["rows"]
print(f"replay median {statistics.median(D['tot']):.1f} us, {len(rows)} reps")
per = {}  # (tag, k) -> list of tuples per rep
for r in rows:
    meta, ent, wt, ex = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
    pw = r[:, 4] if r.shape[1] > 4 else torch.zeros_like(ent)
    t0 = ent[ent > 0].min().item()
    tag = (meta >> 48).tolist()
    by = {}
    for i, tg in enumerate(tag):
        by.setdefault(tg, []).append(i)
    for tg, idx in by.items():
        idx.sort(key=lambda i: ent[i].item())
        groups, cur = [], [idx[0]]
        for a, b in zip(idx, idx[1:]):
            if (ent[b] - ent[a]).item() / 1e3 > GAP:
                groups.append(cur); cur = []
            cur.append(b)
        groups.append(cur)
        for k, gidx in enumerate(groups):
            gi = torch.tensor(gidx)
            e = (ent[gi] - t0).double() / 1e3
            w = wt[gi]; w = (w[w > 0] - t0).double() / 1e3
            x = ex[gi]; x = (x[x > 0] - t0).double() / 1e3
            p = pw[gi]; p = (p[p > 0] - t0).double() / 1e3
            ctas = len(set(((meta[gi] >> 16) & 0xffffffff).tolist()))
            per.setdefault((tg, k), []).append((e.min().item(), e.median().item(),
                w.min().item() if len(w) else float('nan'), w.median().item() if len(w) else float('nan'),
                w.max().item() if len(w) else float('nan'), x.median().item() if len(x) else float('nan'),
                x.max().item() if len(x) else float('nan'), ctas, len(gidx),
                p.min().item() if len(p) else float('nan'), p.median().item() if len(p) else float('nan')))
out = []
for (tg, k), L in per.items():
    if len(L) < len(rows) * 0.8:
        continue
    m = [statistics.median(v[j] for v in L) for j in range(11)]
    out.append((m[0], f"{tags[tg][:44]:44s}#{k} ctas {int(m[7]):4d} warps {int(m[8]):5d} | entry {m[0]:6.1f} med {m[1]:6.1f} | pre {m[9]:6.1f} {m[10]:6.1f} | wait {m[2]:6.1f} {m[3]:6.1f} {m[4]:6.1f} | exit med {m[5]:6.1f} max {m[6]:6.1f}"))
for _, l in sorted(out):
    print(l)
