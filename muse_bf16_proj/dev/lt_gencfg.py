import sys, glob, torch
sys.path.insert(0, "/tmp/pj")
from lt import mod
cfg = {}
for f in sorted(glob.glob("/tmp/pj/lte_*_*.pt")):
    _, m, layer = f[:-3].split("/")[-1].split("_", 2)
    top = torch.load(f)["top"]
    seen = []
    for t, a in top:
        d = mod.lt_desc(a).tolist()
        c = (d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[8])
        if c not in seen:
            seen.append(c)
        if len(seen) == 5: break
    cfg[(layer, int(m))] = seen
    print(layer, m, [round(t, 2) for t, _ in top[:5]], file=sys.stderr)
print("{" + ",\n           ".join(f"{k!r}: {v!r}" for k, v in sorted(cfg.items())) + "}")
