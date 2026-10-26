"""Summarize diag24/diag19d/diag19 lines of a judge log (profiles/<job>.log) against layer main's g24 picks.
usage: python dev/s56/diag_read.py profiles/<job>.log"""
import re, sys
from collections import defaultdict

txt = open(sys.argv[1]).read().splitlines()
# layer main v101/v102 (r-layer layer.py G24_CFG/G24_SKCFG/G19_GRID/TUNE_BASE): (op, m) -> (cfg, grid) in use
LAYER = {("pooler", 16): ((16, 1024, 2, 4, 16, 2, 1), 108), ("unpooler", 16): ((16, 1024, 2, 4, 16, 2, 1), 132),
         ("qkv_gate", 16): ((16, 1024, 2, 4, 16, 2, 1), 132), ("o", 16): ((16, 512, 2, 4, 16, 2, 1), 132),
         ("pooler", 32): ((32, 512, 2, 4, 16, 2, 257), 108), ("unpooler", 32): ((32, 512, 2, 4, 16, 2, 257), 132)}
d24 = defaultdict(dict)  # (op, m) -> {(cfg, grid): us}
for l in txt:
    m = re.match(r"diag24 (\w+) m(\d+) (\([^)]*\)) \(pred ([\d.]+), rel err ([\d.e+-]+)\): (.*)", l)
    if m:
        op, mm, cfg = m.group(1), int(m.group(2)), eval(m.group(3))
        for g, us in re.findall(r"G(\d+) \+([\d.]+)", m.group(6)):
            d24[(op, mm)][(cfg, int(g))] = float(us)
for l in txt:
    if l.startswith("diag19 ") or l.startswith("diag19d") or l.startswith("tune "):
        if " m16" in l or " m32" in l or l.startswith("diag19d"):
            print(l)
print()
for k in sorted(d24):
    rows = sorted(d24[k].items(), key=lambda t: t[1])
    cur = LAYER.get(k)
    cu = d24[k].get(cur) if cur else None
    print("%-9s m%-2d layer %s G%s = %s" % (k[0], k[1], cur[0][2:] if cur else "-", cur[1] if cur else "-", cu))
    for (cfg, g), us in rows:
        print("    %-30s G%-3d %6.2f %s%s" % (cfg[1:], g, us, "SK" if cfg[6] & 256 else "  ",
                                          ("  (%+.2f vs layer)" % (us - cu)) if cu else ""))
