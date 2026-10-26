"""Summarize a moe DIAG job log (n11): python tools/n11/diag_read.py <profiles/submission.<job>.log>

Per load: every diag12 variant's MEAN and its delta vs the load's first variant (gs8 for decode, off for verify),
plus per-seed spread; then the diag9s stream probe (t, TB/s) and gs8 - nm / gs8 - tkp gaps (n9/n10 reading rules).
Layer score weight per us (sliding|full, STATE-layer s12): d1 .161|.056 d16 .072|.025 d32 .056|.020 v1 .096|.033
v16 .041|.014 v32 .031|.011 -> the 'pct' column = delta x sliding weight (rough % of layer score, negative = gain).
"""
import re, sys
from collections import defaultdict

W = {"decode 1": .161, "decode 16": .072, "decode 32": .056, "verify 1": .096, "verify 16": .041, "verify 32": .031}
txt = open(sys.argv[1]).read()
mean, seeds, order = {}, defaultdict(list), defaultdict(list)
for load, n, t in re.findall(r"diag12 MEAN (\w+ \d+) (\S+): ([\d.]+) us", txt):
    mean[(load, n)] = float(t)
    order[load].append(n)
for load, s, n, t in re.findall(r"diag12 (\w+ \d+) s(\d+) (\S+): ([\d.]+) us", txt):
    seeds[(load, n)].append(float(t))
for load, ns in order.items():
    b = mean[(load, ns[0])]
    print(f"== {load} (base {ns[0]} {b:.1f} us)")
    for n in ns:
        v, d = mean[(load, n)], mean[(load, n)] - b
        sp = seeds[(load, n)]
        print(f"  {n:8s} {v:7.1f} {d:+6.1f} us  pct {d * W.get(load, 0):+.2f}  seeds {' '.join(f'{x:.1f}' for x in sp)}")
st = re.findall(r"diag9s MEAN (\w+ \d+) (\S+): ([\d.]+) us,? ([\d.]+) TB/s", txt)
if st:
    print("== diag9s stream probe (t us / TB/s over unique bytes)")
    for load, n, t, bw in st:
        print(f"  {load:10s} {n:7s} {float(t):7.1f} {float(bw):5.2f}")
    for load in W:
        g, nm = mean.get((load, "gs8")), mean.get((load, "nm"))
        tkp = {n: float(t) for l, n, t, _ in st if l == load}.get("tkp")
        if g and nm:
            print(f"  {load}: gs8 - nm = {g - nm:+.1f} us (max consumer-side gain; >= 8 at d16/d32 -> GS16/dq work pays)"
                  + (f", gs8 - tkp = {g - tkp:+.1f} us (tk2f vs its own copy pattern)" if tkp else ""))
