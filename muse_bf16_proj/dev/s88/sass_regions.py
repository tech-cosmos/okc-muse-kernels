# s88: per-region warp-stall split of an ncu source export (g24 NOW). Usage:
#   ncu -i rep.ncu-rep --page source --csv --print-source sass > sass.csv; python dev/s88/sass_regions.py sass.csv
# Groups consecutive SASS lines by "Instructions Executed" (loop body = per-stage count, task ends = per-task count, ...)
import csv, sys
rows = list(csv.reader(open(sys.argv[1]))); h = rows[1]; data = rows[2:]
iS = h.index("Warp Stall Sampling (All Samples)"); iE = h.index("Instructions Executed")
tot = sum(float(r[iS] or 0) for r in data); out = []; cur = None
for j, r in enumerate(data):
    if r[iE] != cur:
        if cur is not None: out.append((a, j - 1, cur, s, ins))
        cur, s, a, ins = r[iE], 0.0, j, []
    s += float(r[iS] or 0); ins.append(r[1].strip()[:40])
out.append((a, len(data) - 1, cur, s, ins))
for a, b, e, s, i in out:
    if s / tot > 0.006: print(f"{a:5d}-{b:5d} exec {e:>6s} {s / tot * 100:5.1f}%  {i[:3]}")
