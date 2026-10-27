"""Per-option H200 tuner deltas vs default over the newest layer judge logs (proj s66).
python tune_stats.py <profiles dir> [n_logs=8] [T list=16,32]: options whose mean delta < -0.4 us."""
import re, glob, os, sys, statistics as S
d = sys.argv[1]; nl = int(sys.argv[2]) if len(sys.argv) > 2 else 8
Ts = {int(t) for t in (sys.argv[3] if len(sys.argv) > 3 else "16,32").split(",")}
fs = sorted(glob.glob(os.path.join(d, "*.log")), key=os.path.getmtime)[-nl:]
D = {}
for f in fs:
    for m in re.finditer(r"layer tune T=(\d+) (\w+): (.*?)\(confirm default", open(f).read()):
        parts = m.group(3).split(" | ")
        d0 = float(re.search(r"default ([\d.]+)us", parts[0]).group(1))
        for p in parts[1:]:
            mm = re.match(r"(\{.*\}) ([\d.]+)us", p.strip())
            if mm:
                D.setdefault((int(m.group(1)), m.group(2), mm.group(1)), []).append(float(mm.group(2)) - d0)
for k, v in sorted(D.items()):
    if k[0] in Ts and len(v) >= 4 and S.mean(v) < -0.4:
        print(k, len(v), "mean %.2f" % S.mean(v), "neg %d/%d" % (sum(x < 0 for x in v), len(v)), " ".join("%.1f" % x for x in v))
