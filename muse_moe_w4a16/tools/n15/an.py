# summarize diagtl lines: per (load, variant) mean of exit max, loss, busy/G, last g13 end
import re, sys, collections
d = collections.defaultdict(list)
for l in open(sys.argv[1]):
    m = re.search(r"diagtl (\w+ \d+) s(\d+) (\w+): G (\d+) items (\d+) start med/max ([\d.]+)/([\d.]+) exit min/med/max ([\d.]+)/([\d.]+)/([\d.]+) busy/G ([\d.]+) loss ([-\d.]+) last g13 end ([\d.]+)", l)
    if m:
        d[(m[1], m[3])].append([float(m[i]) for i in (8, 9, 10, 11, 12, 13)])
for k, v in d.items():
    n = len(v); mean = [sum(x[i] for x in v) / n for i in range(6)]
    print(k, "exit min/med/max %.1f/%.1f/%.1f busy/G %.1f loss %.1f g13end %.1f" % tuple(mean))
