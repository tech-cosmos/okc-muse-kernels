"""Print default vs pr32 (and best other) options from H200 'layer tune T=16/32' lines of a layer judge log.
Usage: python pr32_tune.py <log> ..."""
import re, sys
for path in sys.argv[1:]:
    for line in open(path, errors="replace"):
        m = re.search(r"layer tune T=(16|32) (\w+): (.*)", line)
        if not m:
            continue
        items = re.findall(r"(default|\{[^}]*\}) ([\d.]+)us", m.group(3))
        d = {k: float(v) for k, v in items}
        dflt = d.get("default")
        best = min(d.items(), key=lambda kv: kv[1])
        pr = {k: v for k, v in d.items() if "pr32" in k}
        print(f"T={m.group(1)} {m.group(2)}: default {dflt} | best {best[0]} {best[1]} | " +
              " | ".join(f"{k} {v} ({v - dflt:+.1f})" for k, v in pr.items()))
