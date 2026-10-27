"""Per-shape H200 summary of okc --json outputs: python tools/jsum.py A.json [B.json] (B - A deltas)."""
import json, sys
def load(p):
    d = json.load(open(p)); r = d.get("result") or {}
    out = {}
    for x in r.get("records", []):
        if "median_ns" in x and "case" in x:
            c = x["case"]; out[(c["load"], c["layer"])] = x["median_ns"] / 1000
    return d.get("id"), r.get("speedup"), out
runs = [load(p) for p in sys.argv[1:]]
for i, s, _ in runs: print(i, s)
for k in runs[0][2]:
    v = [r[2].get(k, float("nan")) for r in runs]
    print(f"{k[0]:10s} {k[1]:8s} " + " ".join(f"{x:8.2f}" for x in v) + (f"  d {v[1]-v[0]:+.2f}" if len(v) > 1 else ""))
