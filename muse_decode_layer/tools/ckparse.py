"""Summarize an `okc check/submit --json` output: python tools/ckparse.py out.json"""
import json, sys
d = json.load(open(sys.argv[1]))
print(d.get("id"), d.get("status"))
def walk(o):
    if isinstance(o, dict):
        if o.get("event") == "benchmark":
            c = o["case"]
            print(f"{c['load']:10s} {c['layer']:8s} ours {o['median_ns']/1e3:8.1f}  sglang {o['baseline_ns']/1e3:8.1f}  {o['speedup']:.3f}")
        elif o.get("event") == "test" and o.get("status") != "pass":
            print("TEST", o.get("status"), o.get("message"))
        if "speedup" in o and "score_us" in o:
            print("speedup", round(o["speedup"], 4), o.get("speedup_ci"), "score_us", round(o["score_us"]))
        for v in o.values():
            walk(v)
    elif isinstance(o, list):
        for v in o:
            walk(v)
walk(d)
