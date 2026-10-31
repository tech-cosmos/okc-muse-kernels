"""proj n13s1: ONE prepare, many pr1 option variants (state['tune'] overrides per captured graph; 'flat' = env PR1_FLAT at
capture, needs orc.py's sub_rt.py) at decode 1 sl/full. Alternating cold-L2 profiled replays; medians of pr1 end and graph end.
python tl13b.py submission.py R  -- run in the layer problem dir"""
import sys, json, os, statistics, fcntl, torch
sys.path.insert(0, ".")
import bench
from torch.profiler import profile, ProfilerActivity
sub = bench.load_sub(sys.argv[1]); R = int(sys.argv[2]) if len(sys.argv) > 2 else 7
VAR = [("base", {}), ("flat", None), ("pf2", {"pr1_pf": 2}), ("rb26", {"pr1_rb": 26}), ("rb29", {"pr1_rb": 29}),
       ("rb22", {"pr1_rb": 22}), ("pf2rb26", {"pr1_pf": 2, "pr1_rb": 26}), ("pf2rb22", {"pr1_pf": 2, "pr1_rb": 22}),
       ("npf1", {"pr1_npf": 1}), ("npf2", {"pr1_npf": 2})]
if os.environ.get("VARJ"): VAR = [tuple(v) for v in json.loads(os.environ["VARJ"])]
if os.environ.get("VARS"): VAR = [v for v in VAR if v[0] in os.environ["VARS"].split(",")]
w, st = bench.weights_state(sub)
flush = torch.empty(1 << 30, dtype=torch.uint8, device="cuda"); sink = torch.zeros((), dtype=torch.int64, device="cuda")
for load, c in bench.SCORED:
    if load != os.environ.get("LOAD", "decode 1"): continue
    key = (int(os.environ.get("TOK", "1")), c["layer"] == "sliding")
    data = bench.make_input(c, w, 99)
    orig = dict(st.get("tune", {}).get(key, {}))
    gs = {}
    for v, o in VAR:
        st.setdefault("tune", {})[key] = dict(orig, **(o or {}))
        if o is None: os.environ["PR1_FLAT"] = "1"; os.environ["PR8_FLAT"] = "1"
        gs[v] = bench.capture(lambda: sub.kernel_fn(data, st))
        os.environ.pop("PR1_FLAT", None); os.environ.pop("PR8_FLAT", None)
    st["tune"][key] = orig
    res = {v: [] for v in gs}
    lock = open("/tmp/okc-gpu.lock", "w"); fcntl.flock(lock, fcntl.LOCK_EX)
    for r in range(R + 1):
        for v, g in gs.items():
            torch.sum(flush.view(torch.int64), 0, out=sink); torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as p:
                g.replay(); torch.cuda.synchronize()
            p.export_chrome_trace("/tmp/tl13b.json")
            ev = sorted((e for e in json.load(open("/tmp/tl13b.json"))["traceEvents"] if e.get("cat") == "kernel"), key=lambda e: e["ts"])
            t0 = ev[0]["ts"]; k = next(i for i, e in enumerate(ev) if os.environ.get("KN", "pr1::") in e["name"])
            b = ev[k - 1]
            if r: res[v].append((b["ts"] + b["dur"] - t0, ev[k]["ts"] + ev[k]["dur"] - t0, max(e["ts"] + e["dur"] for e in ev) - t0))
    fcntl.flock(lock, fcntl.LOCK_UN); lock.close()
    md = {v: [statistics.median(x[i] for x in res[v]) for i in range(3)] for v in res}
    for v, (be, pe, ge) in md.items():
        print(f"RES {load} {c['layer']:7s} {v:8s} K post-bnd {pe - be:6.2f}  K end {pe - md['base'][1]:+6.2f}  graph end {ge:7.1f} ({ge - md['base'][2]:+6.2f})  R {R}")
