"""proj n13s1: same-process A/B of pr1 base vs PR1_FLAT (env read at launch = at capture) at decode 1 sl/full.
Prints medians over R cold-L2 profiled replays (alternating A/B) of pr1 start/end and the preceding boundary end (us from graph start).
python tl13.py submission.py [R]   -- run in the layer problem dir; submission = orc.py's sub_rt.py"""
import sys, json, os, statistics, fcntl, torch
sys.path.insert(0, ".")
import bench
from torch.profiler import profile, ProfilerActivity
sub = bench.load_sub(sys.argv[1]); R = int(sys.argv[2]) if len(sys.argv) > 2 else 7
w, st = bench.weights_state(sub)
flush = torch.empty(1 << 30, dtype=torch.uint8, device="cuda"); sink = torch.zeros((), dtype=torch.int64, device="cuda")
for load, c in bench.SCORED:
    if load != "decode 1": continue
    data = bench.make_input(c, w, 99)
    gs = {}
    for v in ("base", "flat"):
        if v == "flat": os.environ["PR1_FLAT"] = "1"
        else: os.environ.pop("PR1_FLAT", None)
        gs[v] = bench.capture(lambda: sub.kernel_fn(data, st))
    os.environ.pop("PR1_FLAT", None)
    res = {v: [] for v in gs}
    lock = open("/tmp/okc-gpu.lock", "w"); fcntl.flock(lock, fcntl.LOCK_EX)
    for r in range(R + 1):
        for v, g in gs.items():
            torch.sum(flush.view(torch.int64), 0, out=sink); torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as p:
                g.replay(); torch.cuda.synchronize()
            p.export_chrome_trace("/tmp/tl13.json")
            ev = sorted((e for e in json.load(open("/tmp/tl13.json"))["traceEvents"] if e.get("cat") == "kernel"), key=lambda e: e["ts"])
            t0 = ev[0]["ts"]; k = next(i for i, e in enumerate(ev) if e["name"].startswith("void pr1::"))
            b = ev[k - 1]
            if r: res[v].append((b["ts"] + b["dur"] - t0, ev[k]["ts"] - t0, ev[k]["ts"] + ev[k]["dur"] - t0))
    fcntl.flock(lock, fcntl.LOCK_UN); lock.close()
    md = {v: [statistics.median(x[i] for x in res[v]) for i in range(3)] for v in res}
    for v, (be, ps, pe) in md.items():
        print(f"RES {load} {c['layer']:7s} {v:4s} bnd_end {be:6.1f} pr1 {ps:6.1f}..{pe:6.1f} post-bnd {pe - be:5.2f} (R {R})")
    print(f"RES {load} {c['layer']:7s} flat-base pr1 end {md['flat'][2] - md['base'][2]:+.2f} us")
