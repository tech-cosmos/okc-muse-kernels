"""proj n9s1: median kernel end times (us from graph start) over R profiled replays, per scored load (cold L2 each).
python tl9.py submission.py [R] [loads...]  -- run in the layer problem dir."""
import sys, json, os, statistics, fcntl, torch
sys.path.insert(0, ".")
import bench
from torch.profiler import profile, ProfilerActivity
sub = bench.load_sub(sys.argv[1]); R = int(sys.argv[2]) if len(sys.argv) > 2 else 7
want = sys.argv[3:] or ["decode 16", "decode 32", "verify 1"]
w, st = bench.weights_state(sub)
flush = torch.empty(1 << 30, dtype=torch.uint8, device="cuda"); sink = torch.zeros((), dtype=torch.int64, device="cuda")
for load, c in bench.SCORED:
    if load not in want: continue
    data = bench.make_input(c, w, 99)
    g = bench.capture(lambda: sub.kernel_fn(data, st))
    runs = []
    lock = open("/tmp/okc-gpu.lock", "w"); fcntl.flock(lock, fcntl.LOCK_EX)
    for r in range(R + 1):
        torch.sum(flush.view(torch.int64), 0, out=sink); torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            g.replay(); torch.cuda.synchronize()
        p.export_chrome_trace("/tmp/tl9.json")
        ev = sorted((e for e in json.load(open("/tmp/tl9.json"))["traceEvents"] if e.get("cat") == "kernel"), key=lambda e: e["ts"])
        t0 = ev[0]["ts"]
        if r: runs.append([(e["name"][:28], e["ts"] - t0, e["ts"] + e["dur"] - t0) for e in ev])
    fcntl.flock(lock, fcntl.LOCK_UN); lock.close()
    n = min(len(x) for x in runs)
    print("==", load, c["layer"], "kernels", n, "R", R)
    for k in range(n):
        print(f"  {runs[0][k][0]:28s} start {statistics.median(x[k][1] for x in runs):7.1f} end {statistics.median(x[k][2] for x in runs):7.1f}")
