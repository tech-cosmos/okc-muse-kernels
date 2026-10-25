"""Kernel timeline (start/end us relative to the first kernel) of one graph replay of a scored shape."""
import sys, json, fcntl, torch
sys.path.insert(0, ".")
import bench
from torch.profiler import profile, ProfilerActivity
sub = bench.load_sub(sys.argv[1]); flt = sys.argv[2]
w, st = bench.weights_state(sub)
for load, c in bench.SCORED:
    if flt != f"{load} {c['layer']}":
        continue
    data = bench.make_input(c, w, 99)
    g = bench.capture(lambda: sub.kernel_fn(data, st))
    flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda"); sink = torch.zeros((), dtype=torch.int64, device="cuda")
    with open("/tmp/okc-gpu.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        torch.sum(flush.view(torch.int64), 0, out=sink); torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            g.replay(); torch.cuda.synchronize()
    p.export_chrome_trace("/tmp/tl.json")
    ev = [e for e in json.load(open("/tmp/tl.json"))["traceEvents"] if e.get("cat") == "kernel"]
    ev.sort(key=lambda e: e["ts"])
    t0 = ev[0]["ts"]
    for e in ev:
        print(f"{e['ts']-t0:8.1f} {e['ts']+e['dur']-t0:8.1f} {e['dur']:7.1f} s{e['args'].get('stream')} {e['name'][:60]}")
