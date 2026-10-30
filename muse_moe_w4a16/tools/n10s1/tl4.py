"""Tail timeline: python tl4.py <submission.py> [reps]. For decode 1/16/32 + verify 1 sliding: per-kernel start/end
(median over reps of single cold-L2 replays) relative to moe_tk2f's END, kernels after tk2f's start only."""
import sys, json, fcntl, os, statistics, torch
sys.path.insert(0, ".")
import bench
from torch.profiler import profile, ProfilerActivity
sub = bench.load_sub(sys.argv[1]); reps = int(sys.argv[2]) if len(sys.argv) > 2 else 7
exec(os.environ.get("STMT", ""), vars(sub))
w, st = bench.weights_state(sub)
LOADS = os.environ.get("LOADS", "decode 1 sliding,decode 16 sliding,decode 32 sliding,verify 1 sliding").split(",")
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda"); sink = torch.zeros((), dtype=torch.int64, device="cuda")
for load, c in bench.SCORED:
    nm = f"{load} {c['layer']}"
    if nm not in LOADS:
        continue
    data = bench.make_input(c, w, 99)
    g = bench.capture(lambda: sub.kernel_fn(data, st))
    rows, tot = [], []
    for r in range(reps):
        with open("/tmp/okc-gpu.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            torch.sum(flush.view(torch.int64), 0, out=sink); torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as p:
                g.replay(); torch.cuda.synchronize()
        fn = f"/tmp/moeend/tl_{os.getpid()}.json"
        p.export_chrome_trace(fn)
        ev = sorted([e for e in json.load(open(fn))["traceEvents"] if e.get("cat") == "kernel"], key=lambda e: e["ts"])
        t0 = ev[0]["ts"]
        k = max(i for i, e in enumerate(ev) if "moe_tk2f" in e["name"])
        te = ev[k]["ts"] + ev[k]["dur"]
        rows.append([(e["name"][:40], e["ts"] - te, e["ts"] + e["dur"] - te) for e in ev[k - 1:]])
        tot.append(ev[-1]["ts"] + ev[-1]["dur"] - t0)
    print(f"== {nm}: total med {statistics.median(tot):.1f}  tk2f end at {statistics.median([0]):.1f}")
    for j in range(len(rows[0])):
        print(f"  {statistics.median(r[j][1] for r in rows):7.1f} {statistics.median(r[j][2] for r in rows):7.1f}  {rows[0][j][0]}")
    del g
