"""Median kernel timeline over N cold replays. usage: tlmed.py <sub.py> "<load> <layer>" [N]"""
import sys, json, fcntl, torch, os, statistics as st_
sys.path.insert(0, ".")
import bench
from torch.profiler import profile, ProfilerActivity
sub = bench.load_sub(sys.argv[1]); flt = sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 25
w, st = bench.weights_state(sub)
exec(os.environ.get("STMT", ""), vars(sub))
for load, c in bench.SCORED:
    if flt != f"{load} {c['layer']}":
        continue
    data = bench.make_input(c, w, int(os.environ.get("SEED", "99")))
    g = bench.capture(lambda: sub.kernel_fn(data, st))
    flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda"); sink = torch.zeros((), dtype=torch.int64, device="cuda")
    runs = []
    with open("/tmp/okc-gpu.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            for _ in range(N):
                torch.sum(flush.view(torch.int64), 0, out=sink); torch.cuda.synchronize()
                g.replay(); torch.cuda.synchronize()
    p.export_chrome_trace("/tmp/tlmed.json")
    ev = [e for e in json.load(open("/tmp/tlmed.json"))["traceEvents"] if e.get("cat") == "kernel"]
    ev.sort(key=lambda e: e["ts"])
    ev = [e for e in ev if "at::native" not in e["name"]]
    k = len(ev) // N
    per = [ev[i*k:(i+1)*k] for i in range(N)]
    names = [e["name"] for e in per[0]]
    for j in range(k):
        s = st_.median(r[j]["ts"] - r[0]["ts"] for r in per); en = st_.median(r[j]["ts"] + r[j]["dur"] - r[0]["ts"] for r in per)
        print(f"{s:8.1f} {en:8.1f} {en-s:7.1f} {names[j][:90]}")
    print("total med", st_.median(r[-1]["ts"] + r[-1]["dur"] - r[0]["ts"] for r in per))
