import sys, json, fcntl, statistics, torch
sys.path.insert(0, ".")
import bench, reference, importlib.util
from torch.profiler import profile, ProfilerActivity
flt, n, f = sys.argv[1], int(sys.argv[2]), sys.argv[3]
w = reference.generate_weights(reference.WEIGHTS_SEED)
spec = importlib.util.spec_from_file_location("sub", f); sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub); st = sub.prepare(w)
c = [c for l, c in bench.SCORED if f"{l} {c['layer']}" == flt][0]
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 99
data = bench.make_input(c, w, seed)
g = bench.capture(lambda: sub.kernel_fn(data, st))
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda"); sink = torch.zeros((), dtype=torch.int64, device="cuda")
rows = {}
for r in range(n):
    with open("/tmp/okc-gpu.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        torch.sum(flush.view(torch.int64), 0, out=sink); torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            g.replay(); torch.cuda.synchronize()
    p.export_chrome_trace("/tmp/moe27/tl.json")
    ev = sorted([e for e in json.load(open("/tmp/moe27/tl.json"))["traceEvents"] if e.get("cat") == "kernel"], key=lambda e: e["ts"])
    t0 = ev[0]["ts"]
    for i, e in enumerate(ev):
        rows.setdefault(i, (e["name"][:60], []))[1].append((e["ts"] - t0, e["ts"] + e["dur"] - t0))
for i, (nm, v) in rows.items():
    print(f"{statistics.median(a for a,b in v):8.1f} {statistics.median(b for a,b in v):8.1f}  {nm}")
