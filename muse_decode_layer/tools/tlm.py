"""python tools/tlm.py "<load> <layer>" <kernel name substring> <replays> <file>: N cold profiled replays; median start of the first matching kernel and end of the last kernel (us from the first kernel). ONE file per process (several modules in one process share .so statics and crash)."""
import sys, json, fcntl, statistics, torch
sys.path.insert(0, ".")
import bench
from torch.profiler import profile, ProfilerActivity
flt, key, n = sys.argv[1], sys.argv[2], int(sys.argv[3]); files = sys.argv[4:]
res = {}
mods = []
import importlib.util, reference
w = reference.generate_weights(reference.WEIGHTS_SEED)
for i, f in enumerate(files):
    spec = importlib.util.spec_from_file_location(f"sub{i}", f)
    sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub); st = sub.prepare(w)
    c = [c for l, c in bench.SCORED if f"{l} {c['layer']}" == flt][0]
    data = bench.make_input(c, w, 99)
    g = bench.capture(lambda sub=sub, data=data, st=st: sub.kernel_fn(data, st))
    mods.append((f, g))
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda"); sink = torch.zeros((), dtype=torch.int64, device="cuda")
for r in range(n):
    for f, g in mods:
        with open("/tmp/okc-gpu.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            torch.sum(flush.view(torch.int64), 0, out=sink); torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as p:
                g.replay(); torch.cuda.synchronize()
        p.export_chrome_trace("/tmp/l39/tl.json")
        ev = sorted([e for e in json.load(open("/tmp/l39/tl.json"))["traceEvents"] if e.get("cat") == "kernel"], key=lambda e: e["ts"])
        t0 = ev[0]["ts"]
        ks = [e["ts"] - t0 for e in ev if key in e["name"]]
        end = max(e["ts"] + e["dur"] for e in ev) - t0
        res.setdefault(f, []).append((ks[0] if ks else -1, end))
for f in files:
    a = res[f]
    print(f"{f:28s} {key} start med {statistics.median(x[0] for x in a):7.1f}  end med {statistics.median(x[1] for x in a):7.1f}  end-start med {statistics.median(x[1]-x[0] for x in a):7.1f}")
