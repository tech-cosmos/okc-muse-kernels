"""n8s1: per-segment in-layer timing of verify-sliding attention: python tools/n8/attn_orc.py <rounds> <reps> files...
Per file and shape (verify 16/32 sliding): R profiled cold-L2 graph replays; reports medians of qkv end -> o-GEMM start
(attention segment), qkv end -> o end, and the whole replay (first kernel start -> last end). Rounds interleave files."""
import sys, json, fcntl, statistics, importlib.util, torch
sys.path.insert(0, ".")
import bench, reference
from torch.profiler import profile, ProfilerActivity
rounds, reps, files = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
SH = ["verify 16 sliding", "verify 32 sliding"]
w = reference.generate_weights(reference.WEIGHTS_SEED)
subs = []
for i, f in enumerate(files):
    spec = importlib.util.spec_from_file_location(f"sub{i}", f)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    subs.append((m, m.prepare(w)))
cfgs = {f"{l} {c['layer']}": c for l, c in bench.SCORED}
data = {n: bench.make_input(cfgs[n], w, 99) for n in SH}
graphs = {(i, n): bench.capture(lambda m=m, st=st, n=n: m.kernel_fn(data[n], st)) for i, (m, st) in enumerate(subs) for n in SH}
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda"); sink = torch.zeros((), dtype=torch.int64, device="cuda")
res = {k: [] for k in graphs}
def one(g):
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(reps):
            torch.sum(flush.view(torch.int64), 0, out=sink); g.replay()
        torch.cuda.synchronize()
    p.export_chrome_trace("/tmp/n8tl.json")
    ev = sorted([e for e in json.load(open("/tmp/n8tl.json"))["traceEvents"] if e.get("cat") == "kernel"], key=lambda e: e["ts"])
    out, cur = [], []
    for e in ev:  # split replays at the flush reduction
        if "reduce" in e["name"].lower() or "sum" in e["name"].lower():
            if cur: out.append(cur)
            cur = []
        else:
            cur.append(e)
    if cur: out.append(cur)
    r = []
    for k in out:
        b = next(i for i, e in enumerate(k) if e["name"] == "_boundary")
        q0 = next(e for e in k if "nvjet" in e["name"] or "g30" in e["name"] or "gemv" in e["name"]); qe = q0["ts"] + q0["dur"]; o = k[b - 1]
        r.append((o["ts"] - qe, o["ts"] + o["dur"] - qe, max(e["ts"] + e["dur"] for e in k) - k[0]["ts"]))
    return r
for _ in range(rounds):
    for k, g in graphs.items():
        with open("/tmp/okc-gpu.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            res[k] += one(g)
for n in SH:
    print(n)
    for i, f in enumerate(files):
        v = res[(i, n)]
        print(f"  {f:28s} attn {statistics.median(x[0] for x in v):6.2f}  attn+o {statistics.median(x[1] for x in v):6.2f}  "
              f"replay {statistics.median(x[2] for x in v):7.2f}  (n={len(v)})")
