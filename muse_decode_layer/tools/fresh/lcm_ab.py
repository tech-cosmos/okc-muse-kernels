"""f1s15: LCM (nvs last-CTA merge) vs _combine_hb at verify 16 sliding (+ any shape given): correctness vs reference,
output diff, per-kernel profile and whole-graph cold-L2 timing, interleaved. python tools/fresh/lcm_ab.py <sub> [rounds] [shape]"""
import sys, json, fcntl, statistics, torch
sys.path.insert(0, ".")
import bench, reference
from torch.profiler import profile, ProfilerActivity
sub = bench.load_sub(sys.argv[1]); sub.TUNE.clear()
rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
shape = sys.argv[3] if len(sys.argv) > 3 else "verify 16 sliding"
w, st = bench.weights_state(sub)
c = {f"{l} {c['layer']}": c for l, c in bench.SCORED}[shape]
base_tune = dict(st["tune"])
def setv(v):
    for k in list(st["tune"]): st["tune"][k] = {**base_tune.get(k, {}), "nvslcm": v}
    if not st["tune"]: st["tune"].update({(t, s): {"nvslcm": v} for t in (1, 8, 16, 32, 128, 256) for s in (True, False)})
outs = {}
for seed in (99, 7):
    data = bench.make_input(c, w, seed)
    for v in (True, False):
        setv(v); o = sub.kernel_fn(data, st); torch.cuda.synchronize()
        ok, msg = reference.check_implementation(data, reference.collect(data, o), w)
        outs[v] = [x.float().clone() for x in (o if isinstance(o, (tuple, list)) else [o])]
        print(seed, "lcm" if v else "cb ", "PASS" if ok else "FAIL", msg[:150])
    print("  max|diff| lcm vs cb:", [float((a - b).abs().max()) for a, b in zip(outs[True], outs[False])])
data = bench.make_input(c, w, 99)
gs = {}
for v in (True, False):
    setv(v); gs[v] = bench.capture(lambda: sub.kernel_fn(data, st))
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda"); sink = torch.zeros((), dtype=torch.int64, device="cuda")
res = {True: [], False: []}; kt = {True: {}, False: {}}
for _ in range(rounds):
    for v in (True, False, False, True):
        with open("/tmp/okc-gpu.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for _ in range(10):
                torch.sum(flush.view(torch.int64), 0, out=sink)
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record(); gs[v].replay(); b.record(); torch.cuda.synchronize()
                res[v].append(a.elapsed_time(b) * 1000)
            with profile(activities=[ProfilerActivity.CUDA]) as p:
                for _ in range(3):
                    torch.sum(flush.view(torch.int64), 0, out=sink); gs[v].replay()
                torch.cuda.synchronize()
        for e in p.key_averages():
            if any(s in e.key for s in ("nvs", "combine", "nvjet", "g30", "lt_", "Gemm", "gemm")):
                kt[v].setdefault(e.key[:60], []).append(e.device_time_total / max(1, e.count) if hasattr(e, "device_time_total") else e.cuda_time_total / max(1, e.count))
for v in (True, False):
    print("lcm" if v else "cb ", f"graph median {statistics.median(res[v]):.1f} us (n={len(res[v])})")
    for k, x in sorted(kt[v].items()):
        print(f"    {k:60s} {statistics.median(x):7.2f}")
