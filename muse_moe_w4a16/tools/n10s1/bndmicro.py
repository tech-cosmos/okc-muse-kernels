"""Per-launch time of the captured second boundary (chain of 20 PDL launches in a graph), orig vs HOP variants."""
import sys, os, torch, fcntl, statistics
sys.path.insert(0, ".")
import bench
sub = bench.load_sub(sys.argv[1])
A = torch.load("/tmp/moeend/bnd_args.pt", weights_only=False)
def tg(fn, n=20, reps=50):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn(); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            for _ in range(n):
                fn()
    ts = []
    with open("/tmp/okc-gpu.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for _ in range(reps):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) * 1000 / n)
    return statistics.median(ts)
for nm, (g, a, k) in A.items():
    res = []
    for extra in ({"TRIG": True}, {"TRIG": True, "HOP": True}, {"TRIG": True, "HOP2": True}):
        kk = dict(k, **extra)
        res.append(tg(lambda: sub._boundary[g](*a, **kk)))
    print(f"{nm:18s} grid {g} orig {res[0]:.2f} hop {res[1]:.2f} hop2 {res[2]:.2f} us/launch")
