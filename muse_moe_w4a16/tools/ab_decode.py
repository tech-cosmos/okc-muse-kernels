import sys, fcntl, torch
sys.path.insert(0, '.')
import bench, reference, submission
w, st = bench.state()
flush = torch.empty(512 << 20, dtype=torch.uint8, device='cuda'); sink = torch.empty((), dtype=torch.int64, device='cuda')
def tg(fn, reps=40):
    g = torch.cuda.CUDAGraph(); fn(); torch.cuda.synchronize()
    with torch.cuda.graph(g): fn()
    ts = []
    with open('/tmp/okc-gpu.lock', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for _ in range(reps):
            torch.sum(flush.view(torch.int64), 0, out=sink); a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1e3)
    ts.sort(); return ts[len(ts) // 2]
VS = [int(v) for v in __import__("os").environ.get("VS", "1,2,1,2").split(",")]
for bs in [int(a) for a in sys.argv[1:]]:
    r = {v: [] for v in VS}
    for sd in (7, 8, 9, 10, 11):
        data = reference.generate_input(bs=bs, sharding='ep8', seed=sd)
        for v in VS:
            submission.TK2F = v; r[v].append(tg(lambda: submission.kernel_fn(data, st)))
    print(f"bs {bs}: " + "  ".join(f"v{v} {sum(r[v])/len(r[v]):.1f}" for v in sorted(r)), flush=True)
