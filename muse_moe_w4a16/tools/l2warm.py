# H200 proxy for decode consumer throughput: weights L2-resident on the H100 (1 local expert, warm L2, no flush),
# so DRAM does not bound and the time is the consumers' issue/latency rate.
# VS=<TK2F cfgs> python tools/l2warm.py [T ...]   (median graph replay, us)
import sys, os, fcntl, torch
sys.path.insert(0, '.')
import bench, importlib
submission = importlib.import_module(os.environ.get("SUB", "submission"))
bench.submission = submission
w, st = bench.state()
def tg(fn, reps=200):
    g = torch.cuda.CUDAGraph(); fn(); torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for _ in range(5): fn()
    ts = []
    with open('/tmp/okc-gpu.lock', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for _ in range(reps):
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1e3 / 5)
    ts.sort(); return ts[len(ts) // 2]
def mk(T, ne):
    ids = torch.full((T, 8), (st["start"] + st["count"] + 5) % 256, dtype=torch.int32, device='cuda')
    ids[:, 0] = st["start"] + torch.arange(T, device='cuda') % ne
    tw = torch.rand(T, 8, device='cuda'); tw = tw / tw.sum(-1, keepdim=True)
    x = torch.randn(T, 4096, device='cuda').bfloat16()
    return {"hidden_states": x, "topk_ids": ids, "topk_weights": tw,
            "config": {"bs": T, "d_hidden": 4096, "local_start": st["start"], "local_count": st["count"]}}
VS = [int(v) for v in os.environ.get("VS", "204").split(",")]
for T in [int(a) for a in sys.argv[1:]] or [8, 16, 32]:
    data = mk(T, 1)
    ref = None
    out = []
    for v in VS:
        submission.TK2F = submission.TK2F_DYN = v
        o = submission.kernel_fn(data, st).float()
        if ref is None: ref = o
        err = (o - ref).abs().max().item()
        out.append(f"{v}: {tg(lambda: submission.kernel_fn(data, st)):.1f}" + (f" (err {err:.2g})" if err else ""))
    print(f"T={T} 1 expert warm: " + "  ".join(out), flush=True)
