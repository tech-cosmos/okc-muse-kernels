# Experts on the LAYER's routing (from /tmp/moe10/routes.pt, made by /tmp/moe10/dump_routes.py in r-layer):
# VS=<cfg,...> python tools/layer_ab.py "decode 32" "decode 16" ...   (cold-L2 graph replay median, us)
# ENV=NAME sets submission.NAME = v for each v in VS (default TK2F).
import sys, os, fcntl, torch
sys.path.insert(0, '.')
import bench, importlib, os
submission = importlib.import_module(os.environ.get("SUB", "submission"))
bench.submission = submission
w, st = bench.state()
R = torch.load(__import__("os").path.join(__import__("os").path.dirname(__file__), "data", "routes.pt"))
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
def mk(loc, seed):
    T = loc.shape[0]; g = torch.Generator(device='cuda').manual_seed(seed)
    ids = loc.cuda().long() + st["start"]
    ids = torch.where(loc.cuda() >= 0, ids, (st["start"] + st["count"] + 5) % 256).int()
    tw = torch.rand(T, 8, device='cuda', generator=g); tw = tw / tw.sum(-1, keepdim=True)
    x = torch.randn(T, 4096, device='cuda', generator=g).bfloat16()
    return {"hidden_states": x, "topk_ids": ids, "topk_weights": tw,
            "config": {"bs": T, "d_hidden": 4096, "local_start": st["start"], "local_count": st["count"]}}
if __name__ == "__main__":
    name = os.environ.get("ENV", "TK2F")
    VS = [int(v) for v in os.environ.get("VS", "204,204").split(",")]
    for load in sys.argv[1:]:
        r = {v: [] for v in VS}
        for (l, sd), loc in sorted(R.items()):
            if l != load: continue
            data = mk(loc, sd)
            for v in VS:
                setattr(submission, name, v); r[v].append(tg(lambda: submission.kernel_fn(data, st)))
        print(f"{load}: " + "  ".join(f"{v}: {sum(r[v])/len(r[v]):.1f}" for v in r), flush=True)
