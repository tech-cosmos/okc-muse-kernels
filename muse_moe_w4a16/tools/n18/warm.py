import sys, os, fcntl, torch
sys.path.insert(0, '/tmp/moe18')
import reference, submission as S
w = reference.generate_weights(reference.WEIGHTS_SEED); st = S.prepare(w)
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
    return {"hidden_states": torch.randn(T, 4096, device='cuda').bfloat16(), "topk_ids": ids, "topk_weights": tw,
            "config": {"bs": T, "d_hidden": 4096, "local_start": st["start"], "local_count": st["count"]}}
for T, ne in [(8, 1), (16, 1), (16, 2), (32, 2), (32,4)]:
    data = mk(T, ne); r = []
    for name, v in [("sq", {}), ("h3", {T: 3})]:
        S.TK2H_M = v
        r.append(f"{name} {tg(lambda: S.kernel_fn(data, st)):.1f}")
    print(f"T={T} ne={ne} warm: " + "  ".join(r), flush=True)
