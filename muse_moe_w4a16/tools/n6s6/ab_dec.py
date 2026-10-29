# n6s6 (from n6s3 ab2.py, decode shapes): paired H100 A/B of two submission files (same process, one prepare each), cold L2 graph replay medians, us.
# usage (muse_moe_w4a16/): python tools/n6s6/ab_dec.py A.py B.py [rounds]   shapes: board bs128/bs256 (5 seeds) + layer v16/v32
import sys, fcntl, importlib.util, statistics as S, torch
sys.path.insert(0, '.'); sys.path.insert(0, 'tools')
import reference
def load(p, n):
    sp = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
W = reference.generate_weights(reference.WEIGHTS_SEED)
mods = [load(sys.argv[1], "subA"), load(sys.argv[2], "subB")]
sts = [m.prepare(W) for m in mods]
rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 2
R = torch.load('tools/data/routes.pt')
st0 = sts[0]
def mk(loc, seed):
    T = loc.shape[0]; g = torch.Generator(device='cuda').manual_seed(seed)
    ids = torch.where(loc.cuda() >= 0, loc.cuda().long() + st0["start"], (st0["start"] + st0["count"] + 5) % 256).int()
    tw = torch.rand(T, 8, device='cuda', generator=g); tw = tw / tw.sum(-1, keepdim=True)
    return {"hidden_states": torch.randn(T, 4096, device='cuda', generator=g).bfloat16(), "topk_ids": ids, "topk_weights": tw,
            "config": {"bs": T, "d_hidden": 4096, "local_start": st0["start"], "local_count": st0["count"]}}
cases = [(f"board{bs}", reference.generate_input(bs=bs, sharding="ep8", seed=3000 + 7 * s + bs)) for bs in (1, 8, 16, 32) for s in range(5)]
cases += [(l.replace("verify ", "v"), mk(loc, sd)) for (l, sd), loc in sorted(R.items()) if loc.shape[0] <= 32]
flush = torch.empty(512 << 20, dtype=torch.uint8, device='cuda'); sink = torch.empty((), dtype=torch.int64, device='cuda')
res = {}
maxd = 0.0
for name, d in cases:
    gs = []
    for m, st in zip(mods, sts):
        m.kernel_fn(d, st); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): o = m.kernel_fn(d, st)
        g.replay(); torch.cuda.synchronize(); gs.append((g, o))
    maxd = max(maxd, (gs[0][1].float() - gs[1][1].float()).abs().max().item())
    ts = [[], []]
    with open('/tmp/okc-gpu.lock', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for _ in range(rounds * 10):
            for i in (0, 1):
                torch.sum(flush.view(torch.int64), 0, out=sink)
                a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                a.record(); gs[i][0].replay(); b.record(); torch.cuda.synchronize(); ts[i].append(a.elapsed_time(b) * 1e3)
    r = res.setdefault(name, [[], []])
    for i in (0, 1): r[i].append(S.median(ts[i]))
    del gs
for name, (a, b) in res.items():
    print(f"{name}: A {S.mean(a):.1f} B {S.mean(b):.1f} d {S.mean(b) - S.mean(a):+.1f} us ({len(a)} seeds)")
print(f"max |A-B| {maxd:.3g}")
