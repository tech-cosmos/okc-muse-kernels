# n22: DQW 0 vs 1 consumer-rate proxy: ne local experts (L2-resident when ne <= 1), T tokens spread over them, default
# config, graph of 5 back-to-back calls, no flush. python tools/n22/dqw_warm.py ne T [T ...]
import sys, fcntl, torch
sys.path.insert(0, '.')
import bench, submission as S
bench.submission = S
w, st = bench.state()
st["w13sq"] = S._sq_pack(st["w13q"], st["w13s2"]); st["w2sq"] = S._sq_pack(st["w2q"], st["w2s2"])
libs = {0: st["lib"]}
S.DQW = 1; libs[1] = S._lib(); S.DQW = 0; S._lib()
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
ne = int(sys.argv[1])
for T in [int(a) for a in sys.argv[2:]]:
    ids = torch.full((T, 8), (st["start"] + st["count"] + 5) % 256, dtype=torch.int32, device='cuda')
    ids[:, 0] = st["start"] + torch.arange(T, device='cuda') % ne
    tw = torch.rand(T, 8, device='cuda'); tw = tw / tw.sum(-1, keepdim=True)
    d = {"hidden_states": torch.randn(T, 4096, device='cuda').bfloat16(), "topk_ids": ids, "topk_weights": tw,
         "config": {"bs": T, "d_hidden": 4096, "local_start": st["start"], "local_count": st["count"]}}
    r = {0: [], 1: []}
    for _ in range(3):
        for q in (0, 1):
            st["lib"] = libs[q]; r[q].append(tg(lambda: S.kernel_fn(d, st)))
    st["lib"] = libs[0]
    m = {q: sorted(v)[1] for q, v in r.items()}
    print(f"ne {ne} T {T}: q0 {m[0]:.2f} q1 {m[1]:.2f} d {m[1] - m[0]:+.2f} us", flush=True)
