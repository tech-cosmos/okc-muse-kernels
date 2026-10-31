"""proj n22: N layer submissions on the same router seeds + inputs, interleaved graph replays (cold L2), tuner off.
python ab_multi.py <layer_dir> <seeds> <inputs> <filter> <base.py> <v1.py> ...   prints per-variant delta vs base"""
import sys, os, fcntl, statistics, collections, torch
ldir = sys.argv[1]; sys.path.insert(0, ldir); os.chdir(ldir)
import bench, reference
seeds = [int(x) for x in sys.argv[2].split(",")]; nin = int(sys.argv[3]); flt = sys.argv[4].split(",")
paths = sys.argv[5:]; subs = [bench.load_sub(p) for p in paths]
for s in subs: s.TUNE = {}
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda"); sink = torch.zeros((), dtype=torch.int64, device="cuda")
def cap(sub, data, st):
    sub.kernel_fn(data, st); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        sub.kernel_fn(data, st)
        with torch.cuda.graph(g, stream=s):
            out = sub.kernel_fn(data, st)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    return g, out
def rp(g):
    torch.sum(flush.view(torch.int64), 0, out=sink)
    a, b = torch.cuda.Event(True), torch.cuda.Event(True); a.record(); g.replay(); b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) * 1e3
tot = collections.defaultdict(list)
for rs in seeds:
    w = reference.generate_weights(reference.WEIGHTS_SEED, router_seed=rs)
    sts = [s.prepare(w) for s in subs]
    for load, c in bench.SCORED:
        name = f"{load} {c['layer']}"
        if not any(f and f in name for f in flt): continue
        ds = [bench.make_input(c, w, 7000) for _ in subs]
        gs = [cap(s, d, st) for s, d, st in zip(subs, ds, sts)]
        for i in range(nin):
            d = bench.make_input(c, w, 9001 + 31 * i + 997 * (rs % 1000))
            fr = reference.ref_front(d, w); ids = fr["biased"].topk(8, dim=-1).indices.long()
            loc = (ids - fr["local_start"]) % 256; loc = loc[loc < 32]
            cnt = sorted(collections.Counter(loc.tolist()).values(), reverse=True)
            for dd in ds:
                for k, v in d.items():
                    if isinstance(v, torch.Tensor): dd[k].copy_(v)
            torch.cuda.synchronize()
            ts = [[] for _ in subs]
            with open("/tmp/okc-gpu.lock", "w") as lk:
                fcntl.flock(lk, fcntl.LOCK_EX)
                for _ in range(8):
                    for j, (g, o) in enumerate(gs): ts[j].append(rp(g))
            same = all(all(torch.equal(x, y) for x, y in zip(gs[0][1], g[1]) if isinstance(x, torch.Tensor)) for g in gs[1:])
            m = [statistics.median(t) for t in ts]
            for j in range(1, len(subs)): tot[(name, j)].append(m[j] - m[0])
            print(f"rs {rs} {name:17s} in {i} base {m[0]:6.1f} " + " ".join(f"{x - m[0]:+5.1f}" for x in m[1:]) +
                  f" bitwise {same} max {cnt[0]} 2nd {cnt[1]} n>64 {sum(c > 64 for c in cnt)} D {len(cnt)}", flush=True)
        del gs, ds; torch.cuda.empty_cache()
    del sts, w; torch.cuda.empty_cache()
for (k, j), v in sorted(tot.items()): print(f"MEAN {k} {os.path.basename(paths[j])}: {statistics.mean(v):+.2f} us over {len(v)} ({sum(x < 0 for x in v)} faster)")
