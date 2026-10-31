"""proj n20: A/B two layer submissions on the same router seeds + inputs: bitwise outputs + interleaved graph timing.
python ab_route.py <layer_dir> <A.py> <B.py> <router_seeds> <inputs> <filter>   (tuner off in both)"""
import sys, os, fcntl, statistics, collections, torch
ldir = sys.argv[1]; sys.path.insert(0, ldir); os.chdir(ldir)
import bench, reference
A, B = bench.load_sub(sys.argv[2]), bench.load_sub(sys.argv[3])
A.TUNE = {}; B.TUNE = {}
seeds = [int(x) for x in sys.argv[4].split(",")]; nin = int(sys.argv[5]); flt = sys.argv[6].split(",")
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
    sa, sb = A.prepare(w), B.prepare(w)
    for load, c in bench.SCORED:
        name = f"{load} {c['layer']}"
        if not any(f and f in name for f in flt): continue
        da, db = bench.make_input(c, w, 7000), bench.make_input(c, w, 7000)
        ga, oa = cap(A, da, sa); gb, ob = cap(B, db, sb)
        for i in range(nin):
            d = bench.make_input(c, w, 9001 + 31 * i + 997 * (rs % 1000))
            fr = reference.ref_front(d, w); ids = fr["biased"].topk(8, dim=-1).indices.long()
            loc = (ids - fr["local_start"]) % 256; loc = loc[loc < 32]
            cnt = sorted(collections.Counter(loc.tolist()).values(), reverse=True)
            for k, v in d.items():
                if isinstance(v, torch.Tensor): da[k].copy_(v); db[k].copy_(v)
            torch.cuda.synchronize()
            ta, tb = [], []
            with open("/tmp/okc-gpu.lock", "w") as lk:
                fcntl.flock(lk, fcntl.LOCK_EX)
                for _ in range(8):
                    ta.append(rp(ga)); tb.append(rp(gb))
            same = all(torch.equal(x, y) for x, y in zip(oa, ob) if isinstance(x, torch.Tensor))
            ok, msg = reference.check_implementation(db, reference.collect(db, ob), w)
            ma, mb = statistics.median(ta), statistics.median(tb)
            tot[name].append(mb - ma)
            print(f"rs {rs} {name:17s} in {i} A {ma:6.1f} B {mb:6.1f} d {mb - ma:+6.1f} bitwise {same} check {ok} max {cnt[0]} D {len(cnt)}", flush=True)
        del ga, gb, oa, ob, da, db; torch.cuda.empty_cache()
    del sa, sb, w; torch.cuda.empty_cache()
for k, v in tot.items(): print(f"MEAN {k}: B-A {statistics.mean(v):+.2f} us over {len(v)}")
