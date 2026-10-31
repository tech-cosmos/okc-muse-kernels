"""proj n20 variance audit: layer submission time vs the run's router draw (H100).
python route_var.py <layer_dir> <submission.py> <router_seeds a,b,..> <inputs per router> <filter> [tune 0|1]
Per router seed: prepare (tuner off unless tune=1), per scored shape matching filter capture one graph, then per fresh
input copy it in, flush L2, replay 9x (median). Prints one line per (router, shape, input) with routing features of the
input on the GPU's local range: distinct experts D, routes R, hot (65..128-token) experts nh, >128 nb, max tokens."""
import sys, os, fcntl, statistics, collections, torch
ldir = sys.argv[1]; sys.path.insert(0, ldir); os.chdir(ldir)
import bench, reference
sub = bench.load_sub(sys.argv[2])
seeds = [int(s) for s in sys.argv[3].split(",")]
nin = int(sys.argv[4]); flt = sys.argv[5]
if len(sys.argv) <= 6 or sys.argv[6] == "0":
    sub.TUNE = {}
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
sink = torch.zeros((), dtype=torch.int64, device="cuda")

def copy_into(dst, src):
    for k, v in src.items():
        if isinstance(v, torch.Tensor):
            dst[k].copy_(v)

def feats(d, w):
    fr = reference.ref_front(d, w)
    ids = fr["biased"].topk(8, dim=-1).indices.long()
    loc = (ids - fr["local_start"]) % reference.N_EXPERTS
    loc = loc[loc < reference.LOCAL_EXPERTS]
    cnt = sorted(collections.Counter(loc.tolist()).values(), reverse=True)
    return len(cnt), sum(cnt), sum(1 for c in cnt if 64 < c <= 128), sum(1 for c in cnt if c > 128), cnt[0] if cnt else 0, cnt

for rs in seeds:
    w = reference.generate_weights(reference.WEIGHTS_SEED, router_seed=rs)
    st = sub.prepare(w)
    for load, c in bench.SCORED:
        name = f"{load} {c['layer']}"
        if not any(f and f in name for f in flt.split(",")):
            continue
        data = bench.make_input(c, w, 7000)
        sub.kernel_fn(data, st); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            sub.kernel_fn(data, st)
            with torch.cuda.graph(g, stream=s):
                out = sub.kernel_fn(data, st)
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
        for i in range(nin):
            d = bench.make_input(c, w, 9001 + 31 * i + 997 * (rs % 1000))
            D, R, nh, nb, mx, cnt = feats(d, w)
            copy_into(data, d); torch.cuda.synchronize()
            ts = []
            with open("/tmp/okc-gpu.lock", "w") as lk:
                fcntl.flock(lk, fcntl.LOCK_EX)
                for _ in range(9):
                    torch.sum(flush.view(torch.int64), 0, out=sink)
                    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                    a.record(); g.replay(); b.record(); torch.cuda.synchronize()
                    ts.append(a.elapsed_time(b) * 1e3)
            print(f"ROW rs={rs} shape={name.replace(' ', '_')} in={i} us={statistics.median(ts):.1f} D={D} R={R} nh={nh} nb={nb} max={mx} cnt={','.join(map(str, cnt))}", flush=True)
            del d
        del g, data, out
        torch.cuda.empty_cache()
    del w, st
    torch.cuda.empty_cache()
