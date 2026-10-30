"""Lead 03:50Z LANE B: held-out robustness of main (layer n11s1).
python tools/n11/heldout.py [file] [routers] [nseeds] [filter]
  routers: comma list of router seeds (0 = WEIGHTS_SEED's own router); the judge picks one router per run.
  Per scored shape, inputs = the scored seed (801..812) + nseeds-1 others (seed + 1000 j). Per input, the variants
  below are captured as graphs and replayed interleaved (cold L2), median per variant minus the empty graph.
  prepare() runs the tuner ONCE (first router); later routers re-prepare with TUNE = {} and reuse the picks.
Variants: main; notune (pick -> base, only if the tuner picked something); at T=128 hot0 / hotxn0 (HOT_XN off);
at T=256 hot32 (old fixed moe_hot grid = hotdyn off) / hotxn0."""
import fcntl, os, sys, statistics, time, torch
sys.path.insert(0, ".")
import bench, reference as R

f = sys.argv[1] if len(sys.argv) > 1 else "submission.py"
routers = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0").split(",")]
nseeds = int(sys.argv[3]) if len(sys.argv) > 3 else 4
flt = sys.argv[4] if len(sys.argv) > 4 else ""
REPS = int(os.environ.get("REPS", "15"))
sub = bench.load_sub(f)
T0 = time.time()
W0 = R.generate_weights(R.WEIGHTS_SEED)
TOK = {"decode 1": 1, "decode 16": 16, "decode 32": 32, "verify 1": 8, "verify 16": 128, "verify 32": 256}
_FL = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
_SK = torch.zeros((), dtype=torch.int64, device="cuda")


def variants(tokens, sliding, picks):
    key = (tokens, sliding)
    v = [("main", None)]
    if picks.get(key):
        v.append(("notune", {}))
    p = dict(picks.get(key, {}))
    if tokens == 128:
        v += [("hot0", dict(p, hot=0)), ("hotxn0", dict(p, hot=sub.HOT128 + 100000000))]
    if tokens == 256:
        v += [("hot32", dict(p, hot=32)), ("hotxn0", dict(p, hot=sub.HOT256 + 100000000))]
    return v


def timed(graphs):
    ts = [[] for _ in graphs]
    for r in range(REPS + 2):
        for i in (range(len(graphs)) if r % 2 == 0 else reversed(range(len(graphs)))):
            torch.sum(_FL.view(torch.int64), 0, out=_SK)
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); graphs[i].replay(); b.record(); torch.cuda.synchronize()
            if r >= 2:
                ts[i].append(a.elapsed_time(b) * 1000)
    return [statistics.median(t) for t in ts]


empty = None
picks = None
for ri, rs in enumerate(routers):
    W = W0 if rs == 0 else dict(W0, **R._router(torch.Generator(device="cuda").manual_seed(rs), "cuda", W0["pre_ffn_norm"]))
    if picks is None:
        st = sub.prepare(W)
        picks = {k: dict(v) for k, v in st.get("tune", {}).items()}
        print("TUNEPICKS", {f"{k[0]}{'s' if k[1] else 'f'}": v for k, v in picks.items() if v}, flush=True)
    else:
        saved, sub.TUNE = sub.TUNE, {}
        st = sub.prepare(W)
        sub.TUNE = saved
        st["tune"] = {k: dict(v) for k, v in picks.items()}
    if empty is None:
        empty = bench.time_graph(lambda: None, reps=60)
    print(f"# router {rs} prepared at {time.time() - T0:.0f}s", flush=True)
    for load, c in bench.SCORED:
        name = f"{load} {c['layer']}"
        if flt not in name:
            continue
        S = 801 + bench.SCORED.index((load, c))
        tokens, sliding = TOK[load], c["layer"] == "sliding"
        key = (tokens, sliding)
        for j in range(nseeds):
            seed = S + 1000 * j
            d = bench.make_input(c, W, seed)
            ids = R.ref_front(d, W)["biased"].topk(8, dim=-1).indices
            ls = int(d["local_start"]); loc = ids[(ids - ls) % R.N_EXPERTS < R.LOCAL_EXPERTS]
            cnt = sorted(torch.bincount((loc - ls) % R.N_EXPERTS, minlength=32).tolist(), reverse=True)
            nh = sum(65 <= x <= 128 for x in cnt); big = sum(x > 128 for x in cnt)
            vs = variants(tokens, sliding, picks)
            graphs = []
            for vn, opts in vs:
                st["tune"][key] = picks.get(key, {}) if opts is None else opts
                graphs.append(bench.capture(lambda: sub.kernel_fn(d, st)))
            st["tune"][key] = picks.get(key, {})
            with open("/tmp/okc-gpu.lock", "w") as lk:
                fcntl.flock(lk, fcntl.LOCK_EX)
                ms = timed(graphs)
            print(f"R {rs} | {name:15s} | seed {seed:5d} | routes {sum(cnt):4d} dist {sum(x > 0 for x in cnt):2d} nh {nh}+{big} top {cnt[:3]} | "
                  + " ".join(f"{vn} {m - empty:.1f}" for (vn, _), m in zip(vs, ms)), flush=True)
            del graphs, d
            getattr(R.generate_input, "cache_clear", lambda: None)()
            torch.cuda.empty_cache()
    del st
    if W is not W0:
        del W
    torch.cuda.empty_cache()
print(f"# done {time.time() - T0:.0f}s", flush=True)
