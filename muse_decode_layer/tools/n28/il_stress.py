"""n28: graph-replay vs eager BITWISE stress of the T=128/256 expert path (HOT_SL/HOT_IL slices) on dominant-expert routers.
Per (router seed, verify load/layer): capture once, replay n fresh inputs (cold L2), compare each to an eager run on the
same input bitwise, and run reference.check_implementation. Also counts inputs whose max expert > 128 tokens.
python tools/n28/il_stress.py [submission.py] [n] [seeds] [filter]"""
import sys, torch
sys.path.insert(0, ".")
import bench, reference

path = sys.argv[1] if len(sys.argv) > 1 else "submission.py"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
seeds = [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else [5, 12, 17]
flt = sys.argv[4] if len(sys.argv) > 4 else "verify"
sub = bench.load_sub(path)
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
sink = torch.zeros((), dtype=torch.int64, device="cuda")


def copy_into(dst, src):
    for k, v in src.items():
        if isinstance(v, torch.Tensor):
            dst[k].copy_(v)


tot = dict(bad=0, fail=0, n=0)
for rs in seeds:
    w = reference.generate_weights(reference.WEIGHTS_SEED, router_seed=rs)
    st = sub.prepare(w)
    for load, c in bench.SCORED:
        name = f"{load} {c['layer']}"
        if flt not in name or c["batch"] == 1:
            continue
        data = bench.make_input(c, w, 9000)
        sub.kernel_fn(data, st)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            sub.kernel_fn(data, st)
            with torch.cuda.graph(g, stream=s):
                out = sub.kernel_fn(data, st)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        bad = fails = 0
        for i in range(n):
            d = bench.make_input(c, w, 9001 + 31 * i + 1000 * rs)
            copy_into(data, d)
            torch.sum(flush.view(torch.int64), 0, out=sink)
            g.replay()
            torch.cuda.synchronize()
            r = reference.collect(data, out)
            r = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in r.items()} if isinstance(r, dict) else \
                (r.clone() if isinstance(r, torch.Tensor) else r)
            ok, msg = reference.check_implementation(data, r, w)
            fails += not ok
            e = sub.kernel_fn(data, st)
            torch.cuda.synchronize()
            re = reference.collect(data, e)
            if isinstance(r, dict):
                same = all(torch.equal(r[k], re[k]) for k in r if isinstance(r[k], torch.Tensor))
            else:
                same = torch.equal(r, re) if isinstance(r, torch.Tensor) else all(torch.equal(a, b) for a, b in zip(r, re))
            bad += not same
            if not same or not ok:
                print(f"    rs {rs} {name} i {i} bitwise {same} ok {ok} {'' if ok else msg[:160]}", flush=True)
            del d, e, re, r
        print(f"rs {rs} {name:16s} replay!=eager {bad}/{n} check fails {fails}/{n}", flush=True)
        tot["bad"] += bad; tot["fail"] += fails; tot["n"] += n
        del g, data, out
        torch.cuda.empty_cache()
    del st, w
    torch.cuda.empty_cache()
print("TOTAL", tot)
