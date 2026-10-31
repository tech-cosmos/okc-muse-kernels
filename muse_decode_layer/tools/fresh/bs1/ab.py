"""fresh x11: interleaved cold-L2 graph A/B of standalone moe variants (dirs with submission.py).
python tools/fresh/bs1/ab.py <bs> <seeds> <rounds> dirA dirB ...  (run from muse_moe_w4a16/: bench/reference from cwd)
Prints per-seed medians (us, graph minus empty graph) and the mean per variant; checks outputs equal to A's."""
import sys, fcntl, statistics, importlib.util, torch
sys.path.insert(0, ".")
import bench, reference
bs, seeds, rounds = int(sys.argv[1]), [int(x) for x in sys.argv[2].split(",")], int(sys.argv[3])
dirs = sys.argv[4:]
mods = []
for i, d in enumerate(dirs):
    spec = importlib.util.spec_from_file_location(f"sub{i}", d + "/submission.py"); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); mods.append(m)
W = reference.generate_weights(reference.WEIGHTS_SEED)
sts = [m.prepare(W) for m in mods]
emp = bench.empty_cost()
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
res = {d: [] for d in dirs}
for sd in seeds:
    data = reference.generate_input(bs=bs, sharding="ep8", seed=sd)
    gs, outs = [], []
    for m, st in zip(mods, sts):
        o = m.kernel_fn(data, st); torch.cuda.synchronize(); outs.append(o.clone() if torch.is_tensor(o) else [x.clone() for x in o])
        g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            m.kernel_fn(data, st)
            with torch.cuda.graph(g, stream=s):
                m.kernel_fn(data, st)
        torch.cuda.current_stream().wait_stream(s); g.replay(); torch.cuda.synchronize(); gs.append(g)
    eq = ["=" if (torch.equal(o, outs[0]) if torch.is_tensor(o) else all(torch.equal(a, b) for a, b in zip(o, outs[0]))) else "DIFF" for o in outs]
    t = [[] for _ in dirs]
    with open("/tmp/okc-gpu.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for r in range(rounds):
            order = list(range(len(dirs))) + list(range(len(dirs)))[::-1]
            for i in order:
                flush.add_(1)
                a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                a.record(); gs[i].replay(); b.record(); torch.cuda.synchronize()
                t[i].append(a.elapsed_time(b) * 1e3 - emp)
        fcntl.flock(lk, fcntl.LOCK_UN)
    med = [statistics.median(x) for x in t]
    for d, v in zip(dirs, med): res[d].append(v)
    print(f"bs{bs} s{sd}: " + "  ".join(f"{d.rstrip('/').split('/')[-1]} {v:.2f}{'' if e == '=' else ' DIFF'}" for d, v, e in zip(dirs, med, eq)), flush=True)
    del gs
print("mean: " + "  ".join(f"{d.rstrip('/').split('/')[-1]} {statistics.mean(v):.2f}" for d, v in res.items()))
