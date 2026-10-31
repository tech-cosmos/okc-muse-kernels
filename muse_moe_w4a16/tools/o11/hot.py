"""o11: moe_hot options on moe-board bs128/bs256 routings (H100). usage: python tools/o11/hot.py <bs> <nseeds> <variants json>
variants: {"name": {M: hotvalue or 0}}. Prints per-seed us per variant (+ hot count), bitwise check vs the first variant."""
import json, sys, statistics, torch
sys.path.insert(0, ".")
import bench, reference, submission
bs, ns, vs = int(sys.argv[1]), int(sys.argv[2]), json.loads(sys.argv[3])
w, st = bench.state()
res = {k: [] for k in vs}
for sd in range(ns):
    data = reference.generate_input(bs=bs, sharding="ep8", seed=100 + sd)
    ids = data["topk_ids"]; c = torch.bincount(ids[ids < 32], minlength=32)
    nh = int(((c > 64) & (c <= 128)).sum()); nb = int((c > 128).sum())
    outs = {}
    for k, hv in vs.items():
        submission.HOT_M = {int(m): v for m, v in hv.items() if v}
        o = submission.kernel_fn(data, st); torch.cuda.synchronize(); outs[k] = o.clone()
    k0 = next(iter(vs))
    for k in vs:
        assert torch.equal(outs[k], outs[k0]), (sd, k)
    ok, msg = reference.check_implementation(data, outs[k0], w)
    assert ok, msg
    row = []
    for rep in range(2):
        for k, hv in (vs.items() if rep == 0 else reversed(list(vs.items()))):
            submission.HOT_M = {int(m): v for m, v in hv.items() if v}
            res[k].append((sd, nh, nb, bench.time_graph(lambda: submission.kernel_fn(data, st), 20)))
    line = " ".join(f"{k}={statistics.mean(t for s_, _, _, t in res[k] if s_ == sd):7.1f}" for k in vs)
    print(f"seed {sd:3d} hot65 {nh} big {nb}  {line}", flush=True)
for k in vs:
    a = [t for _, _, _, t in res[k]]
    h = [t for _, nh, _, t in res[k] if nh]
    z = [t for _, nh, _, t in res[k] if not nh]
    print(f"{k:10s} mean {statistics.mean(a):7.2f} median {statistics.median(a):7.2f} | hot {statistics.mean(h) if h else 0:7.2f} ({len(h)//2}) nohot {statistics.mean(z) if z else 0:7.2f}")
