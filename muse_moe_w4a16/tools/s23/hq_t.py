# s23: moe_hot<NH 64> + moe_twq (bit 20 skip) vs plain twq: determinism (eager + 3 replays), diff, reference check.
# python tools/s23/hq_t.py 16 24 ...   (hot grid sizes)
import sys, torch
sys.path.insert(0, '.'); sys.path.insert(0, 'tools')
import layer_ab as L
import reference
sub = L.submission; st = L.st
GS = [int(a) for a in sys.argv[1:]] or [16]
def run(data, hq, graph):
    sub.TWQ = 4; sub.TWQ_M = {}; sub.HOTQ_M = {data["config"]["bs"]: hq} if hq else {}
    if not graph:
        o = sub.kernel_fn(data, st); torch.cuda.synchronize(); return o.clone()
    g = torch.cuda.CUDAGraph(); sub.kernel_fn(data, st); torch.cuda.synchronize()
    with torch.cuda.graph(g): o = sub.kernel_fn(data, st)
    outs = []
    for _ in range(3): g.replay(); torch.cuda.synchronize(); outs.append(o.clone())
    return outs
cases = [(f"rand bs{bs} s{sd} {dist}", lambda bs=bs, sd=sd, dist=dist: reference.generate_input(sharding="ep8", bs=bs, seed=sd, **({"dist": dist} if dist else {})))
         for bs in (33, 128, 256, 257) for sd in (1, 2) for dist in (None, "skewed")]
cases += [(f"layer {l} s{sd}", lambda loc=loc, sd=sd: L.mk(loc, sd)) for (l, sd), loc in sorted(L.R.items()) if l in ("verify 16", "verify 32")]
bad = n = 0; dmax = 0.0
for name, mkd in cases:
    try: data = mkd()
    except TypeError: continue
    base = run(data, 0, False)
    for g in GS:
        a = run(data, g, False); bs = run(data, g, True)
        n += 1
        det = all(torch.equal(a, x) for x in bs)
        d = (a.float() - base.float()).abs().max().item(); dmax = max(dmax, d)
        ok, msg = reference.check_implementation(data, bs[-1], L.w)
        if not det or not ok:
            bad += 1; print("BAD", name, g, "det" if det else "NONDET", d, msg, flush=True)
    print("ok", name, f"maxdiff vs plain {d:.3g}", flush=True)
print(f"hq_t: {bad}/{n} bad, max |hq - plain| {dmax:.3g}")
