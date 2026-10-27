# s24: TWQ_NE (moe_norme after moe_twq, per-expert gemm2 flags) vs TWQ_NE 0: must be BITWISE equal; eager + 3 replays,
# reference check. python tools/s24/ne_t.py   (twq flags: plain q4, n21 split s33h, s23 split 66 + HOTQ 32)
import sys, torch
sys.path.insert(0, '.'); sys.path.insert(0, 'tools')
import layer_ab as L
import reference
sub = L.submission; st = L.st
QS = [4, 73988, 536953348]
def run(data, q, ne, graph):
    sub.TWQ = 0; sub.TWQ_M = {data["config"]["bs"]: q}; sub.HOTQ_M = {}; sub.TWQ_NE = ne
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
bad = n = 0
for name, mkd in cases:
    try: data = mkd()
    except TypeError: continue
    for q in QS:
        base = run(data, q, 0, False)
        a = run(data, q, 1, False); bs = run(data, q, 1, True)
        n += 1
        same = torch.equal(a, base) and all(torch.equal(base, x) for x in bs)
        ok, msg = reference.check_implementation(data, bs[-1], L.w)
        if not same or not ok:
            bad += 1; print("BAD", name, q, "same" if same else "DIFF", msg, flush=True)
    print("ok", name, flush=True)
print(f"ne_t: {bad}/{n} bad")
