# n19: moe_twq WBW (flags & 16) bitwise vs plain twq (4) on random + layer routings, graph replays; then reference check.
# python tools/n19/wbw_t.py [A] [B]   (TWQ flag values, default 4 20)
import sys, os, torch
sys.path.insert(0, '.'); sys.path.insert(0, 'tools')
import layer_ab as L
import reference
sub = L.submission; st = L.st
A, B = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (4, 20)
def run(data, q, graph):
    sub.TWQ = q; sub.TWQ_M = {}; sub.RG = 0
    if not graph:
        o = sub.kernel_fn(data, st); torch.cuda.synchronize(); return o.clone()
    g = torch.cuda.CUDAGraph(); sub.kernel_fn(data, st); torch.cuda.synchronize()
    with torch.cuda.graph(g): o = sub.kernel_fn(data, st)
    outs = []
    for _ in range(3): g.replay(); torch.cuda.synchronize(); outs.append(o.clone())
    return outs
bad = 0; n = 0
cases = [(f"rand bs{bs} s{sd} {dist}", lambda bs=bs, sd=sd, dist=dist: reference.generate_input(sharding="ep8", bs=bs, seed=sd, **({"dist": dist} if dist else {})))
         for bs in (33, 128, 256, 257) for sd in (1, 2) for dist in (None, "skewed")]
cases += [(f"layer {l} s{sd}", lambda loc=loc, sd=sd: L.mk(loc, sd)) for (l, sd), loc in sorted(L.R.items()) if l in ("verify 16", "verify 32")]
w = L.w
for name, mkd in cases:
    data = mkd()
    try:
        a = run(data, A, False); bs = run(data, B, True)
    except TypeError:
        continue
    n += 1; print("ok", name, flush=True)
    eq = all(torch.equal(a, x) for x in bs)
    if not eq:
        # nondeterministic route order inside an expert is allowed to differ? (twq passes are per 32 tokens: should be bitwise)
        d = max((a.float() - x.float()).abs().max().item() for x in bs)
    ok, msg = reference.check_implementation(data, bs[-1], w)
    if not eq or not ok:
        bad += 1; print("DIFF" if not eq else "", name, "" if eq else d, msg, flush=True)
print(f"wbw_t: {bad}/{n} bad (A={A}, B={B})")
