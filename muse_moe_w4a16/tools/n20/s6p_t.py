# n20: S6P (padded S6 images) bitwise vs SQ / S6 + H100 timing on the layer routings (DIAG-style build, no diag run).
import sys, torch
sys.path.insert(0, '.'); sys.path.insert(0, 'tools')
import submission as S
S.DIAG = 1; S._DIAG_DONE = True
import layer_ab as L, reference
st = L.st
assert "w13s6p" in st and "w13sq" in st
V = {"sq": {"TK2F_S6_ON": False, "TK2F_S6P": False}, "s6": {"TK2F_S6_ON": True, "TK2F_S6P": False},
     "s6p": {"TK2F_S6_ON": True, "TK2F_S6P": True}}
def run(v, d):
    for k, x in V[v].items(): setattr(S, k, x)
    return S.kernel_fn(d, st)
cases = [(f"{l} {sd}", L.mk(loc, sd)) for (l, sd), loc in sorted(L.R.items()) if loc.shape[0] <= 32]
bad = 0
for h in (None, 3):
    for l, d in cases:
        S.TK2H_M = {} if h is None else {d["hidden_states"].shape[0]: h}
        o = [run(v, d).clone() for v in ("sq", "s6", "s6p", "s6p")]
        bad += not all(torch.equal(o[0], x) for x in o[1:])
print("bitwise bad", bad, "of", 2 * len(cases), flush=True)
for h in (None, 3):
    for load in ("decode 16", "decode 32", "verify 1"):
        r = {v: [] for v in V}
        for (l, sd), loc in sorted(L.R.items()):
            if l != load: continue
            d = L.mk(loc, sd); S.TK2H_M = {} if h is None else {d["hidden_states"].shape[0]: h}
            for v in V:
                r[v].append(L.tg(lambda: run(v, d)))
        print(load, h, "  ".join(f"{v} {sum(x)/len(x):.2f}" for v, x in r.items()), flush=True)
