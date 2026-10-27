# n17: TK2G_PAR4 bitwise check: SUB=<module> OUT=<file.pt> python tools/n17/par4_t.py ; then compare two OUT files
# (old = TK2G_PAR4 0 build, new = 1). tk2h GS16 W 2/3/4 (+SQ) on the layer's d16/d32 routings + random T 9..32.
import sys, os, torch
sys.path.insert(0, '.'); sys.path.insert(0, 'tools')
import layer_ab as L, reference
S, st = L.submission, L.st
cases = [(f"{l} {sd}", L.mk(loc, sd)) for (l, sd), loc in sorted(L.R.items())]
cases = [(l, d) for l, d in cases if 9 <= d["hidden_states"].shape[0] <= 32]
for M in (9, 12, 16, 20, 24, 32):
    for sd in range(6):
        cases.append((f"rand{M} {sd}", reference.generate_input(bs=M, sharding="ep8", seed=8000 + 31 * M + sd)))
res = {}
for W in (2, 3, 4):
    for l, d in cases:
        S.TK2H_M = {d["hidden_states"].shape[0]: W}
        o = S.kernel_fn(d, st).clone()
        o2 = S.kernel_fn(d, st)
        assert torch.equal(o, o2), (W, l)
        res[(W, l)] = o.cpu()
torch.save(res, os.environ["OUT"]); print("saved", len(res))
