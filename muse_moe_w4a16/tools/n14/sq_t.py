# n14: TK2F_SQ bitwise vs off (layer routings + random seeds, M 1..32, reps), then H100 layer-routing timing A/B.
import sys, os, torch
sys.path.insert(0, '.'); sys.path.insert(0, 'tools')
import layer_ab as L, reference
S, st = L.submission, L.st
bad = 0; n = 0
cases = [(l, L.mk(loc, sd)) for (l, sd), loc in sorted(L.R.items()) if not l.startswith("verify 1") or True]
cases = [(l, d) for l, d in cases if d["hidden_states"].shape[0] <= 32]
for M in (1, 2, 4, 8, 16, 24, 32):
    for sd in range(8):
        d = reference.generate_input(bs=M, sharding="ep8", seed=7000 + 31 * M + sd); cases.append((f"rand{M}", d))
for l, d in cases:
    S.TK2F_SQ = 0; ref = S.kernel_fn(d, st).clone()
    S.TK2F_SQ = 1
    for r in range(3):
        o = S.kernel_fn(d, st); n += 1
        if not torch.equal(o, ref): bad += 1; print("NOT BITWISE", l, r, (o.float() - ref.float()).abs().max().item())
torch.cuda.synchronize(); print(f"bitwise: {n - bad}/{n}", flush=True)
for load in ("decode 1", "decode 16", "decode 32", "verify 1"):
    r = {0: [], 1: []}
    for rnd in range(3):
        for (l, sd), loc in sorted(L.R.items()):
            if l != load: continue
            data = L.mk(loc, sd)
            for v in (0, 1):
                S.TK2F_SQ = v; r[v].append(L.tg(lambda: S.kernel_fn(data, st)))
    print(f"{load}: off {sum(r[0])/len(r[0]):.1f}  sq {sum(r[1])/len(r[1]):.1f}", flush=True)
