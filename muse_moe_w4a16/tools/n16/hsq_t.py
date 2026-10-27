# n16: tk2h (GS16 + PF4, TK2H_M) on SQ stages vs the 3-copy stages: bitwise, then H100 layer-routing timing.
# SUB=sub_h4 (submission with #define TK2H_P4 1) python tools/n16/hsq_t.py
import sys, torch
sys.path.insert(0, '.'); sys.path.insert(0, 'tools')
import layer_ab as L, reference
S, st = L.submission, L.st
bad = 0; n = 0
cases = [(l, L.mk(loc, sd)) for (l, sd), loc in sorted(L.R.items())]
cases = [(l, d) for l, d in cases if 9 <= d["hidden_states"].shape[0] <= 32]
for M in (9, 12, 16, 20, 24, 32):
    for sd in range(8):
        d = reference.generate_input(bs=M, sharding="ep8", seed=8000 + 31 * M + sd); cases.append((f"rand{M}", d))
for W in (2, 3):
    for l, d in cases:
        M = d["hidden_states"].shape[0]
        S.TK2H_M = {M: W}
        S.TK2H_SQ = 0; ref = S.kernel_fn(d, st).clone()
        assert S._tk2f_cfg(M, st) < 10000000
        S.TK2H_SQ = 1; assert S._tk2f_cfg(M, st) >= 10000000
        for r in range(3):
            o = S.kernel_fn(d, st); n += 1
            if not torch.equal(o, ref): bad += 1; print("NOT BITWISE", W, l, r, (o.float() - ref.float()).abs().max().item())
torch.cuda.synchronize(); print(f"bitwise: {n - bad}/{n}", flush=True)
for load in ("decode 16", "decode 32"):
    r = {}
    for rnd in range(3):
        for (l, sd), loc in sorted(L.R.items()):
            if l != load: continue
            data = L.mk(loc, sd); M = data["hidden_states"].shape[0]
            for k, (hm, sq) in {"gs8sq": ({}, 0), "h16w3": ({M: 3}, 0), "h16w3sq": ({M: 3}, 1)}.items():
                S.TK2H_M = hm; S.TK2H_SQ = sq; r.setdefault(k, []).append(L.tg(lambda: S.kernel_fn(data, st)))
    print(load + ": " + "  ".join(f"{k} {sum(v)/len(v):.1f}" for k, v in r.items()), flush=True)
