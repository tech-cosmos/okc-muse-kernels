# n20: S6 (6-bit scale-code SQ stages) bitwise vs the 8-bit SQ stages: tk2f (204 / 84204) and tk2h GS16 W3 on the
# layer's decode routings + random T 1..32, 2 replays each. python tools/n20/s6_t.py  (from muse_moe_w4a16/)
import sys, torch
sys.path.insert(0, '.'); sys.path.insert(0, 'tools')
import submission as S
S.CUDA_SRC = S.CUDA_SRC.replace("#define TK2H_P4 0 ", "#define TK2H_P4 1 ")  # build the tk2h SQ/S6 kernels
import layer_ab as L, reference
st = L.st
assert "w13s6" in st, "S6 images not built"
st["w13sq"] = S._sq_pack(st["w13q"], st["w13s2"]); st["w2sq"] = S._sq_pack(st["w2q"], st["w2s2"])
cases = [(f"{l} {sd}", L.mk(loc, sd)) for (l, sd), loc in sorted(L.R.items())]
cases = [(l, d) for l, d in cases if d["hidden_states"].shape[0] <= 32]
for M in (1, 2, 5, 8, 9, 16, 24, 32):
    for sd in range(4):
        cases.append((f"rand{M} {sd}", reference.generate_input(bs=M, sharding="ep8", seed=9100 + 37 * M + sd)))
bad = n = 0
for h in (None, 3):
    for l, d in cases:
        M = d["hidden_states"].shape[0]
        S.TK2H_M = {} if h is None else {M: h}
        outs = []
        for on in (False, True, False, True):
            S.TK2F_S6_ON = on
            outs.append(S.kernel_fn(d, st).clone())
        torch.cuda.synchronize()
        ok = all(torch.equal(outs[0], o) for o in outs[1:])
        n += 1; bad += not ok
        if not ok: print("DIFF", h, l, (outs[0].float() - outs[1].float()).abs().max().item(), flush=True)
S.TK2H_M = {}; S.TK2F_S6_ON = True
for bs in (1, 4, 8, 16, 32):
    d = reference.generate_input(bs=bs, sharding="ep8", seed=777 + bs)
    ok, msg = reference.check_implementation(d, S.kernel_fn(d, st), L.w)
    print("ref", bs, ok, msg)
print(f"bitwise S6 vs SQ: {bad} bad of {n}")
