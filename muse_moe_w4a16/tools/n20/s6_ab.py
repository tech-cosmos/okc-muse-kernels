# n20: S6 vs SQ timing on the layer's routings (moe-only, cold-L2 graph replay median, A B A B rounds).
# python tools/n20/s6_ab.py [rounds] [tk2h W or -1]
import sys, torch
sys.path.insert(0, '.'); sys.path.insert(0, 'tools')
import submission as S
S.CUDA_SRC = S.CUDA_SRC.replace("#define TK2H_P4 0 ", "#define TK2H_P4 1 ")
import layer_ab as L
st = L.st
st["w13sq"] = S._sq_pack(st["w13q"], st["w13s2"]); st["w2sq"] = S._sq_pack(st["w2q"], st["w2s2"])
rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 2
W = int(sys.argv[2]) if len(sys.argv) > 2 else -1
for load in ("decode 1", "decode 16", "decode 32", "verify 1"):
    r = {False: [], True: []}
    for (l, sd), loc in sorted(L.R.items()):
        if l != load: continue
        d = L.mk(loc, sd)
        S.TK2H_M = {} if W < 0 else {d["hidden_states"].shape[0]: W}
        for _ in range(rounds):
            for on in (False, True):
                S.TK2F_S6_ON = on
                r[on].append(L.tg(lambda: S.kernel_fn(d, st)))
    m = {k: sum(v) / len(v) for k, v in r.items()}
    print(f"{load} (tk2h {W}): sq {m[False]:.2f}  s6 {m[True]:.2f}  d {m[True] - m[False]:+.2f} us", flush=True)
