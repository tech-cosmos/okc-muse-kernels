# s22: TWM_FN (moe_norm fused into moe_twm's drain) vs main: correctness + bitwise, many big-M routings, repeated replays
import sys, os; sys.path.insert(0, os.path.expanduser('~/r-moe/muse_moe_w4a16'))
import torch, bench, submission as S, reference
w, st = bench.state()
MS = (64, 128, 256, 257, 100, 33)
V = {"main": {}, "fn": {"TWM_FN": 1}, "fn_o76": {"TWM_FN": 1, "TWM_ORDER": 76}, "fn_hot_a22": {"TWM_FN": 1, "TWM_HOT": 1, "TWF_SPLIT": 1022},
     "fn_tail": {"TWM_FN": 1, "TWM_TAIL": 32}}
old = {k: getattr(S, k) for d in V.values() for k in d}
nbad = 0; ref_main = {}
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
for name, d in V.items():
    for k, v in old.items(): setattr(S, k, v)
    for k, v in d.items(): setattr(S, k, v)
    bad = 0; nd = 0
    for bs in MS:
        for dist in ("normal", "skewed"):
            for sd in range(N):
                data = reference.generate_input(bs=bs, sharding="ep8", seed=9100 + sd, dist=dist)
                for rep in range(2):
                    out = S.kernel_fn(data, st); torch.cuda.synchronize()
                    ok, msg = reference.check_implementation(data, out, w)
                    if not ok: bad += 1; print("FAIL", name, bs, dist, sd, msg, flush=True)
                    key = (bs, dist, sd)
                    if name == "main" and rep == 0: ref_main[key] = out.clone()
                    elif not torch.equal(out, ref_main[key]): nd += 1
    print(name, "bad", bad, "not bitwise vs main", nd, "fnb clean", int(st["fnb"].abs().sum().item()) == 0, flush=True); nbad += bad
print("TOTAL bad", nbad)
