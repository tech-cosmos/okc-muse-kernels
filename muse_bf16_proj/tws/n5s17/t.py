# n5s17 TWS correctness: steal on vs off on big-M routings (repeat reps: nondeterministic schedules)
import sys, os; sys.path.insert(0, '.')
import torch, bench, submission as S, reference
w, st = bench.state()
bad = 0; worst = 0
cases = [dict(bs=b, seed=s) for b in (128, 256) for s in range(20, 20 + int(os.environ.get("NS", 4)))] + [dict(bs=128, dist="skewed", seed=5), dict(bs=257, seed=3), dict(bs=33, seed=4), dict(bs=64, seed=6)]
V = [int(v, 0) for v in os.environ.get("STW", "160,0x1030A0,0x40860,0x40460").split(",")]
for c in cases:
    data = reference.generate_input(sharding="ep8", **c)
    M = c["bs"]
    S.TWM_STEAL_M = {}
    ref = S.kernel_fn(data, st).float(); torch.cuda.synchronize()
    for stw in V:
        S.TWM_STEAL_M = {M: stw}
        for rep in range(4):
            out = S.kernel_fn(data, st); torch.cuda.synchronize()
            ok, msg = reference.check_implementation(data, out, w) if rep == 0 else (True, "")
            d = (out.float() - ref).abs().max().item()
            worst = max(worst, d)
            if not ok or d > 0.05: bad += 1; print("BAD", c, hex(stw), msg, d, flush=True)
        print(" ", hex(stw), d, flush=True)
    print(c, "done", flush=True)
S.TWM_STEAL_M = {}
print("bad", bad, "worst maxdiff", worst)
