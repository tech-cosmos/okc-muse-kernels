# proj n5s11: KQ (2^kq K pieces, last-ticket sum) correctness vs twm on big-M routings
import sys; sys.path.insert(0, '.')
import torch, bench, submission as S, reference
w, st = bench.state()
bad = 0; worst = 0
cases = [dict(bs=b, seed=s) for b in (128, 256) for s in range(20, 26)] + [dict(bs=128, dist="skewed", seed=5), dict(bs=257, seed=3), dict(bs=33, seed=4), dict(bs=64, seed=6)]
for c in cases:
    data = reference.generate_input(sharding="ep8", **c)
    M = c["bs"]
    print(c, flush=True)
    S.TWF_SPLIT_M = {}; S.TWM_KQ_M = {}
    ref = S.kernel_fn(data, st).float(); torch.cuda.synchronize()
    for ns, kq in ((66, 0), (66, 1), (33, 2), (16, 3), (40, 2), (132, 1)):
        S.TWF_SPLIT_M = {M: ns}; S.TWM_KQ_M = {M: kq}
        for rep in range(3):
            out = S.kernel_fn(data, st); torch.cuda.synchronize()
            ok, msg = reference.check_implementation(data, out, w) if rep == 0 else (True, "")
            d = (out.float() - ref).abs().max().item()
            worst = max(worst, d)
            if not ok or d > 0.05: bad += 1; print("BAD", c, ns, kq, msg, d, flush=True)
S.TWF_SPLIT_M = {}; S.TWM_KQ_M = {}
print("bad", bad, "worst maxdiff", worst)
