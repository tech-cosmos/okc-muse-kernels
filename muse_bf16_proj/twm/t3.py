# proj n5s10: moe_tw3 vs moe_twm: correctness (big-M cases) + H100 timing bs128/256 (seeds 7,8,9), interleaved
import sys; sys.path.insert(0, '.')
import torch, bench, submission, reference
class All(dict):
    def get(self, k, d=None): return 1
w, st = bench.state()
lib = submission._lib()
bad = 0
for c in [dict(bs=33, seed=1), dict(bs=257, seed=2), dict(bs=128, dist="skewed", seed=3), dict(bs=128, seed=4),
          dict(bs=256, seed=5), dict(bs=64, seed=6), dict(bs=256, seed=9), dict(bs=256, seed=8)]:
    data = reference.generate_input(sharding="ep8", **c)
    submission.TW3_M = All()
    out = submission.kernel_fn(data, st); torch.cuda.synchronize()
    submission.TW3_M = {}
    ref = submission.kernel_fn(data, st); torch.cuda.synchronize()
    ok, msg = reference.check_implementation(data, out, w)
    d = (out.float() - ref.float()).norm() / ref.float().norm()
    if not ok: bad += 1
    print(c, "ok" if ok else "FAIL " + msg, f"rel vs twm {d:.2e}", flush=True)
print("bad", bad)
if bad or len(sys.argv) > 1 and sys.argv[1] == "noperf": sys.exit()
for rnd in range(2):
    for bs in (128, 256):
        r = []
        for v in ({}, All()):
            submission.TW3_M = v
            r.append([round(bench.run_shape(bs, seeds=(s,)), 1) for s in (7, 8, 9)])
        print(f"round {rnd} bs {bs}: twm {r[0]}  tw3 {r[1]}", flush=True)
