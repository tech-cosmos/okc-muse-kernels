import sys, torch
sys.path.insert(0, ".")
import reference, submission
bad = 0
for seed in range(3):
    for b, L in [(1,1),(16,1),(32,1),(1,8),(16,8),(32,8)]:
        c = dict(batch=b, ctx=80000, heads="tp8", layer="sliding", page=1, q_len=L, dist="soft")
        if b > 1: c["spread"] = "agentic"
        data = reference.generate_input(seed=777 + seed * 13 + b + L, **c)
        out = submission.kernel_fn(data); torch.cuda.synchronize()
        ok, msg = reference.check_implementation(data, reference.collect(data, out))
        bad += not ok
        print("OK " if ok else "BAD", b, L, msg[:60])
print("bad", bad)
