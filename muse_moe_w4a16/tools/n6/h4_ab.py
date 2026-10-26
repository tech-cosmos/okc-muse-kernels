# n6: tk2h (TK2H 8 / 16, weights 0/3) with TK2H_PF4 0/1 vs main (tk2f 84204 + moe_norm3) bitwise, layer routings + random.
# Needs a TK2_EXP 1 copy: mkdir /tmp/n6; sed 's/^#define TK2_EXP 0 /#define TK2_EXP 1 /' submission.py > /tmp/n6/sub_x.py;
# (or a TK2H_P4 1 copy with P4ONLY=1)
# cd muse_moe_w4a16 && SUB=sub_x PYTHONPATH=/tmp/n6 python tools/n6/h4_ab.py [n]
import sys, os, torch
sys.path.insert(0, '.'); sys.path.insert(0, 'tools')
from layer_ab import mk, R, st, submission
g = torch.Generator(device='cuda').manual_seed(6)
bad = tot = 0
cases = [mk(loc, sd) for (l, sd), loc in sorted(R.items()) if loc.shape[0] <= 32]
for r in range(int(sys.argv[1]) if len(sys.argv) > 1 else 200):
    T = (1, 8, 9, 12, 16, 24, 32)[r % 7]
    ids = torch.randint(0, (6, 12, 40, 3)[r % 4], (T, 8), device='cuda', generator=g, dtype=torch.int32)
    ids = torch.where(ids >= 32, torch.full_like(ids, 200), ids) + st["start"]
    tw = torch.rand(T, 8, device='cuda', generator=g)
    x = torch.randn(T, 4096, device='cuda', generator=g).bfloat16()
    cases.append({"hidden_states": x, "topk_ids": ids.int(), "topk_weights": tw, "config": {"bs": T, "local_start": st["start"], "local_count": st["count"]}})
V = [(h, w, p) for h in (8, 16) for w in ((0,) if h == 8 else (0, 3)) for p in ((1,) if os.environ.get("P4ONLY") else (0, 1))]  # P4ONLY: TK2H_P4 build
for d in cases:
    submission.TK2H = 0; ref = submission.kernel_fn(d, st).clone()
    for h, w, p in V:
        submission.TK2H = h; submission.TK2H_W = w; submission.TK2H_PF4 = p
        for _ in range(2):  # twice: counters re-zeroed by the norm
            a = submission.kernel_fn(d, st).clone(); torch.cuda.synchronize(); tot += 1
            if not torch.equal(a, ref):
                bad += 1; print("MISMATCH", h, w, p, d["hidden_states"].shape[0], (a.float() - ref.float()).abs().max().item(), flush=True)
submission.TK2H = 0; submission.TK2H_PF4 = 0
print("bad", bad, "of", tot)
