# s13 PF (cfg 81204/82204/83204: next-segment x L1 prefetch) vs main tk2f bitwise over many routings: python tools/pf_ab.py [n] [cfg]
import sys, torch
sys.path.insert(0, '.'); sys.path.insert(0, 'tools')
import bench, submission
from layer_ab import mk, R, st
g = torch.Generator(device='cuda').manual_seed(6)
CFGS = [int(c) for c in (sys.argv[2] if len(sys.argv) > 2 else "81204,82204,83204,87204").split(",")]
bad = tot = 0
cases = [mk(loc, sd) for (l, sd), loc in sorted(R.items()) if loc.shape[0] <= 32]
for r in range(int(sys.argv[1]) if len(sys.argv) > 1 else 200):
    T = (1, 8, 9, 12, 16, 24, 32)[r % 7]
    ids = torch.randint(0, (6, 12, 40, 3)[r % 4], (T, 8), device='cuda', generator=g, dtype=torch.int32)
    ids = torch.where(ids >= 32, torch.full_like(ids, 200), ids) + st["start"]
    tw = torch.rand(T, 8, device='cuda', generator=g)
    x = torch.randn(T, 4096, device='cuda', generator=g).bfloat16()
    cases.append({"hidden_states": x, "topk_ids": ids.int(), "topk_weights": tw, "config": {"bs": T, "local_start": st["start"], "local_count": st["count"]}})
for d in cases:
    submission.TK2F = submission.TK2F_DYN = 204; ref = submission.kernel_fn(d, st).clone()
    for cfg in CFGS:
        submission.TK2F = submission.TK2F_DYN = cfg; a = submission.kernel_fn(d, st).clone()
        torch.cuda.synchronize(); tot += 1
        if not torch.equal(a, ref):
            bad += 1; print("MISMATCH", cfg, d["hidden_states"].shape[0], (a.float() - ref.float()).abs().max().item(), flush=True)
submission.TK2F = submission.TK2F_DYN = 204
print("bad", bad, "of", tot)
