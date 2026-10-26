# s12 merged-phase tk2h (TK2H 8 / 16, TK2H_PSC 0 / 1) vs main tk2f bitwise over many routings (incl. T=1, T=8): python tools/h_ab.py [n]
import sys, torch
sys.path.insert(0, '.'); sys.path.insert(0, 'tools')
import bench, submission
from layer_ab import mk, R, st
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
for d in cases:
    submission.TK2H = 0; ref = submission.kernel_fn(d, st).clone()
    for h, psc in ((8, 0), (16, 0), (8, 1), (16, 1)):
        submission.TK2H = h; submission.TK2H_PSC = psc; a = submission.kernel_fn(d, st).clone()
        torch.cuda.synchronize(); tot += 1
        if not torch.equal(a, ref):
            bad += 1; print("MISMATCH", h, psc, d["hidden_states"].shape[0], (a.float() - ref.float()).abs().max().item(), flush=True)
submission.TK2H = 0; submission.TK2H_PSC = 0
print("bad", bad, "of", tot)
