# GS16 vs GS8 bitwise over many routings (skewed random + the layer's): python tools/gs_ab.py [n]
import sys, torch
sys.path.insert(0, '.')
import bench, submission
from layer_ab import mk, R, st
g = torch.Generator(device='cuda').manual_seed(5)
bad = tot = 0
cases = [mk(loc, sd) for (l, sd), loc in sorted(R.items()) if 8 < loc.shape[0] <= 32]
for r in range(int(sys.argv[1]) if len(sys.argv) > 1 else 200):
    T = (9, 12, 16, 24, 32)[r % 5]
    ids = torch.randint(0, (6, 12, 40, 3)[r % 4], (T, 8), device='cuda', generator=g, dtype=torch.int32)
    ids = torch.where(ids >= 32, torch.full_like(ids, 200), ids) + st["start"]
    tw = torch.rand(T, 8, device='cuda', generator=g)
    x = torch.randn(T, 4096, device='cuda', generator=g).bfloat16()
    cases.append({"hidden_states": x, "topk_ids": ids.int(), "topk_weights": tw, "config": {"bs": T, "local_start": st["start"], "local_count": st["count"]}})
for d in cases:
    submission.TK2G = 1; a = submission.kernel_fn(d, st).clone()
    submission.TK2G = 0; b = submission.kernel_fn(d, st).clone()
    torch.cuda.synchronize(); tot += 1
    if not torch.equal(a, b):
        bad += 1; print("MISMATCH", d["hidden_states"].shape[0], (a.float() - b.float()).abs().max().item(), flush=True)
submission.TK2G = 1
print("bad", bad, "of", tot)
