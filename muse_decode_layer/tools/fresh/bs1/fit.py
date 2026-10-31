"""fresh x10: standalone muse_moe_w4a16 bs1: graph time (cold L2) vs distinct local experts over seeds; LSQ fit.
Run from muse_moe_w4a16/: python ../muse_decode_layer/tools/fresh/bs1/fit.py [nseeds] [bs]"""
import sys, torch
sys.path.insert(0, ".")
import bench, reference, submission
n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
bs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
_, st = bench.state()
rows = []
for sd in range(100, 100 + n):
    data = reference.generate_input(bs=bs, sharding="ep8", seed=sd)
    ids = data["topk_ids"] if "topk_ids" in data else None
    if ids is None:
        ids = [v for k, v in data.items() if torch.is_tensor(v) and v.dtype in (torch.int32, torch.int64) and v.dim() == 2][0]
    ls = int(data.get("local_start", 0)) if "local_start" in data else None
    t = bench.time_graph(lambda: submission.kernel_fn(data, st), 30) - bench.empty_cost()
    rows.append((sd, ids, t))
import numpy as np
E = [int(((i >= 0) & (i < 32)).sum().item()) for _, i, _ in [(a, b.flatten().unique(), c) for a, b, c in rows]]
T = [t for _, _, t in rows]
for e in sorted(set(E)):
    v = sorted(t for t, x in zip(T, E) if x == e); print(f"E={e}: n={len(v)} med {v[len(v)//2]:.2f} min {v[0]:.2f} max {v[-1]:.2f}")
A = np.vstack([np.ones(len(E)), E]).T; c = np.linalg.lstsq(A, np.array(T), rcond=None)[0]
print(f"fit: t = {c[0]:.2f} + {c[1]:.2f} * E (us)")
