import sys, torch
sys.path.insert(0, ".")
import reference, submission, bench
w, st = bench.state()
out = {}
cases = [(b, s, "normal") for b in (33, 128, 256, 257) for s in range(1, 9)] + [(128, s, "skewed") for s in range(1, 5)] + [(256, s, "skewed") for s in range(1, 5)]
for bs, seed, dist in cases:
    d = reference.generate_input(bs=bs, sharding="ep8", seed=seed, dist=dist)
    o1 = submission.kernel_fn(d, st).clone()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): o2 = submission.kernel_fn(d, st)
    g.replay(); g.replay(); torch.cuda.synchronize()
    out[(bs, seed, dist)] = (o1.cpu(), o2.clone().cpu())
torch.save(out, sys.argv[1])
print("ok", len(out))
