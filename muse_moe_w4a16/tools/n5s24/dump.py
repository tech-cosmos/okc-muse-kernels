# n5s24: outputs of a submission.py on decode + big-M routings (eager + graph), for bitwise A/B.
# usage (from muse_moe_w4a16/): python tools/n5s24/dump.py <path/to/submission.py> <out.pt>
import sys, importlib.util, torch
sys.path.insert(0, ".")
import reference
spec = importlib.util.spec_from_file_location("submission", sys.argv[1])
sub = importlib.util.module_from_spec(spec); sys.modules["submission"] = sub; spec.loader.exec_module(sub)
st = sub.prepare(reference.generate_weights(reference.WEIGHTS_SEED))
out = {}
cases = [(b, s, "normal") for b in (1, 2, 8, 16, 32, 33, 128, 256) for s in range(1, 7)]
cases += [(b, s, "skewed") for b in (16, 32, 128, 256) for s in range(1, 4)]
for bs, seed, dist in cases:
    d = reference.generate_input(bs=bs, sharding="ep8", seed=seed, dist=dist)
    o1 = sub.kernel_fn(d, st).clone()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): o2 = sub.kernel_fn(d, st)
    g.replay(); g.replay(); torch.cuda.synchronize()
    out[(bs, seed, dist)] = (o1.cpu(), o2.clone().cpu())
torch.save(out, sys.argv[2])
print("ok", len(out))
