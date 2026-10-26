# SUB=<module> python outdump.py : outputs of kernel_fn on big-M routings (seeds_big set + layer v16/v32) -> out_<SUB>.pt
import sys, os
sys.path.insert(0, '/tmp/moe16')
import layer_ab as L, torch, reference
sub = L.submission
outs = []
for bs, dist in [(128, "normal"), (256, "normal"), (128, "skewed"), (64, "normal"), (257, "normal")]:
    for sd in range(6):
        data = reference.generate_input(bs=bs, sharding="ep8", seed=9000 + sd, dist=dist)
        outs.append(sub.kernel_fn(data, L.st).clone())
for (l, sd), loc in sorted(L.R.items()):
    if l in ("verify 16", "verify 32"):
        for o in ({}, {256: 76}):
            sub.TWM_ORDER_M = o
            outs.append(sub.kernel_fn(L.mk(loc, sd), L.st).clone())
torch.save([o.cpu() for o in outs], f"/tmp/moe16/out_{os.environ['SUB']}.pt")
print("saved", len(outs))
