"""ncu driver for standalone moe: python prof.py [bs] [seed] (profiles one call). Run from muse_moe_w4a16/."""
import sys, torch
sys.path.insert(0, ".")
import bench, reference, submission
bs = int(sys.argv[1]) if len(sys.argv) > 1 else 1; sd = int(sys.argv[2]) if len(sys.argv) > 2 else 100
_, st = bench.state()
data = reference.generate_input(bs=bs, sharding="ep8", seed=sd)
for _ in range(3): submission.kernel_fn(data, st)
torch.cuda.synchronize(); torch.cuda.profiler.start(); submission.kernel_fn(data, st); torch.cuda.synchronize(); torch.cuda.profiler.stop()
