"""o11: interleaved H100 A/B, main (this dir) vs a copy (argv[1] dir) at bs argv[2], argv[3] seeds x argv[4] rounds (us)."""
import sys, statistics, torch, importlib.util
sys.path.insert(0, ".")
import bench, reference
import submission as main
spec = importlib.util.spec_from_file_location("subx", sys.argv[1] + "/submission.py")
subx = importlib.util.module_from_spec(spec); spec.loader.exec_module(subx)
bs, ns, nrnd = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
w = reference.generate_weights(reference.WEIGHTS_SEED)
stm = main.prepare(w); stx = subx.prepare(w)
if bs <= 32: print("cfg main", main._tk2f_cfg(bs, stm), "copy", subx._tk2f_cfg(bs, stx))
A, B = [], []
for sd in range(ns):
    d = reference.generate_input(bs=bs, sharding="ep8", seed=300 + sd)
    a, b = [], []
    for r in range(nrnd):
        for k in ((0, 1) if r % 2 == 0 else (1, 0)):
            t = bench.time_graph(lambda: (main.kernel_fn(d, stm) if k == 0 else subx.kernel_fn(d, stx)), 30)
            (a if k == 0 else b).append(t)
    A.append(statistics.median(a)); B.append(statistics.median(b))
    print(f"seed {sd}: main {A[-1]:7.2f} copy {B[-1]:7.2f} d {B[-1] - A[-1]:+5.2f}", flush=True)
print(f"bs {bs} mean main {statistics.mean(A):7.2f} copy {statistics.mean(B):7.2f} d {statistics.mean(B) - statistics.mean(A):+5.2f} us")
