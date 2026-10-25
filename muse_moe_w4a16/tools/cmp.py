import sys, os, statistics
sys.path.insert(0, os.environ.get("MV", ".")); sys.path.insert(1, ".")
import torch, bench, submission, reference
print(submission.__file__)
w, st = bench.state()
for bs in (16, 32):
    ts = []
    for sd in range(9000, 9008):
        d = reference.generate_input(bs=bs, sharding="ep8", seed=sd)
        ts.append(bench.time_graph(lambda: submission.kernel_fn(d, st), 20))
    print(bs, [round(x, 1) for x in ts], "median %.1f" % statistics.median(ts), flush=True)
