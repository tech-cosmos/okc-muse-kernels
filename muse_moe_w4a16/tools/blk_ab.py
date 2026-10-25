# A/B of tk2_map block sizes on decode (local graph timing, cold L2). usage: blk_ab.py 16,32 0,2,4,8 [nseeds]
import sys, statistics
sys.path.insert(0, ".")
import torch, bench, submission, reference
w, st = bench.state()
bss = [int(b) for b in sys.argv[1].split(",")]
vs = [int(v) for v in sys.argv[2].split(",")]
ns = int(sys.argv[3]) if len(sys.argv) > 3 else 10
for bs in bss:
    res = {v: [] for v in vs}
    for sd in range(9000, 9000 + ns):
        d = reference.generate_input(bs=bs, sharding="ep8", seed=sd)
        for v in vs:
            submission.TK2_BLK = {bs: v}
            res[v].append(bench.time_graph(lambda: submission.kernel_fn(d, st), 30))
    print(bs, "  ".join("blk%d %.1f (mean %.1f)" % (v, statistics.median(res[v]), statistics.mean(res[v])) for v in vs), flush=True)
    print("   per seed:", [[round(res[v][i], 1) for v in vs] for i in range(ns)], flush=True)
