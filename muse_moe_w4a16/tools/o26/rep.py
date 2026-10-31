"""o26: repeat timing of chosen (bs, seed): python tools/o26/rep.py bs seed,seed,..."""
import sys, os, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch, bench, reference, submission
bs = int(sys.argv[1]); seeds = [int(s) for s in sys.argv[2].split(",")]
_, st = bench.state()
ds = {sd: reference.generate_input(bs=bs, sharding="ep8", seed=sd) for sd in seeds}
ts = {sd: [] for sd in seeds}
for rnd in range(4):
    for sd, d in ds.items():
        ts[sd].append(bench.time_graph(lambda: submission.kernel_fn(d, st), 20) - bench.empty_cost())
for sd, d in ds.items():
    c = torch.bincount(d["topk_ids"][d["topk_ids"] < 32].long(), minlength=32)
    print(f"bs {bs} seed {sd} experts {int((c>0).sum())} top {sorted(c.tolist(), reverse=True)[:4]} t {statistics.median(ts[sd]):.1f} (spread {min(ts[sd]):.1f}-{max(ts[sd]):.1f})")
