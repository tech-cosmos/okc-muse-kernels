"""o26: bs32 TK2H_M weight A/B on chosen seeds. python tools/o26/ab32.py seed,seed,... W,W,... (W=-1: TK2H_M off)"""
import sys, os, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch, bench, reference, submission
seeds = [int(s) for s in sys.argv[1].split(",")]; Ws = [int(w) for w in sys.argv[2].split(",")]
_, st = bench.state()
for sd in seeds:
    data = reference.generate_input(bs=32, sharding="ep8", seed=sd)
    cnt = torch.bincount(data["topk_ids"][data["topk_ids"] < 32].long(), minlength=32)
    out = []
    for rnd in range(3):
        for W in Ws:
            submission.TK2H_M = {} if W < 0 else {32: W}
            out.append((W, bench.time_graph(lambda: submission.kernel_fn(data, st), 20)))
    med = {W: statistics.median(t for w, t in out if w == W) for W in Ws}
    print(f"seed {sd} experts {int((cnt>0).sum())} cnt>8 {sorted([int(c) for c in cnt if c > 8], reverse=True)}: " + "  ".join(f"W{W} {t:.1f}" for W, t in med.items()), flush=True)
