"""Distinct local experts in the LAYER's own topk_ids vs the reference's, per seed. usage: layer_distinct.py <file> "<load> <layer>" seeds,..."""
import sys
sys.path.insert(0, ".")
import torch, bench, reference
sub = bench.load_sub(sys.argv[1]); flt = sys.argv[2]; seeds = [int(x) for x in sys.argv[3].split(",")]
w, st = bench.weights_state(sub)
for l, c in bench.SCORED:
    if flt != f"{l} {c['layer']}":
        continue
    for sd in seeds:
        data = bench.make_input(c, w, sd); ls = int(data["local_start"])
        out = sub.kernel_fn(data, st); torch.cuda.synchronize()
        ids = out[2].long()
        ref = reference.ref_front(data, w)["biased"].topk(reference.TOP_K, dim=-1).indices
        print(flt, f"s{sd}", "layer", reference._local_load(ids, ls), "ref", reference._local_load(ref.long(), ls), flush=True)
