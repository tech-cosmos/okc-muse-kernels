"""Per scored shape: this GPU's routes per local expert in the layer (reference front + route, local ids)."""
import sys, collections, torch
sys.path.insert(0, ".")
import reference, bench
w = reference.generate_weights(reference.WEIGHTS_SEED)
for load, c in bench.SCORED:
    if c["layer"] != "sliding":
        continue
    for seed in (99, 7, 11):
        d = bench.make_input(c, w, seed)
        fr = reference.ref_front(d, w)
        ids = fr["biased"].topk(8, dim=-1).indices
        loc = (ids.long() - fr["local_start"]) % reference.N_EXPERTS
        loc = loc[loc < 32]
        cnt = sorted(collections.Counter(loc.tolist()).values(), reverse=True)
        print(f"{load:10s} seed {seed}: T={ids.shape[0]:3d} routes {sum(cnt):4d} experts {len(cnt):2d} per expert {cnt}")
