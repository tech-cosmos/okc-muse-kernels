# run from ~/r-layer/muse_decode_layer: dumps the layer's local routing (ids in [0,32) or -1) per scored shape/seed
import sys, torch
sys.path.insert(0, ".")
import reference, bench
w = reference.generate_weights(reference.WEIGHTS_SEED)
out = {}
for load, c in bench.SCORED:
    if c["layer"] != "sliding": continue
    for seed in (99, 7, 11, 13, 17):
        d = bench.make_input(c, w, seed)
        fr = reference.ref_front(d, w)
        tv, ti = fr["biased"].topk(8, dim=-1)
        loc = (ti.long() - fr["local_start"]) % reference.N_EXPERTS
        loc = torch.where(loc < 32, loc, torch.full_like(loc, -1)).int()
        out[(load, seed)] = loc.cpu()
        print(load, seed, ti.shape[0], (loc >= 0).sum().item(), flush=True)
torch.save(out, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "data", "routes.pt"))
