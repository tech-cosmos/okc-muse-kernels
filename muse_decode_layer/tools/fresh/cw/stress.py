"""fresh x12 CW stress: the layer's T=1 expert path (_experts_n2u) with TK2F_CW on vs off, bitwise, on many random
routings (0..8 local experts, duplicates impossible as in top-k), eager and in CUDA-graph replay (one capture per mode,
fresh inputs copied in before each replay). python tools/fresh/cw/stress.py [n] (run from muse_decode_layer/)"""
import sys, torch
sys.path.insert(0, ".")
import bench
sub = bench.load_sub("submission.py")
w, s = bench.weights_state(sub)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
g = torch.Generator(device="cuda").manual_seed(12)
dev = "cuda"


def rnd(r):
    x = torch.randn(1, 4096, device=dev, generator=g).bfloat16()
    nloc = r % 9  # 0..8 local experts
    perm = torch.randperm(64, device=dev, generator=g)[:8].int()
    loc = perm % 32
    ids = torch.where(torch.arange(8, device=dev) < nloc, loc, torch.full_like(loc, -1)).view(1, 8)
    # local expert ids must be distinct (top-k): use the first nloc of a permutation of 0..31
    ids[0, :nloc] = torch.randperm(32, device=dev, generator=g)[:nloc].int()
    tw = torch.rand(1, 8, device=dev, generator=g)
    return x, tw, ids.int().contiguous()


def run(cw, x, tw, ids):
    sub.TK2F_CW = cw
    return sub._experts_n2u(x, tw, ids, s).clone()


bad = 0
for r in range(n):
    x, tw, ids = rnd(r)
    a = run(True, x, tw, ids); b = run(False, x, tw, ids)
    torch.cuda.synchronize()
    if not torch.equal(a, b):
        bad += 1; print("EAGER MISMATCH", r, ids.tolist(), (a.float() - b.float()).abs().max().item())
print("eager bad", bad, "of", n, flush=True)
# graph replay
xs, tws, idss = rnd(1)
outs, graphs = {}, {}
for cw in (True, False):
    run(cw, xs, tws, idss)  # warm
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        sub.TK2F_CW = cw
        outs[cw] = sub._experts_n2u(xs, tws, idss, s)
    graphs[cw] = gr
gbad = 0
for r in range(n):
    x, tw, ids = rnd(r + 1000)
    res = {}
    for cw in (True, False):
        xs.copy_(x); tws.copy_(tw); idss.copy_(ids)
        graphs[cw].replay()
        res[cw] = outs[cw].clone()
    torch.cuda.synchronize()
    if not torch.equal(res[True], res[False]):
        gbad += 1; print("REPLAY MISMATCH", r, ids.tolist())
print("replay bad", gbad, "of", n)
sub.TK2F_CW = True
