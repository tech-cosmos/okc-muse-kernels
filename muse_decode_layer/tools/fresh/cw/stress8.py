"""fresh x13 CW stress at T=8: the layer's nu8 expert path (_experts_nu8, NU8D y32 [81, 4096]) with TK2F_CW8 on vs off,
bitwise on y32 rows 0..63 (tk2f's output; nu8's rows 64..80 are atomics, close only), on random routings (0..8 local experts per token, distinct per token), eager and in
CUDA-graph replay. python tools/fresh/cw/stress8.py [n] (run from muse_decode_layer/)"""
import sys, torch
sys.path.insert(0, ".")
import bench
sub = bench.load_sub("submission.py")
w, s = bench.weights_state(sub)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
g = torch.Generator(device="cuda").manual_seed(13)
dev = "cuda"


def rnd(r):
    x = torch.randn(8, 4096, device=dev, generator=g).bfloat16()
    ids = torch.full((8, 8), -1, dtype=torch.int32, device=dev)
    mode = r % 4  # 0: uniform 0..8 local per token, 1: all 8 local (max segments), 2: few hot experts, 3: none / one token
    for t in range(8):
        nloc = int(torch.randint(0, 9, (1,), generator=g, device=dev)) if mode == 0 else 8 if mode == 1 else 4 if mode == 2 else (8 if t == r % 8 else 0)
        pool = torch.randperm(32 if mode != 2 else 10, device=dev, generator=g)[:nloc].int()
        sl = torch.randperm(8, device=dev, generator=g)[:nloc]
        ids[t, sl] = pool
    tw = torch.rand(8, 8, device=dev, generator=g)
    return x, tw, ids.contiguous()


def run(cw, x, tw, ids, y32):
    sub.TK2F_CW8 = cw
    y32.zero_()
    f, sc = sub._experts_nu8(x, tw, ids, s, y32)
    return f, y32


def same(a, b):  # y32 rows 0..63 = moe_tk2f's output (bitwise); rows 64..80 (nu8's atomics) differ even base vs base
    return torch.equal(a[1][:64], b[1][:64]) and torch.allclose(a[0], b[0], rtol=1e-2, atol=1e-2 * a[0].abs().max().item())


y32a = torch.zeros(81, 4096, device=dev); y32b = torch.zeros(81, 4096, device=dev)
bad = 0
for r in range(n):
    x, tw, ids = rnd(r)
    a = [t.clone() for t in run(True, x, tw, ids, y32a)]; b = [t.clone() for t in run(False, x, tw, ids, y32b)]
    torch.cuda.synchronize()
    if not same(a, b):
        bad += 1; print("EAGER MISMATCH", r, (a[1] - b[1]).abs().max().item(), flush=True)
print("eager bad", bad, "of", n, flush=True)
xs, tws, idss = rnd(1)
outs, graphs, ys = {}, {}, {True: y32a, False: y32b}
for cw in (True, False):
    run(cw, xs, tws, idss, ys[cw])
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        outs[cw] = run(cw, xs, tws, idss, ys[cw])
    graphs[cw] = gr
gbad = 0
for r in range(n):
    x, tw, ids = rnd(r + 1000)
    res = {}
    for cw in (True, False):
        xs.copy_(x); tws.copy_(tw); idss.copy_(ids)
        graphs[cw].replay()
        res[cw] = [t.clone() for t in outs[cw]]
    torch.cuda.synchronize()
    if not same(res[True], res[False]):
        gbad += 1; print("REPLAY MISMATCH", r, flush=True)
print("replay bad", gbad, "of", n)
sub.TK2F_CW8 = True
