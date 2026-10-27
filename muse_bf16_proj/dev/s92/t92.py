"""proj s92: layer g30 escape-fix cost. Times variants of g30 (CU=file, DEFS=a,b) on the reference weights (unpool T=128),
checks exactness vs a no-variant build. usage: CUS=g30_base.cu,g30v.cu DEFSETS='|G30_ESC 0|...' python t92.py [seed]"""
import os, sys
LD = os.path.expanduser("~/r-layer/muse_decode_layer")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LD); os.chdir(LD)
import torch, torch.nn.functional as F
import bench, reference
L = bench.load_sub("submission.py")
W = reference.generate_weights(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
keys = os.environ.get("KEYS", "unpool").split(",")
variants = []
for cu in os.environ.get("CUS", "g30_base.cu").split(","):
    for ds in os.environ.get("DEFSETS", "").split("|"):
        variants.append((cu, ds))
libs = {}
for cu, ds in variants:
    L._G30.clear()
    L._G30_SRC = "".join(f"#define {d}\n" for d in ds.split(",") if d) + open(os.path.join(HERE, cu)).read()
    L._g30_build()
    libs[(cu, ds)] = L._G30["lib"]
M = 128
res = {v: [] for v in variants}
for key in keys:
    w = W["unpooler"] if key == "unpool" else W[key]
    if os.environ.get("ESCW"):  # adversarial escapes (proj g30e.py ESCW): tiny/zero/subnormal at 32 cols, ~75% rows x3
        w = w.clone(); g_ = torch.Generator(device="cuda").manual_seed(5)
        ce = torch.randperm(w.shape[1], device="cuda", generator=g_)[:32]
        mx = w.float().abs().max().item()
        vals = torch.tensor([0.0, -0.0, 2.0 ** -133, -3 * 2.0 ** -131, 2.0 ** -100, -1.37 * 2.0 ** -60,
                             mx * 2.0 ** -17, -mx * 1.9 * 2.0 ** -18, 1.5 * 2.0 ** -126], device="cuda").bfloat16()
        for _ in range(int(os.environ.get("ESCW"))):
            rows = torch.arange(w.shape[0], device="cuda")[torch.rand(w.shape[0], device="cuda", generator=g_) < 0.75]
            w[rows, ce[torch.randint(0, 32, (len(rows),), device="cuda", generator=g_)]] = vals[torch.randint(0, len(vals), (len(rows),), device="cuda", generator=g_)]
    c = L._compress(w)
    if c is None: print(key, "compress failed (too many escapes)"); continue
    code, sm, hdr, scale = c
    print(key, "max esc/row", int((hdr[:, :, 1] != 2.0 ** -15).sum(1).max()))
    nesc = int((hdr[:, :, 1] != 2.0 ** -15).sum())
    wp = L._pack30(code, sm)
    x = torch.randn(M, w.shape[1], device="cuda").bfloat16()
    out = torch.empty(M, w.shape[0], device="cuda", dtype=torch.bfloat16)
    refo = None
    for v in variants:
        libs[v].g30_gemm(x, wp, hdr, out, scale, 0); torch.cuda.synchronize()
        if refo is None: refo = out.clone()
        print(f"{key} {v}: bitwise== first {torch.equal(out, refo)}  vs F.linear maxdiff {(out.float() - F.linear(x, w).float()).abs().max().item():.3g}", flush=True)
    e0 = bench.empty_cost()
    tc = bench.time_graph(lambda: F.linear(x, w)) - e0
    for rnd in range(int(os.environ.get("RND", 4))):
        for v in variants:
            res[v].append(bench.time_graph(lambda: libs[v].g30_gemm(x, wp, hdr, out, scale, 0)) - bench.empty_cost())
    print(f"{key} {tuple(w.shape)} esc {nesc} cuBLAS {tc:.2f}")
    for v in variants:
        print(f"  {v[0]:14s} [{v[1]}]: " + " ".join(f"{t:.2f}" for t in res[v]))
        res[v] = []
