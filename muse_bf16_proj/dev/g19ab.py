# Interleaved A/B of two gemv19 builds (DEFS_A vs DEFS_B), after a PDL predecessor (MODE 2, like the layer):
# python dev/g19ab.py "M,KC,AP,NS[,XB];..."  env DEFS_A, DEFS_B, GRID (72 ~ the layer's 108 blocks on H200), SH, R (rounds),
# PRED (ns, 3000). Prints per (shape, cfg) the median over R rounds of each build and B - A. Correctness: 2 reps vs fp64.
import os, sys, torch, hashlib, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.cpp_extension import load_inline
import bench, reference
d = os.path.dirname(os.path.abspath(__file__))
_g18 = open(os.path.join(d, "g18.py")).read()
exec(_g18[_g18.index("ESC = 8"):_g18.index("cfgs = ")])
exec(_g18[_g18.index("def pack16"):_g18.index("SH = [")].replace('"G18_ILV" in', '"G18_NEVER" in'))
cfgs = [tuple(map(int, c.split(","))) for c in sys.argv[1].split(";")]
cfgs = [c + (1,)[len(c) - 4:] for c in cfgs]
CASES = " ".join("G19_CASE(%d,%d,%d,%d,1,%d,1,8,1)" % c for c in sorted(set(cfgs)))
# env SRC_A / SRC_B: source file per build (default dev/gemv19.cu for both)
srcs = {b: open(os.environ.get("SRC_" + b, os.path.join(d, "gemv19.cu"))).read().replace("G19_CONFIGS", CASES) for b in "AB"}
CPP = "void g19_gemm(torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t); void g19_pred(torch::Tensor, torch::Tensor, int64_t);"
def build(defs, base):
    src = "".join("#define %s\n" % x.replace("=", " ") for x in defs.split(",") if x) + base
    return load_inline(name="pj_g19ab_" + hashlib.md5(src.encode()).hexdigest()[:10], cpp_sources=CPP, cuda_sources=src,
                       functions=["g19_gemm", "g19_pred"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])
mods = {b: build(os.environ.get("DEFS_" + b, ""), srcs[b]) for b in "AB"}
def pack19r(w, kc, ap, rg=8):
    S, N, hdr, scale = pack16(w, kc)
    n, k = w.shape
    ks, ntr, spk = k // kc, n // (16 * rg), kc // 32 // ap
    S = S.view(ks, ntr, rg, spk, ap * 512).permute(0, 1, 3, 2, 4).reshape(ks * ntr * spk, rg * ap * 512)
    N = N.view(torch.uint8).view(ks, ntr, rg, spk, ap * 256).permute(0, 1, 3, 2, 4).reshape(ks * ntr * spk, rg * ap * 256)
    return torch.cat([S, N], 1).contiguous(), hdr, scale
SH = [("qkv", "qkv_gate"), ("o", "o"), ("pool", "pooler"), ("unp", "unpooler")]
only = os.environ.get("SH", "pool,unp")
GRID = int(os.environ.get("GRID", 72)); R = int(os.environ.get("R", 5)); PRED = int(os.environ.get("PRED", 3000))
Wr = reference.generate_weights(reference.WEIGHTS_SEED)
ctr = torch.zeros(4, dtype=torch.int32, device="cuda")
for nm, layer in SH:
    if nm not in only.split(","): continue
    W = Wr[layer].contiguous(); n, k = W.shape
    for (M, kc, ap, ns, xb) in cfgs:
        P, hdr, scale = pack19r(W, kc, ap)
        x = torch.randn(M, k, device="cuda").bfloat16(); xs = x.clone()
        out = torch.empty(M, n, device="cuda")
        ref = (x.double() @ W.double().T)
        run = {key: (lambda m_=m_: (m_.g19_pred(xs, x, PRED), m_.g19_gemm(P, hdr, scale, x, out, ctr, kc, ap, ns, 1, GRID, 0, 2, ns, xb, 1, 8, 1)))
               for key, m_ in mods.items()}
        for key in run:
            for _ in range(2):
                out.fill_(1.0); run[key](); torch.cuda.synchronize()
                err = ((out.double() - ref).norm() / ref.norm()).item()
                assert err < 1e-5, (key, nm, M, err)
        e = bench.time_graph(lambda: mods["A"].g19_pred(xs, x, PRED))
        t = {"A": [], "B": []}
        for r in range(R):
            for key in ("A", "B") if r % 2 == 0 else ("B", "A"):
                t[key].append(bench.time_graph(run[key]) - e)
        a, b = statistics.median(t["A"]), statistics.median(t["B"])
        print(f"{nm:5s} m{M} {kc},{ap},{ns},x{xb}: A {a:6.2f}  B {b:6.2f}  B-A {b - a:+.2f}   (A {min(t['A']):.2f}-{max(t['A']):.2f}, B {min(t['B']):.2f}-{max(t['B']):.2f})", flush=True)
    del W
