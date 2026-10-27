# s57 gemv27 sweep: python dev/s57/g27.py "M,KQ,AP,NS,RGB[,OBF];..."  env SH=o,unp,... PRED (ns; <0 = no predecessor,
# graph's first op), ESCW=1 adversarial escapes (as g24.py), G24="M,KC,AP,NS,WPB,XB,R;..." g24 configs for comparison.
# Times = graph(pred + gemm) - graph(pred), cold L2 (bench.time_graph).
import os, sys, torch, hashlib, torch.nn.functional as F
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(D)))
sys.path.insert(0, os.path.join(os.path.dirname(D), "s35"))
from torch.utils.cpp_extension import load_inline
import bench, reference, dec3
_g18 = open(os.path.join(os.path.dirname(D), "g18.py")).read()
exec(_g18[_g18.index("def pack16"):_g18.index("def pack18")])
ESC = 8
cfgs = [tuple(map(int, c.split(","))) for c in sys.argv[1].split(";")]
cfgs = [c + (0,)[len(c) - 5:] for c in cfgs]
g24c = [tuple(map(int, c.split(","))) for c in os.environ.get("G24", "").split(";") if c]
DEFS = [d_ for d_ in os.environ.get("DEFS", "").split(",") if d_]
src = "".join("#define %s\n" % d_.replace("=", " ") for d_ in DEFS) + open(os.path.join(D, "gemv27.cu")).read()
src = src.replace("G27_CONFIGS", " ".join("G27_CASE(%d,%d,%d,%d,%d,%s)" % (c[:5] + ("true" if c[5] else "false",)) for c in sorted(set(cfgs))))
mod = load_inline(name="pj_g27_" + hashlib.md5(src.encode()).hexdigest()[:10],
                  cpp_sources="torch::Tensor g27_trace(); void g27_gemm(torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, torch::Tensor); void g27_pred(torch::Tensor, torch::Tensor, int64_t);",
                  cuda_sources=src, functions=["g27_gemm", "g27_pred", "g27_trace"],
                  extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"] + (["-Xptxas", "-v"] if os.environ.get("PTXV") else []))
if g24c:
    s24 = open(os.path.join(os.path.dirname(D), "s53", "gemv24.cu")).read().replace("G24_CONFIGS", " ".join("G24_CASE(%d,%d,%d,%d,%d,%d,%d)" % c for c in sorted(set(g24c))))
    m24 = load_inline(name="pj_g24c_" + hashlib.md5(s24.encode()).hexdigest()[:10],
                      cpp_sources="torch::Tensor g24_trace(); void g24_gemm(torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, torch::Tensor); void g24_pred(torch::Tensor, torch::Tensor, int64_t);",
                      cuda_sources=s24, functions=["g24_gemm"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])


def pack24(w, kc, ap, r=1):
    S, N, hdr, scale = pack16(w, kc)
    n, k = w.shape
    ks, ntr, spk = k // kc, n // (16 * r), kc // 32 // ap
    S = S.reshape(ks, ntr, r, spk, ap * 512).permute(0, 1, 3, 2, 4).reshape(ks * ntr * spk, r * ap * 512)
    N = N.view(torch.uint8).reshape(ks, ntr, r, spk, ap * 256).permute(0, 1, 3, 2, 4).reshape(ks * ntr * spk, r * ap * 256)
    P = dec3.tr3_packed(torch.cat([S, N], 1).contiguous(), r * ap * 512)
    return P, hdr, scale


SH = [("qkv", "qkv_gate", 2304), ("o", "o", 8192), ("pool", "pooler", 4096), ("unp", "unpooler", 8192)]
only = os.environ.get("SH")
PRED = int(os.environ.get("PRED", 3000))
NPRE = int(os.environ.get("NPRE", 99))
Wr = reference.generate_weights(reference.WEIGHTS_SEED)
ESCW = bool(os.environ.get("ESCW"))
CE = {}
if ESCW:
    g_ = torch.Generator(device="cuda").manual_seed(7)
    for name in ["qkv_gate", "o", "pooler", "unpooler"]:
        t = Wr[name]; n_, k_ = t.shape; mx = t.float().abs().max().item()
        vals = torch.tensor([0.0, -0.0, 2.0 ** -133, -3 * 2.0 ** -131, 2.0 ** -100, -1.37 * 2.0 ** -60,
                             mx * 2.0 ** -17, -mx * 1.9 * 2.0 ** -18, 1.5 * 2.0 ** -126], device="cuda").to(torch.bfloat16)
        CE[name] = torch.randperm(k_, device="cuda", generator=g_)[:32]
        for j in range(2 if k_ == 8192 else 3):
            rows = torch.arange(n_, device="cuda")[torch.rand(n_, device="cuda", generator=g_) < 0.75]
            cols = CE[name][torch.randint(0, 32, (len(rows),), device="cuda", generator=g_)]
            t[rows, cols] = vals[torch.randint(0, len(vals), (len(rows),), device="cuda", generator=g_)]
ctr = torch.zeros(4, dtype=torch.int32, device="cuda")
NOZ = torch.empty(0, device="cuda")
for nm, layer, n in SH:
    if only and nm not in only.split(","): continue
    W = Wr[layer][:n].contiguous()
    n, k = W.shape
    packs = {}
    for M in sorted({c[0] for c in cfgs}):
        x = torch.randn(M, k, device="cuda").bfloat16()
        if ESCW:
            for j_, r_ in enumerate(range(0, M - 1, 2)):
                x[r_].zero_(); x[r_, CE[layer][j_ % 32]] = 8.0
        xsrc = x.clone()
        pre = (lambda: mod.g27_pred(xsrc, x, PRED)) if PRED >= 0 else (lambda: None)
        e = bench.time_graph(pre) if PRED >= 0 else 0.0
        tc = bench.time_graph(lambda: (pre(), F.linear(x, W))) - e
        line = f"{nm:5s} m{M}: pred {e:6.2f} cublas {tc:6.2f}"
        ref = x.double() @ W.double().T
        for c24 in g24c:
            if c24[0] != M or k % c24[1]: continue
            P, hdr, scale = pack24(W, c24[1], c24[2], c24[6] & 0xff)
            o24 = torch.empty(M, n, device="cuda")
            cf = c24[1:6] + (132, min(NPRE, c24[3]), c24[6])
            m24.g24_gemm(P, hdr, scale, x, o24, ctr, *cf)
            dt = bench.time_graph(lambda: (pre(), m24.g24_gemm(P, hdr, scale, x, o24, ctr, *cf))) - e
            line += f" | g24 {c24[1:]} {dt:6.2f}"
        for (m_, kq, ap, ns, rgb, obf) in cfgs:
            if m_ != M or k % (kq * 32 * ap): continue
            if k % (kq * 32 * ap) or (ns >= 100 and (ns - 100) * 32 * kq != k): continue
            if (kq, ap) not in packs:
                packs[(kq, ap)] = pack24(W, k // kq, ap)
            P, hdr, scale = packs[(kq, ap)]
            out = torch.empty(M, n, device="cuda", dtype=torch.bfloat16 if obf else torch.float32)
            cf = (kq, ap, ns, rgb, min(NPRE, ns) | 0)
            tag = f"{kq},{ap},{ns},{rgb}{',bf' if obf else ''}"
            if "G27_XREP" in DEFS:  # timing probe: x replicated per block (written by the pred too: L2-resident)
                xr = x.repeat(n // (16 * rgb), 1).contiguous(); xrs = xr.clone()
                pre = (lambda: mod.g27_pred(xrs, xr, PRED)); e = bench.time_graph(pre)
                dt = bench.time_graph(lambda: (pre(), mod.g27_gemm(P, hdr, scale, xr[:M], out, *cf, NOZ))) - e
                line += f" | {tag} XREP {dt:6.2f}"; continue
            try:
                for rep in range(0 if os.environ.get("NOCHK") else 10 if ESCW else 2):
                    out.fill_(1.0); mod.g27_gemm(P, hdr, scale, x, out, *cf, NOZ); torch.cuda.synchronize()
                    if obf:
                        assert torch.equal(out, ref.float().bfloat16()) or ((out.double() - ref).norm() / ref.norm()).item() < 4e-3, tag
                    else:
                        err = ((out.double() - ref).norm() / ref.norm()).item()
                        assert err < 1e-5, (err, nm, tag)
                        for j_, r_ in enumerate(range(0, M - 1, 2) if ESCW else []):
                            want = (W[:, CE[layer][j_ % 32]].double() * 8).float()
                            want = torch.where(want.abs() < 2.0 ** -126, torch.zeros_like(want), want)
                            if not torch.equal(out[r_], want) and not torch.equal(out[r_], (W[:, CE[layer][j_ % 32]].double() * 8).float()):
                                bad = (out[r_] != want).nonzero()[:4, 0]
                                raw = (W[:, CE[layer][j_ % 32]].double() * 8).float()
                                print("   spike", nm, r_, int((out[r_] != want).sum()), "out", out[r_][bad].tolist(), "want", want[bad].tolist(), "raw", raw[bad].tolist(), "eq raw", torch.equal(out[r_], raw))
                                raise AssertionError("spike")
            except (RuntimeError, AssertionError) as ex:
                line += f" | {tag} ERR {str(ex).splitlines()[0][:70]}"; continue
            dt = bench.time_graph(lambda: (pre(), mod.g27_gemm(P, hdr, scale, x, out, *cf, NOZ))) - e
            line += f" | {tag} {dt:6.2f}"
            if "G27_TRACE" in DEFS:
                nb = n // (16 * rgb)
                for rep in range(3):
                    torch.sum(bench._FLUSH.view(torch.int64), 0, out=bench._SINK)
                    pre(); mod.g27_gemm(P, hdr, scale, x, out, *cf, NOZ); torch.cuda.synchronize()
                    tr = mod.g27_trace().view(-1, 6)[:nb].cpu().double()
                    t0 = tr[:, 1].min()
                    q = lambda col: "/".join(f"{v:.2f}" for v in torch.quantile((tr[:, col] - t0) / 1000, torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)).tolist())
                    print(f"   trace {tag}: start {q(0)} wait {q(1)} x {q(2)} comp {q(3)} sync {q(4)} end {q(5)}")
        print(line, flush=True)
    del W, packs
