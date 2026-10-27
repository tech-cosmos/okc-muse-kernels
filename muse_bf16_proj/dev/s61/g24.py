# s61: s60 harness (dev/s61/gemv24.cu). s60: s59 harness for the s60 gemv24 (zo arg; r = 513 = WG wgmma mode). s53 gemv24 (per-warp TMA rings) sweep: python dev/s53/g24.py "M,KC,AP,NS,WPB,XB;..."  env SH=qkv,o,pool,unp,
# GRID (default #SMs), PRED (ns; predecessor writes x and triggers PDL at start, as g19.py MODE 2), NPRE, DEFS (G24_NOW).
# Times = graph(pred + gemm) - graph(pred), cold L2 (bench.time_graph). Check: vs x @ W'^T, W' = W with escaped weights
# set to 2^-15 * scale (the prototype has no escape fixups); skipped for G24_NOW.
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
cfgs = [c + (1,)[len(c) - 6:] for c in cfgs]  # R (row groups per task) 1 by default
DEFS = [d_ for d_ in os.environ.get("DEFS", "").split(",") if d_]
src = "".join("#define %s\n" % d_.replace("=", " ") for d_ in DEFS) + open(os.path.join(D, "gemv24.cu")).read()
src = src.replace("G24_CONFIGS", " ".join("G24_CASE(%d,%d,%d,%d,%d,%d,%d)" % c for c in sorted(set(cfgs))))
mod = load_inline(name="pj_g24_" + hashlib.md5(src.encode()).hexdigest()[:10],
                  cpp_sources="torch::Tensor g24_clk(); torch::Tensor g24_trace(); void g24_gemm(torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, torch::Tensor); void g24_pred(torch::Tensor, torch::Tensor, int64_t);",
                  cuda_sources=src, functions=["g24_clk", "g24_trace", "g24_gemm", "g24_pred"],
                  extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"] + (["-Xptxas", "-v"] if os.environ.get("PTXV") else []))


def pack24(w, kc, ap, r):
    """stage (ks, tile of r row groups, st) = [r][ap atoms][32 lanes x 16 B sign|mantissa] [r][ap][32 x 8 B codes], DEC3."""
    S, N, hdr, scale = pack16(w, kc)
    n, k = w.shape
    ks, ntr, spk = k // kc, n // (16 * r), kc // 32 // ap
    S = S.reshape(ks, ntr, r, spk, ap * 512).permute(0, 1, 3, 2, 4).reshape(ks * ntr * spk, r * ap * 512)
    N = N.view(torch.uint8).reshape(ks, ntr, r, spk, ap * 256).permute(0, 1, 3, 2, 4).reshape(ks * ntr * spk, r * ap * 256)
    P = dec3.tr3_packed(torch.cat([S, N], 1).contiguous(), r * ap * 512)
    return P, hdr, scale


SH = [("qkv", "qkv_gate", 2304), ("o", "o", 8192), ("pool", "pooler", 4096), ("unp", "unpooler", 8192)]
only = os.environ.get("SH")
GRID = int(os.environ.get("GRID", torch.cuda.get_device_properties(0).multi_processor_count))
PRED = int(os.environ.get("PRED", 3000))
NPRE = int(os.environ.get("NPRE", 99))
EZB = (int(os.environ.get("EZ", 0)) << 17) | (int(os.environ.get("XSP", 0)) << 19) | (int(os.environ.get("DF", 0)) << 20)  # s61: EZ=1 sets npre bit 17 (out assumed zeroed: timing probe, no check)
NOW = "G24_NOW" in DEFS or "G24_NOZ" in DEFS or (EZB >> 17) & 1
NOFIX = "G24_NOFIX" in DEFS
Wr = reference.generate_weights(reference.WEIGHTS_SEED)
ESCW = bool(os.environ.get("ESCW"))  # adversarial escapes (as dev/g19.py ESCW) + exact spike rows at them, 20 reps
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
FL = torch.zeros(64 << 20, device="cuda") if "G24_TRACE" in os.environ.get("DEFS", "") else None
ctr = torch.zeros(4, dtype=torch.int32, device="cuda")
ZO = torch.empty(0, device="cuda")
for nm, layer, n in SH:
    if only and nm not in only.split(","): continue
    W = Wr[layer][:n].contiguous()
    k = W.shape[1]
    packs = {}
    for M in sorted({c[0] for c in cfgs}):
        x = torch.randn(M, k, device="cuda").bfloat16()
        xsrc = x.clone()
        pre = (lambda: mod.g24_pred(xsrc, x, PRED)) if PRED >= 0 else (lambda: None)  # PRED<0: no predecessor (graph's first op)
        e = bench.time_graph(pre) if PRED >= 0 else 0.0
        tc = bench.time_graph(lambda: (pre(), F.linear(x, W))) - e
        line = f"{nm:5s} m{M}: pred {e:6.2f} cublas {tc:6.2f}"
        out = torch.empty(M, n, device="cuda")
        for (m_, kc, ap, ns, wpb, xb, r) in cfgs:
            if m_ != M or k % kc: continue
            if (kc, ap, r & 0xff) not in packs:
                P, hdr, scale = pack24(W, kc, ap, r & 0xff)
                Wp = W.clone()
                er = (hdr[:, :, 1] != 2.0 ** -15).nonzero()
                if NOFIX and len(er):
                    cols = hdr[er[:, 0], er[:, 1], 0].view(torch.int32).long()
                    Wp[er[:, 0], cols] = torch.tensor(2.0 ** -15 * scale, dtype=torch.bfloat16, device="cuda")
                col = int(hdr[er[0, 0], er[0, 1], 0].view(torch.int32)) if len(er) else 3
                packs[(kc, ap, r & 0xff)] = (P, hdr, scale, Wp, col)
            P, hdr, scale, Wp_, col = packs[(kc, ap, r & 0xff)]
            xp_ = x.clone(); xp_[M - 1].zero_(); xp_[M - 1, col] = 16.0  # probe row: one escaped weight x 16, exact
            sp = list(range(0, M - 1, 2)) if ESCW else []
            for j_, r_ in enumerate(sp):
                xp_[r_].zero_(); xp_[r_, CE[layer][j_ % 32]] = 8.0
            ref = xp_.double() @ Wp_.double().T
            cf = (kc, ap, ns, wpb, xb, GRID, min(NPRE, ns) | EZB, r, ZO)
            try:
                for rep in range(20 if ESCW else 3):
                    out.fill_(1.0); mod.g24_gemm(P, hdr, scale, xp_, out, ctr, *cf); torch.cuda.synchronize()
                    err = ((out.double() - ref).norm() / ref.norm()).item()
                    if not NOW:
                        assert err < 1e-5, (err, nm, m_, cf)
                        pw_ = (Wp_[:, col].double() * 16).float()
                        assert NOFIX or torch.equal(out[M - 1], torch.where(pw_.abs() < 2.0 ** -126, 0 * pw_, pw_)), ("probe", nm, col)
                        for j_, r_ in enumerate(sp):
                            want = (Wp_[:, CE[layer][j_ % 32]].double() * 8).float()
                            want = torch.where(want.abs() < 2.0 ** -126, torch.zeros_like(want), want)
                            assert torch.equal(out[r_], want), ("spike", nm, r_, int((out[r_] != want).sum()))
            except RuntimeError as ex:
                line += f" | {kc},{ap},{ns},{wpb},{xb},r{r} ERR {str(ex).splitlines()[0][:60]}"; continue
            if ESCW: print("  esc ok", nm, M, cf, "rel err", err, flush=True)
            dt = bench.time_graph(lambda: (pre(), mod.g24_gemm(P, hdr, scale, x, out, ctr, *cf))) - e
            line += f" | {kc},{ap},{ns},{wpb},{xb},r{r} {dt:6.2f}"
            if "G24_CLK" in DEFS:
                pre(); mod.g24_gemm(P, hdr, scale, x, out, ctr, *cf); torch.cuda.synchronize()
                ck = mod.g24_clk().view(-1, 8)[:GRID * wpb].double().cpu()
                qq_ = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0], dtype=torch.float64)
                print(f"    clk {nm} m{M} setup q0/10/50/90/100 " + " ".join(f"{v/1e3:.2f}" for v in torch.quantile(ck[:, 6], qq_).tolist()) + " | pass@ " + " ".join(f"{v/1e3:.2f}" for v in torch.quantile(ck[:, 7], qq_).tolist()) + " | loop+end " + " ".join(f"{v/1e3:.2f}" for v in torch.quantile(ck[:, 0] + ck[:, 1], qq_).tolist()))
                for tot_ in sorted(set(ck[:, 3].long().tolist())):
                    sel = ck[ck[:, 3] == tot_]
                    print(f"    clk {nm} m{M} stages {tot_:3d} warps {len(sel):4d}: loop {sel[:, 0].median()/1e3:7.2f} end {sel[:, 1].median()/1e3:6.2f} xwait {sel[:, 2].median()/1e3:6.2f} (fix {sel[:, 4].median()/1e3:5.2f} bar {sel[:, 5].median()/1e3:5.2f}) setup {sel[:, 6].median()/1e3:5.2f} pass@ {sel[:, 7].median()/1e3:5.2f} kcyc (median)")
            if "G24_TRACE" in DEFS:
                ends = []
                for rep in range(5):
                    FL.add_(1)
                    pre(); mod.g24_gemm(P, hdr, scale, x, out, ctr, *cf); torch.cuda.synchronize()
                    tr = mod.g24_trace().view(-1, 3)[:GRID * wpb].cpu()
                    t0 = tr[:, 0].min(); e_ = (tr[:, 1] - t0).float() / 1000; b_ = (tr[:, 0] - t0).float() / 1000
                    nq_ = (tr[:, 2] & 0xffffffff)
                    q = torch.quantile(e_, torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0]))
                    ends.append(q)
                    bl = e_.view(GRID, wpb).max(1).values
                    print(f"    trace {nm} m{M} start spread {b_.max():.2f} end q0/10/50/90/100 " + " ".join(f"{v:.2f}" for v in q.tolist())
                          + f" | block-end min/med/max {bl.min():.2f}/{bl.median():.2f}/{bl.max():.2f} intra-block spread med {(bl - e_.view(GRID, wpb).min(1).values).median():.2f} nq {nq_.min()}-{nq_.max()}")
        print(line, flush=True)
    del W, packs
