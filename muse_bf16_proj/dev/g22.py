# gemv22 (m = 128/256 12-bit wgmma GEMM) test + timing: python dev/g22.py "M,AP,NS;..."  env SH=qkv,o,pool,unp, GRID,
# PRED=<ns> (a predecessor kernel that writes x and triggers PDL at its start; then MODE 1 = out zeroed outside, npre),
# ESCW=1 (adversarial escapes, 20 reps), NPRE. H100 graph replay, cold L2; cuBLAS F.linear for reference.
import os, sys, torch, torch.nn.functional as F, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.cpp_extension import load_inline
import bench, reference
d = os.path.dirname(os.path.abspath(__file__))
_g18 = open(os.path.join(d, "g18.py")).read()
exec(_g18[_g18.index("ESC = 8"):_g18.index("cfgs = ")])
exec(_g18[_g18.index("def pack16"):_g18.index("SH = [")].replace('"G18_ILV" in', '"G18_NEVER" in'))
cfgs = [tuple(map(int, c.split(","))) for c in sys.argv[1].split(";")]
src = open(os.path.join(d, "gemv22.cu")).read()
src = "".join("#define %s\n" % d_.replace("=", " ") for d_ in os.environ.get("DEFS", "").split(",") if d_) + src
cfgs = [c + (128 if os.environ.get("ESCW") else 32,) if len(c) == 3 else c for c in cfgs]
cfgs = [c + (int(os.environ.get("CL", 1)),) if len(c) == 4 else c for c in cfgs]
src = src.replace("G22_CONFIGS", " ".join("G22_CASE(%d,%d,%d,%d,%d)" % c for c in sorted(set(cfgs))))
src += "\nvoid g22_pred(torch::Tensor src, torch::Tensor dst, int64_t ns_);\n"
g20src = open(os.path.join(d, "gemv20.cu")).read()
pred = g20src[g20src.index("// test predecessor"):g20src.index("// diagnostics: a bare")]
src = src.replace("}  // namespace g22", pred + "}  // namespace g22", 1)
src += "void g22_pred(torch::Tensor src, torch::Tensor dst, int64_t ns_) { g22::pred<<<32, 256, 0, c10::cuda::getCurrentCUDAStream()>>>((const uint4*)src.data_ptr(), (uint4*)dst.data_ptr(), src.numel() * src.element_size() / 16, ns_); }\n"
TRC = "G22_TRACE" in os.environ.get("DEFS", "")
mod = load_inline(name="pj_g22_" + hashlib.md5(src.encode()).hexdigest()[:10],
                  cpp_sources="void g22_gemm(torch::Tensor, double, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t); void g22_pred(torch::Tensor, torch::Tensor, int64_t);" + (" torch::Tensor g22_trace();" if TRC else ""),
                  cuda_sources=src, functions=["g22_gemm", "g22_pred"] + (["g22_trace"] if TRC else []), extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo", "-Xptxas=-v"], verbose=bool(os.environ.get("VERB")))


def pack22(w, ap, E=32):
    """tile-major stages u = tile * S + ks, each [8 row groups x ap atoms x 32 lanes x 16 B][8 x ap x 32 x 8 B] then E
    escapes int32 (row in tile | k in stage << 8, true / scale bits) padded with (-1, 0). None if a stage has > E."""
    kc = 32 * ap
    S_, N_, hdr, scale = pack16(w, kc)
    n, k = w.shape
    ks, ntr = k // kc, n // 128
    S_ = S_.view(ks, ntr, 8, ap * 512)
    N_ = N_.view(torch.uint8).view(ks, ntr, 8, ap * 256)
    P = torch.cat([S_.reshape(ks, ntr, -1), N_.reshape(ks, ntr, -1)], 2).transpose(0, 1).contiguous().view(ks * ntr, -1)
    EV = torch.full((ks * ntr, E, 2), -1, dtype=torch.int32, device=w.device)
    EV[:, :, 1] = 0
    rows, slots = (hdr[:, :, 1] != 2.0 ** -15).nonzero(as_tuple=True)
    if len(rows):
        col = hdr[rows, slots, 0].view(torch.int32).long()
        u = (rows // 128) * ks + col // kc
        o = torch.argsort(u * (1 << 20) + rows * 8 + slots)
        rows, col, u, hy = rows[o], col[o], u[o], hdr[rows[o], slots[o], 1]
        rank = torch.arange(len(u), device=w.device) - torch.searchsorted(u, u)
        if int(rank.max()) >= E: return None
        EV[u, rank, 0] = ((rows % 128) | ((col % kc) << 8)).int()
        EV[u, rank, 1] = hy.view(torch.int32)
    return torch.cat([P, EV.view(torch.uint8).view(ks * ntr, -1)], 1).contiguous(), hdr, scale


SH = [("qkv", "qkv_gate", 2304), ("o", "o", 8192), ("pool", "pooler", 4096), ("unp", "unpooler", 8192)]
only = os.environ.get("SH")
GRID = int(os.environ.get("GRID", torch.cuda.get_device_properties(0).multi_processor_count))
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
e = bench.empty_cost()
PRED = int(os.environ.get("PRED", -1))
MODE = 1 if PRED >= 0 else 0
NPRE = int(os.environ.get("NPRE", 99))
for nm, layer, n in SH:
    if only and nm not in only.split(","): continue
    W = Wr[layer][:n].contiguous()
    k = W.shape[1]
    packs = {}
    for M in sorted({c[0] for c in cfgs}):
        x = torch.randn(M, k, device="cuda").bfloat16()
        xsrc = x.clone()
        pre = (lambda: mod.g22_pred(xsrc, x, PRED)) if PRED >= 0 else (lambda: None)
        ep = bench.time_graph(pre) if PRED >= 0 else e
        tc = bench.time_graph(lambda: (pre(), F.linear(x, W))) - ep
        line = f"{nm:5s} m{M}: cublas {tc:6.2f}"
        out = torch.empty(M, n, device="cuda")
        for (m_, ap, ns, E, CLv) in cfgs:
            if m_ != M: continue
            if (ap, E) not in packs: packs[(ap, E)] = pack22(W, ap, E)
            if packs[(ap, E)] is None: line += f" | {ap},{ns},{E} overflow"; continue
            P, hdr, scale = packs[(ap, E)]
            er = (hdr[:, :, 1] != 2.0 ** -15).nonzero()
            col = int(hdr[er[0, 0], er[0, 1], 0].view(torch.int32)) if len(er) else 3
            xp_ = x.clone(); xp_[M - 1].zero_(); xp_[M - 1, col] = 16.0
            sp = []
            if ESCW:
                sp = list(range(0, M - 1, 2))
                for j, r_ in enumerate(sp):
                    xp_[r_].zero_(); xp_[r_, CE[layer][j % 32]] = 8.0
            refp = xp_.double() @ W.double().T
            for rep in range(0 if os.environ.get("SKIPCHK") else 20 if ESCW else 2):
                out.fill_(1.0)
                mod.g22_gemm(P, scale, xp_, out, ap, ns, E, CLv, GRID, 0, 0); torch.cuda.synchronize()
                err = ((out.double() - refp).norm() / refp.norm()).item()
                assert err < 1e-5, (err, nm, m_, ap, ns)
                pw_ = (W[:, col].double() * 16).float()
                assert torch.allclose(out[M - 1], pw_, rtol=2 ** -22, atol=2.0 ** -126), ("probe", nm, col)
                for j, r_ in enumerate(sp):
                    want = (W[:, CE[layer][j % 32]].double() * 8).float()
                    want = torch.where(want.abs() < 2.0 ** -126, torch.zeros_like(want) * want.sign(), want)
                    if not torch.allclose(out[r_], want, rtol=2 ** -22, atol=2.0 ** -126):
                        bad = (out[r_] != want).nonzero().flatten()[:6]
                        print("SPIKE MISMATCH", nm, M, r_, [(int(i), float(out[r_, i]), float(want[i])) for i in bad]); raise SystemExit(1)
            if ESCW: print("  esc ok", nm, M, (ap, ns), "max rel err", err, flush=True)
            if TRC:
                fl = torch.empty(512 << 20, dtype=torch.uint8, device="cuda"); sk = torch.zeros(1, dtype=torch.int64, device="cuda")
                for _ in range(3):
                    torch.sum(fl.view(torch.int64), 0, out=sk[0]); mod.g22_gemm(P, scale, x, out, ap, ns, E, CLv, GRID, 0, 0); torch.cuda.synchronize()
                tr = mod.g22_trace().cpu()[:GRID].double(); t0 = tr[:, 0].min()
                nmz = {1: "prodWait", 2: "stage0", 3: "run0mma", 4: "run0flush", 5: "run1mma", 6: "run1flush", 7: "fixEnd", 8: "fixScan", 9: "consEnd"}
                for sl, nm_ in nmz.items():
                    v = sorted(((tr[:, sl] - t0) / 1000).tolist()); v = [a for a in v if a > -1e6 and a < 1e6]
                    if v: print(f"  {nm_:9s} n{len(v):3d} min {v[0]:.1f} med {v[len(v)//2]:.1f} max {v[-1]:.1f}")
                del fl
            if MODE == 1:
                dt = bench.time_graph(lambda: (pre(), mod.g22_gemm(P, scale, x, out, ap, ns, E, CLv, GRID, 1, min(NPRE, ns)))) - ep
            else:
                dt = bench.time_graph(lambda: mod.g22_gemm(P, scale, x, out, ap, ns, E, CLv, GRID, 0, min(NPRE, ns))) - e
            line += f" | {ap},{ns},{E},c{CLv} {dt:6.2f}"
        print(line, flush=True)
    del W, packs
