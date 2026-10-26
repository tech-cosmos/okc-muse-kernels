# gemv19 (gemv18 + FP32 out + dynamic tail) sweep: python dev/g19.py "M,KC,AP,NS,IS,TAILPB;..."  (TAILPB = tail stages
# per block, rounded to IS). env SH=qkv,o,pool,unp, XWARM=1 (x written right before), GRID. H100 graph replay, cold L2.
import os, sys, torch, torch.nn.functional as F, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.cpp_extension import load_inline
import bench, reference
d = os.path.dirname(os.path.abspath(__file__))
_g18 = open(os.path.join(d, "g18.py")).read()
exec(_g18[_g18.index("ESC = 8"):_g18.index("cfgs = ")])
exec(_g18[_g18.index("def pack16"):_g18.index("SH = [")].replace('"G18_ILV" in', '"G18_NEVER" in'))
cfgs = [tuple(map(int, c.split(","))) for c in sys.argv[1].split(";")]
cfgs = [c + (2, 1, 8, 1)[len(c) - 6:] for c in cfgs]  # XB (x buffers) 2, TS 1, RG 8, RPW 1 by default
src = open(os.path.join(d, "gemv19.cu")).read()
src = "".join("#define %s\n" % d_.replace("=", " ") for d_ in os.environ.get("DEFS", "").split(",") if d_) + src
src = src.replace("G19_CONFIGS", " ".join("G19_CASE(%d,%d,%d,%d,%d,%d,%d,%d,%d)" % (c[:5] + c[6:]) for c in sorted(set(cfgs))))
fns = ["g19_gemm", "g19_gemm_bf16", "g19_pred", "g19_zero", "g19_cvt"] + (["g19_trace"] if "G19_TRACE" in os.environ.get("DEFS", "") else [])
mod = load_inline(name="pj_g19_" + hashlib.md5(src.encode()).hexdigest()[:10],
                  cpp_sources="void g19_gemm(torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t); void g19_gemm_bf16(torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t); void g19_pred(torch::Tensor, torch::Tensor, int64_t); void g19_zero(torch::Tensor, torch::Tensor); void g19_cvt(torch::Tensor, torch::Tensor);" + (" torch::Tensor g19_trace();" if "G19_TRACE" in os.environ.get("DEFS", "") else ""),
                  cuda_sources=src, functions=fns, extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"])
def pack19r(w, kc, ap, rg):
    """pack18 with rg 16-row groups per tile: stage (ks, tile of 16 rg rows, st) contiguous."""
    S, N, hdr, scale = pack16(w, kc)
    n, k = w.shape
    ks, ntr, spk = k // kc, n // (16 * rg), kc // 32 // ap
    pm = (1, 0, 3, 2, 4) if TM else (0, 1, 3, 2, 4)  # tile-major: (tile, ks, st)
    S = S.view(ks, ntr, rg, spk, ap * 512).permute(*pm).reshape(ks * ntr * spk, rg * ap * 512)
    N = N.view(torch.uint8).view(ks, ntr, rg, spk, ap * 256).permute(*pm).reshape(ks * ntr * spk, rg * ap * 256)
    return torch.cat([S, N], 1).contiguous(), hdr, scale


RAW = "G19_RAW" in os.environ.get("DEFS", "")
def pack19raw(w, kc, ap, rg):
    """G19_RAW: plain BF16 stages, per (row group, atom): [s][lane (g, c)][hk][h][e] = the mma A fragments."""
    n, k = w.shape
    ks, ntr, spk = k // kc, n // (16 * rg), kc // 32 // ap
    A = w.view(n // 16, 2, 8, ks, kc // 32, 2, 2, 4, 2).permute(3, 0, 4, 5, 2, 7, 6, 1, 8).reshape(ks, ntr, rg, spk, ap * 512)
    pm = (1, 0, 3, 2, 4) if TM else (0, 1, 3, 2, 4)
    A = A.permute(*pm).reshape(ks * ntr * spk, rg * ap * 512).contiguous().view(torch.uint8)
    return A, torch.full((n, ESC, 2), 2.0 ** -15, device=w.device), 1.0


SH = [("qkv", "qkv_gate", 2304), ("o", "o", 8192), ("pool", "pooler", 4096), ("unp", "unpooler", 8192), ("poolr", "pooler", 4608)]
only = os.environ.get("SH")
GRID = int(os.environ.get("GRID", torch.cuda.get_device_properties(0).multi_processor_count))
Wr = reference.generate_weights(reference.WEIGHTS_SEED)
TM = int(os.environ.get("TM", 0))  # mode 3: tile-major stage order
ESCW = bool(os.environ.get("ESCW"))  # adversarial escapes (bench.py test_esc's injection) + spike rows at them, 40 reps
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
ctr3 = torch.zeros(4096, dtype=torch.int32, device="cuda")  # mode 3 tile tickets (never reset)
e = 0.0 if os.environ.get("PROF") else bench.empty_cost()
XW = bool(os.environ.get("XWARM"))
PRED = int(os.environ.get("PRED", -1))  # predecessor kernel: writes x, triggers PDL at start, spins PRED ns
MODE = int(os.environ.get("MODE", 0))   # 0: zero kernel + gemm; 1: out zeroed before the predecessor, gemm PDL on it
ZG = int(os.environ.get("ZG", 1))
NPRE = int(os.environ.get("NPRE", 99))  # stages before griddepcontrol.wait (capped at NS)       # mode 1: zero kernel inside the timed graph (else outside: assume fused elsewhere)
for nm, layer, n in SH:
    if only and nm not in only.split(","): continue
    W = Wr[layer][:n].contiguous() if n <= Wr[layer].shape[0] else torch.cat([Wr[layer], Wr[layer][:n - Wr[layer].shape[0]]]).contiguous()  # poolr: pooler + 512 rows (router hi/lo stand-in)
    k = W.shape[1]
    packs = {}
    for M in sorted({c[0] for c in cfgs}):
        x = torch.randn(M, k, device="cuda").bfloat16()
        xsrc = x.clone()
        pre = (lambda: mod.g19_pred(xsrc, x, PRED)) if PRED >= 0 else (lambda: x.mul_(1)) if XW else (lambda: None)
        if (XW or PRED >= 0) and not os.environ.get("PROF"): e = bench.time_graph(pre)
        tc = 0.0 if os.environ.get("PROF") else bench.time_graph(lambda: (pre(), F.linear(x, W))) - e
        line = f"{nm:5s} m{M}: cublas {tc:6.2f}"
        out = torch.empty(M, n, device="cuda")
        for (m_, kc, ap, ns, is_, tpb, xb_, ts_, rg_, rpw_) in cfgs:
            if m_ != M or k % kc: continue
            if (kc, ap, rg_) not in packs: packs[(kc, ap, rg_)] = (pack19raw if RAW else pack19r)(W, kc, ap, rg_)
            P, hdr, scale = packs[(kc, ap, rg_)]
            tail = tpb * GRID // is_ * is_
            er = (hdr[:, :, 1] != 2.0 ** -15).nonzero()
            col = int(hdr[er[0, 0], er[0, 1], 0].view(torch.int32)) if len(er) else 3
            xp_ = x.clone(); xp_[M - 1].zero_(); xp_[M - 1, col] = 16.0
            sp = []
            if ESCW:
                sp = list(range(0, M - 1, 2))
                for j, r_ in enumerate(sp):
                    xp_[r_].zero_(); xp_[r_, CE[layer][j % 32]] = 8.0
            refp = xp_.double() @ W.double().T
            if MODE == 3:
                ws3 = torch.zeros(M, n, device="cuda"); ob3 = torch.empty(M, n, dtype=torch.bfloat16, device="cuda")
                g3 = lambda xin: mod.g19_gemm_bf16(P, hdr, scale, xin, ws3, ob3, ctr3, kc, ap, ns, GRID, min(NPRE, ns), xb_, ts_, rg_, rpw_, TM)
            for rep in range(0 if os.environ.get("SKIPCHK") else 40 if ESCW else 3):
                if MODE == 3:  # BF16 out: like cuBLAS (FP32 sum rounded once); probes/spikes exact (W * 2^j is BF16)
                    ob3.fill_(1.0); g3(xp_); torch.cuda.synchronize()
                    err = ((ob3.double() - refp).norm() / refp.norm()).item()
                    assert err < 4e-3, (err, nm, m_, kc, ap, ns)
                    assert torch.count_nonzero(ws3) == 0, "ws not re-zeroed"
                    ftz = lambda v: torch.where(v.abs() < 2.0 ** -126, torch.zeros_like(v), v).bfloat16()  # red.add.f32 flushes subnormals
                    assert torch.equal(ob3[M - 1], ftz(W[:, col].float() * 16)), ("probe", nm, col)
                    for j, r_ in enumerate(sp):
                        assert torch.equal(ob3[r_], ftz(W[:, CE[layer][j % 32]].float() * 8)), ("spike", nm, r_)
                    continue
                out.fill_(1.0)
                if MODE == 1: mod.g19_zero(out, ctr)
                mod.g19_gemm(P, hdr, scale, xp_, out, ctr, kc, ap, ns, is_, GRID, tail, MODE, min(NPRE, ns), xb_, ts_, rg_, rpw_); torch.cuda.synchronize()
                err = ((out.double() - refp).norm() / refp.norm()).item()
                assert err < 1e-5, (err, nm, m_, kc, ap, ns, is_, tpb)
                pw_ = (W[:, col].double() * 16).float()
                assert torch.equal(out[M - 1], torch.where(pw_.abs() < 2.0 ** -126, 0 * pw_, pw_)), ("probe", nm, col)
                for j, r_ in enumerate(sp):
                    want = (W[:, CE[layer][j % 32]].double() * 8).float()
                    want = torch.where(want.abs() < 2.0 ** -126, torch.zeros_like(want) * want.sign(), want)  # red.add.f32 flushes subnormals
                    if not torch.equal(out[r_], want):
                        bad = (out[r_] != want).nonzero().flatten()[:6]
                        print("SPIKE MISMATCH", nm, M, r_, int(CE[layer][j % 32]), [(int(i), float(out[r_, i]), float(want[i]), float(W[i, CE[layer][j % 32]])) for i in bad], "nbad", int((out[r_] != want).sum()))
                        raise SystemExit(1)
            if ESCW: print("  esc ok", nm, M, (kc, ap, ns), "max rel err", err, flush=True)
            if "G19_TRACE" in os.environ.get("DEFS", ""):
                fl = torch.empty(512 << 20, dtype=torch.uint8, device="cuda"); sk = torch.zeros(1, dtype=torch.int64, device="cuda")
                for _ in range(3):
                    torch.sum(fl.view(torch.int64), 0, out=sk[0]); (mod.g19_zero(out, ctr) if MODE == 1 else None); pre(); mod.g19_gemm(P, hdr, scale, x, out, ctr, kc, ap, ns, is_, GRID, tail, MODE, min(NPRE, ns), xb_, ts_, rg_, rpw_); torch.cuda.synchronize()
                tr = mod.g19_trace().cpu()[:GRID]; t0 = int(tr[:, 0].min())
                if os.environ.get("SMDUMP"):
                    import pickle; R = pickle.load(open("/tmp/pj/smd.pkl", "rb")) if os.path.exists("/tmp/pj/smd.pkl") else []
                    R.append((nm, (kc, ap, ns, is_, tpb), tr[:, 10].tolist(), ((tr[:, 4] - t0).double() / 1000).tolist(), ((tr[:, 1] - t0).double() / 1000).tolist()))
                    pickle.dump(R, open("/tmp/pj/smd.pkl", "wb"))
                if os.environ.get("STAGES"):
                    for b_ in (0, 1, 66):
                        r_ = tr[b_]
                        print(f"  block {b_}: start {(int(r_[0]) - t0) / 1000:.2f} prod issue:", " ".join(f"{(int(r_[40 + j]) - t0) / 1000:.2f}" for j in range(24) if r_[40 + j] > 0))
                        print(f"           cons: wait {(int(r_[9]) - t0) / 1000:.2f} x0 {(int(r_[5]) - t0) / 1000:.2f} c0 {(int(r_[6]) - t0) / 1000:.2f} hd {(int(r_[7]) - t0) / 1000:.2f} c1 {(int(r_[8]) - t0) / 1000:.2f}")
                        print(f"           flush0: start {(int(r_[11]) - t0) / 1000:.2f} fix {(int(r_[12]) - t0) / 1000:.2f} done {(int(r_[14]) - t0) / 1000:.2f} | q1 pempty {(int(r_[13]) - t0) / 1000:.2f} done {(int(r_[15]) - t0) / 1000:.2f} | fin0 {(int(r_[2]) - t0) / 1000:.2f} fin1 {(int(r_[3]) - t0) / 1000:.2f}")
                        print(f"           cons got:", " ".join(f"{(int(r_[16 + j]) - t0) / 1000:.2f}" for j in range(24) if r_[16 + j] > 0))
                for nm_, sl in (("prodEnd", 1), ("consEnd", 4)):
                    v = sorted(((tr[:, sl] - t0).double() / 1000).tolist())
                    print(f"  {kc},{ap},{ns},{is_},{tpb} {nm_:8s} min {v[0]:.1f} p10 {v[GRID//10]:.1f} med {v[GRID//2]:.1f} p90 {v[GRID*9//10]:.1f} max {v[-1]:.1f}")
                del fl
            if os.environ.get("PROF"):  # plain launches for ncu, no timing
                for _ in range(5): mod.g19_gemm(P, hdr, scale, x, out, ctr, kc, ap, ns, is_, GRID, tail, MODE, min(NPRE, ns), xb_, ts_, rg_, rpw_)
                torch.cuda.synchronize(); continue
            ob = torch.empty(M, n, dtype=torch.bfloat16, device="cuda")
            CVT = bool(os.environ.get("CVT"))  # + the BF16 copy (PDL)
            if MODE == 3: dt = bench.time_graph(lambda: (pre(), g3(x))) - e; line += f" | {kc},{ap},{ns},{is_},{tpb},x{xb_},t{ts_},r{rg_},w{rpw_} {dt:6.2f}"; continue
            dt = bench.time_graph(lambda: ((mod.g19_zero(out, ctr) if MODE == 1 and ZG else None), pre(), mod.g19_gemm(P, hdr, scale, x, out, ctr, kc, ap, ns, is_, GRID, tail, MODE, min(NPRE, ns), xb_, ts_, rg_, rpw_), mod.g19_cvt(out, ob) if CVT else None)) - e
            line += f" | {kc},{ap},{ns},{is_},{tpb},x{xb_},t{ts_},r{rg_},w{rpw_} {dt:6.2f}"
        print(line, flush=True)
    del W, packs
