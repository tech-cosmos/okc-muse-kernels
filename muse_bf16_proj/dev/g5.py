"""Build dev/gemv5.cu, check it, sweep configs: python dev/g5.py [layer:m ...]"""
import os, sys, itertools, hashlib, torch
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(D))
from torch.utils.cpp_extension import load_inline
import reference, bench

US = (8, 16) if os.environ.get("KER") in ("6", "7") else (4, 8)
BF = [(mt, nwr, nwk, u, 1) for mt in (1, 2, 4) for (nwr, nwk) in ((4, 2), (2, 4), (8, 1), (4, 1), (2, 2)) for u in US]
if os.environ.get("DEEP"): BF = [(1, nwr, nwk, u, 1) for (nwr, nwk) in ((2, 4), (2, 8), (1, 8), (4, 4), (1, 16), (2, 2), (4, 2)) for u in (16, 24, 32)]
if os.environ.get("WIDE"): BF = [(mt, nwr, nwk, u, 1) for mt in (1, 2) for (nwr, nwk) in ((1, 16), (1, 8), (2, 8), (2, 16), (4, 8)) for u in US]
HL = [(mt, nwr, nwk, US[0], 2) for mt in (2, 4, 8, 16) for (nwr, nwk) in ((4, 2), (2, 4), (4, 1), (2, 2), (1, 8), (1, 4), (2, 1))]
CFGS = [c for c in BF + HL if not (c[0] == 16 and c[3] == 8)]


def build(cfgs=CFGS):
    K = os.environ.get("KER", "5")
    src = open(os.path.join(D, f"gemv{K}.cu")).read().replace(f"G{K}_CONFIGS", " ".join(f"G{K}_CASE(%d,%d,%d,%d,%d)" % c for c in cfgs))
    name = "g5_" + hashlib.md5(src.encode()).hexdigest()[:10]
    return load_inline(name=name, cpp_sources=f"void g{K}_gemv(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, "
                       "int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);" + (f"int64_t g{K}_occ(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);" if K in "67" else ""),
                       cuda_sources=src, functions=[f"g{K}_gemv"] + ([f"g{K}_occ"] if K in "67" else []),
                       extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3", "--use_fast_math"],
                       build_directory=None, verbose=False)


def pack(w):  # [n, k] bf16 -> [n/16, k/16, 512] fragment order
    n, k = w.shape
    return w.view(n // 16, 2, 8, k // 16, 2, 4, 2).permute(0, 3, 2, 5, 4, 1, 6).reshape(n // 16, k // 16, 256)


def pack_ilv(p, n, k, mt, nwr, nwk, u, na, cs):
    """p: [n/16, k/16, na*256] -> [tile t][grid warp][na*256] for this grid (token groups share W: tgs = 1 layout)."""
    nrg, kb = n // (nwr * 16), k // cs
    kt = kb // 16 // nwk
    b = torch.arange(cs * nrg, device=p.device)
    ks, rg = b % cs, b // cs
    w = torch.arange(nwr * nwk, device=p.device)
    wr, wk = w % nwr, w // nwr
    s = (rg[:, None] * nwr + wr[None, :]).reshape(-1)                              # [gw]
    t0 = (ks[:, None] * (kb // 16) + wk[None, :] * kt).reshape(-1)                  # [gw]
    t = torch.arange(kt, device=p.device)
    pf = p.reshape(n // 16, k // 16, -1)
    return pf[s[None, :], t0[None, :] + t[:, None]].contiguous()                     # [kt, gw, na*256]


def pack_router(r):
    hi = r.to(torch.bfloat16); lo = (r - hi.float()).to(torch.bfloat16)
    return torch.stack([pack(hi), pack(lo)], 2).contiguous()


if __name__ == "__main__":
    mod = build(); mod.g5_gemv = getattr(mod, "g%s_gemv" % os.environ.get("KER", "5"))
    W = reference.generate_weights(reference.WEIGHTS_SEED)
    P = {l: (pack_router(w) if l == "router" else pack(w).contiguous()) for l, w in W.items()}
    for arg in sys.argv[1:]:
        layer, m = arg.split(":"); m = int(m)
        data = reference.generate_input(m, layer, seed=7)
        x, n = data["x"], data["config"]["n"]
        na = 2 if layer == "router" else 1
        out = torch.empty((m, n), dtype=torch.float32 if na == 2 else torch.bfloat16, device="cuda")
        wl = P[layer]
        if n < wl.shape[0] * 16: wl = wl[: n // 16].contiguous()
        ref = data["x"].float() @ W[layer][:n].float().T
        if layer != "router":
            tc = bench.time_graph(lambda: torch.nn.functional.linear(x, W[layer][:n])) - bench.empty_cost()
        else:
            hi_t = P.get("hi_t")
            tc = float("nan")
        res = []
        for (mt, nwr, nwk, u, a) in CFGS:
            if a != na or mt * 8 < min(m, 128 if na == 1 else 32) or (mt > 4 and (mt // 2) * 8 >= m): continue
            tgs = -(-m // (mt * 8))
            for cs in ((1, 2, 4, 8, 16) if na == 2 else (1, 2, 4, 8)):
                k = x.shape[1]
                if k % (cs * 16 * nwk) or n % (nwr * 16): continue
                if min(m, mt * 8) * (k // cs + 8) * 2 > 200 * 1024: continue
                blocks = cs * (n // (nwr * 16)) * tgs
                if blocks < (32 if na == 2 else 100) or blocks > 4096: continue
                cfg = (n, mt, nwr, nwk, u, na, cs, tgs)
                if hasattr(mod, "g%s_occ" % os.environ.get("KER")):
                    occ = getattr(mod, "g%s_occ" % os.environ.get("KER"))(wl, x, out, *cfg) * cs
                    if blocks > occ: continue
                wuse = wl
                if os.environ.get("ILV"):
                    if tgs > 1: continue
                    wuse = pack_ilv(wl, n, k, mt, nwr, nwk, u, na, cs)
                    cfg = cfg[:5] + (na + 16,) + cfg[6:]
                try:
                    mod.g5_gemv(wuse, x, out, *cfg); torch.cuda.synchronize()
                except Exception as e:
                    print("ERR", cfg, str(e)[:80]); continue
                err = ((out.float() - ref).norm() / ref.norm()).item()
                t = bench.time_graph(lambda: mod.g5_gemv(wuse, x, out, *cfg), reps=50) - bench.empty_cost()
                res.append((t, cfg, err))
        res.sort()
        print(f"{layer} m={m} cublas {tc:.2f}")
        for t, cfg, err in res[:6]:
            print(f"   {t:6.2f} us  {cfg}  err {err:.1e}", flush=True)
