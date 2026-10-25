"""Build dev/gemv8.cu, check, sweep: python dev/g8.py layer:m ...   (env CFGS=all|small)"""
import os, sys, hashlib, torch
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(D)); sys.path.insert(0, D)
from torch.utils.cpp_extension import load_inline
import reference, bench
from g5 import pack, pack_router

BF = [(mt, nwr, nwk, u, 1) for mt in (1, 2, 4) for (nwr, nwk) in ((2, 4), (4, 2), (1, 8), (4, 4), (8, 1), (8, 2), (2, 2), (16, 1)) for u in (8, 16)]
HL = [(mt, nwr, nwk, 8, 2) for mt in (2, 4, 8) for (nwr, nwk) in ((4, 2), (2, 4), (1, 8), (2, 2))]
CFGS = BF + HL
SIG = "(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t)"


def build(cfgs=CFGS):
    src = open(os.path.join(D, "gemv8.cu")).read().replace("G8_CONFIGS", " ".join("G8_CASE(%d,%d,%d,%d,%d)" % c for c in cfgs))
    name = "g8_" + hashlib.md5(src.encode()).hexdigest()[:10]
    return load_inline(name=name, cpp_sources=f"void g8_gemv{SIG}; int64_t g8_occ{SIG};", cuda_sources=src,
                       functions=["g8_gemv", "g8_occ"], extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"])


def pack_units(p, nwr, cs):
    """p: [n/16, k/16, na*256] (fragment-packed) -> [row group][K chunk][stripe in group][k-tile in chunk][na*256]."""
    s, kt, e = p.shape
    return p.reshape(s // nwr, nwr, cs, kt // cs, e).permute(0, 2, 1, 3, 4).contiguous()


if __name__ == "__main__":
    mod = build()
    W = reference.generate_weights(reference.WEIGHTS_SEED)
    P = {l: (pack_router(w) if l == "router" else pack(w).contiguous()) for l, w in W.items()}
    ws = torch.zeros(1 << 20, dtype=torch.float32, device="cuda"); cnt = torch.zeros(4096, dtype=torch.int32, device="cuda")
    for arg in sys.argv[1:]:
        layer, m = arg.split(":"); m = int(m)
        data = reference.generate_input(m, layer, seed=7)
        x, n, k = data["x"], data["config"]["n"], data["config"]["k"]
        na = 2 if layer == "router" else 1
        out = torch.empty((m, n), dtype=torch.float32 if na == 2 else torch.bfloat16, device="cuda")
        wl = P[layer][: n // 16]
        ref = x.float() @ W[layer][:n].float().T
        tc = bench.time_graph(lambda: torch.nn.functional.linear(x, W[layer][:n])) - bench.empty_cost() if na == 1 else float("nan")
        res = []
        for (mt, nwr, nwk, u, a) in CFGS:
            if a != na or mt * 8 < min(m, 32) or (mt > 1 and (mt // 2) * 8 >= m): continue
            tgs = -(-m // (mt * 8))
            for cs in (1, 2, 4, 8, 16, 32):
                if k % (cs * 16 * nwk) or n % (nwr * 16): continue
                if min(m, mt * 8) * (k // cs + 8) * 2 > 160 * 1024: continue
                blocks = cs * (n // (nwr * 16)) * tgs
                if blocks < 100 or blocks > 8192: continue
                cfg = (n, mt, nwr, nwk, u, na, cs, tgs)
                wu = pack_units(wl, nwr, cs)
                try:
                    mod.g8_gemv(wu, x, out, ws, cnt, *cfg); torch.cuda.synchronize()
                except Exception as e:
                    print("ERR", cfg, str(e)[:80]); continue
                err = ((out.float() - ref).norm() / ref.norm()).item()
                mod.g8_gemv(wu, x, out, ws, cnt, *cfg); torch.cuda.synchronize()
                err2 = ((out.float() - ref).norm() / ref.norm()).item()
                t = bench.time_graph(lambda: mod.g8_gemv(wu, x, out, ws, cnt, *cfg), reps=50) - bench.empty_cost()
                res.append((t, cfg, max(err, err2), blocks))
                del wu
        res.sort()
        print(f"{layer} m={m} cublas {tc:.2f}")
        for t, cfg, err, b in res[:4]:
            print(f"   {t:6.2f} us  {cfg}  err {err:.1e} blocks {b}", flush=True)
