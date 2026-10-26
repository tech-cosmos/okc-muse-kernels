# gemv16 (gemv14 + cluster DSMEM split-K) sweep: python dev/g16.py "M,KCU,WG,PF,BK,XP;..." (BK = row runs = clusters)   (H100, graph replay, cold L2)
# env SH=qkv,o,pool,unp1k,unp selects shapes. Real weights (reference seed), exact probe columns incl. escapes.
import os, sys, torch, torch.nn.functional as F, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.cpp_extension import load_inline
import bench, reference
from g13 import ESC

cfgs = [tuple(map(int, c.split(","))) for c in sys.argv[1].split(";")]
cfgs = [c if len(c) == 7 else c + (1,) for c in cfgs]
cfgs = [c for c in cfgs if (32 * c[1] // c[2]) % c[3] == 0 and (32 * c[1] // c[2]) % c[5] == 0 and c[5] <= 4]
d = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(d, "gemv16.cu")).read()
src = "".join("#define %s\n" % d_ for d_ in os.environ.get("DEFS", "").split(",") if d_) + src
src = src.replace("G16_CONFIGS", " ".join("G16_CASE(%d,%d,%d,%d,%d,%d)" % (m, u, wg, pf, xp, mc) for (m, u, wg, pf, bk, xp, mc) in sorted(set(cfgs))))
mod = load_inline(name="pj_g16_" + hashlib.md5(src.encode()).hexdigest()[:10],
                  cpp_sources="void g16_gemm(torch::Tensor, torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);",
                  cuda_sources=src, functions=["g16_gemm"], verbose=bool(os.environ.get("PTXV")), extra_cuda_cflags=["-O3"] + (["-Xptxas", "-v"] if os.environ.get("PTXV") else []) + ["-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"])


def pack16(w, kc):
    """W bf16 [n, k] -> (S uint8 [ks, rg16, atom, lane, 16], N int32 [ks, rg16, atom, lane, 2], header, scale)."""
    n, k = w.shape
    b = w.contiguous().view(torch.int16).int() & 0xFFFF
    e = (b >> 7) & 0xFF
    base = int(e.max()) - 15
    code = e - base
    esc = code < 0
    assert int(esc.sum(1).max()) <= ESC
    code = torch.where(esc, 0, code)
    sm = torch.where(esc, 0, ((b >> 8) & 0x80) | (b & 0x7F))
    # [rg, h, g, ks, atom, s, hk, c, e] -> [ks, rg, atom, g, c, s, hk, h, e]
    perm = lambda t: t.view(n // 16, 2, 8, k // kc, kc // 32, 2, 2, 4, 2).permute(3, 0, 4, 2, 7, 5, 6, 1, 8).contiguous()
    S = perm(sm).to(torch.uint8).reshape(k // kc, n // 16, kc // 32, 32, 16)
    sh = torch.tensor([4 * (i // 2) + 16 * (i % 2) for i in range(8)], device=w.device)
    N = (perm(code).reshape(k // kc, n // 16, kc // 32, 32, 2, 8).long() << sh).sum(-1)
    N = torch.where(N >= 2 ** 31, N - 2 ** 32, N).int().contiguous()
    scale = 2.0 ** (base - 112)
    hdr = torch.zeros(n, ESC, 2, dtype=torch.float32, device=w.device)
    hdr[:, :, 1] = 2.0 ** -15  # empty slot (format of sub_template._compress)
    rows, cols = esc.nonzero(as_tuple=True)
    if len(rows):
        slot = torch.arange(len(rows), device=w.device) - torch.searchsorted(rows, rows)
        hdr[rows, slot, 0] = cols.int().view(torch.float32)
        hdr[rows, slot, 1] = w[rows, cols].float() / scale
    return S, N, hdr, scale


SH = [("qkv", "qkv_gate", 2304), ("o", "o", 8192), ("pool", "pooler", 4096), ("unp1k", "unpooler", 1024), ("unp", "unpooler", 8192)]
only = os.environ.get("SH")
Wr = reference.generate_weights(reference.WEIGHTS_SEED)
ws = torch.zeros(8 * 32 * 8192, device="cuda"); cnt = torch.zeros(1024, dtype=torch.int32, device="cuda")
e = bench.empty_cost()
for nm, layer, n in SH:
    if only and nm not in only.split(","): continue
    W = Wr[layer][:n].contiguous()
    k = W.shape[1]
    packs = {}
    for M in sorted({c[0] for c in cfgs}):
        if nm == "unp1k" and M > 8: continue
        if nm == "unp" and M <= 8: continue
        x = torch.randn(M, k, device="cuda").bfloat16()
        x[M - 1].zero_()
        hdr_any = None
        ref = x.double() @ W.double().T
        tc = bench.time_graph(lambda: F.linear(x, W)) - e
        line = f"{nm:5s} m{M}: cublas {tc:6.2f}"
        for (m_, u, wg, pf, bk, xp, mc) in cfgs:
            if m_ != M or k % (1024 * u) or (n // 16) % bk or (n // 16 // bk) * wg > 32: continue
            thr = (n // 16 // bk) * wg * 32
            if M * (2048 * u + 16) + thr // 32 * (M // 8) * 512 + (n // 16 // bk) * 16 * ESC * 8 + (n // 16 // bk) * M * 64 > 227 * 1024 or k // (1024 * u) * mc > 16 or bk % mc: continue
            if u not in packs: packs[u] = pack16(W, 1024 * u)
            S, N, hdr, scale = packs[u]
            # probe: last token = one power of two at an escaped column (exact column of W expected)
            er = (hdr[:, :, 1] != 2.0 ** -15).nonzero()
            col = int(hdr[er[0, 0], er[0, 1], 0].view(torch.int32)) if len(er) else 3
            xp_ = x.clone(); xp_[M - 1, col] = 16.0
            refp = xp_.double() @ W.double().T
            out = torch.empty(M, n, dtype=torch.bfloat16, device="cuda")
            mod.g16_gemm(S, N, hdr, scale, xp_, out, n // 16, u, wg, pf, bk, xp, mc); torch.cuda.synchronize()
            err = ((out.double() - refp).norm() / refp.norm()).item()
            if not os.environ.get("DEFS"): assert err < 5e-3, (err, nm, m_, u, wg, pf, bk, xp)
            if not os.environ.get("DEFS"): assert torch.equal(out[M - 1], (W[:, col].double() * 16).bfloat16()), ("probe", nm, col)
            dt = bench.time_graph(lambda: mod.g16_gemm(S, N, hdr, scale, x, out, n // 16, u, wg, pf, bk, xp, mc)) - e
            line += f" | KC{u}W{wg}P{pf}X{xp}C{mc} S{k // (1024 * u)}c{bk}x{(n // 16 // bk) * wg * 32} {dt:6.2f}"
        print(line, flush=True)
    del W, packs
