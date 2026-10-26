# gemv14 (compressed gemv12) sweep: python dev/g14.py "M,KCU,WG,PF,BK,XP;..."   (H100, graph replay, cold L2)
# env SH=qkv,o,pool,unp1k,unp selects shapes. Real weights (reference seed), exact probe columns incl. escapes.
import os, sys, torch, torch.nn.functional as F, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.cpp_extension import load_inline
import bench, reference
from g13 import ESC

cfgs = [tuple(map(int, c.split(","))) for c in sys.argv[1].split(";")]
cfgs = [c for c in cfgs if (32 * c[1] // c[2]) % (c[3] * c[5]) == 0 and c[5] <= 4]
d = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(d, "gemv14.cu")).read()
src = src.replace("G14_CONFIGS", " ".join("G14_CASE(%d,%d,%d,%d,%d)" % (m, u, wg, pf, xp) for (m, u, wg, pf, bk, xp) in sorted(set(cfgs))))
mod = load_inline(name="pj_g14_" + hashlib.md5(src.encode()).hexdigest()[:10],
                  cpp_sources="void g14_gemm(torch::Tensor, torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);",
                  cuda_sources=src, functions=["g14_gemm"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"] + ["-D" + d_ for d_ in os.environ.get("DEFS", "").split()])


def pack14(w, kc):
    """W bf16 [n, k] -> (S uint8 [ks, rg, atom, lane, 8], N int32 [ks, rg, atom, lane], header [n, ESC, 2], scale)."""
    n, k = w.shape
    b = w.contiguous().view(torch.int16).int() & 0xFFFF
    e = (b >> 7) & 0xFF
    base = int(e.max()) - 15
    code = e - base
    esc = code < 0
    assert int(esc.sum(1).max()) <= ESC
    code = torch.where(esc, 0, code)
    sm = torch.where(esc, 0, ((b >> 8) & 0x80) | (b & 0x7F))
    perm = lambda t: t.view(n // 8, 8, k // kc, kc // 32, 4, 8).permute(2, 0, 3, 1, 4, 5).contiguous()
    S = perm(sm).to(torch.uint8)
    sh = torch.tensor([4 * (i // 2) + 16 * (i % 2) for i in range(8)], device=w.device)
    N = (perm(code).long() << sh).sum(-1)
    N = torch.where(N >= 2 ** 31, N - 2 ** 32, N).int()
    scale = 2.0 ** (base - 112)
    hdr = torch.zeros(n, ESC, 2, dtype=torch.float32, device=w.device)
    hdr[:, :, 1] = 2.0 ** -15  # empty slot (format of sub_template._compress)
    rows, cols = esc.nonzero(as_tuple=True)
    if len(rows):
        slot = torch.arange(len(rows), device=w.device) - torch.searchsorted(rows, rows)
        hdr[rows, slot, 0] = cols.int().view(torch.float32)
        hdr[rows, slot, 1] = w[rows, cols].float() / scale
    return S, N, hdr, scale


import re as _re
_ns = {"torch": torch}
exec("G13_ESC = 8\n" + _re.search(r"(def _compress\(w\):.*?)\ndef _pack16", open(os.path.join(d, "sub_template.py")).read(), _re.S).group(1), _ns)
_pack14s = _ns["_pack14"]
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
        for (m_, u, wg, pf, bk, xp) in cfgs:
            if m_ != M or k % (1024 * u) or (n // 8) % bk or (n // 8 // bk) * wg > 32: continue
            thr = (n // 8 // bk) * wg * 32
            if M * (2048 * u + 64) + thr // 32 * ((M + 15) // 16) * 512 + (n // 8 // bk) * 8 * ESC * 8 > 227 * 1024: continue
            if u not in packs: packs[u] = _pack14s(W, 1024 * u)  # s36: sub_template packer (DEC3 layout)
            S, N, hdr, scale = packs[u]
            # probe: last token = one power of two at an escaped column (exact column of W expected)
            er = (hdr[:, :, 1] != 2.0 ** -15).nonzero()
            col = int(hdr[er[0, 0], er[0, 1], 0].view(torch.int32)) if len(er) else 3
            xp_ = x.clone(); xp_[M - 1, col] = 16.0
            refp = xp_.double() @ W.double().T
            out = torch.empty(M, n, dtype=torch.bfloat16, device="cuda")
            mod.g14_gemm(S, N, hdr, scale, xp_, out, ws, cnt, n // 8, u, wg, pf, bk, xp); torch.cuda.synchronize()
            err = ((out.double() - refp).norm() / refp.norm()).item()
            assert err < 5e-3 and cnt.abs().max().item() == 0, (err, nm, m_, u, wg, pf, bk, xp)
            assert torch.equal(out[M - 1], (W[:, col].double() * 16).bfloat16()), ("probe", nm, col)
            dt = bench.time_graph(lambda: mod.g14_gemm(S, N, hdr, scale, x, out, ws, cnt, n // 8, u, wg, pf, bk, xp)) - e
            line += f" | KC{u}W{wg}P{pf}X{xp} b{(k // (1024 * u)) * bk}x{(n // 8 // bk) * wg * 32} {dt:6.2f}"
        print(line, flush=True)
    del W, packs
