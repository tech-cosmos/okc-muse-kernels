# gemv18 (TMA ring streamer over 12-bit weights, m16/m32, stream-K) sweep: python dev/g18.py "M,KC,AP,NS;..."
# env SH=qkv,o,pool,unp selects shapes; GRID (default #SMs). H100 graph replay, cold L2; real weights, exact probes.
import os, sys, torch, torch.nn.functional as F, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.cpp_extension import load_inline
import bench, reference
ESC = 8
F32 = bool(os.environ.get("F32"))  # mode 2: FP32 out into wsf, zero kernel as PDL primary
MODE = 2 if F32 else 0

cfgs = [tuple(map(int, c.split(","))) for c in sys.argv[1].split(";")]
d = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(d, "gemv18.cu")).read()
src = "".join("#define %s\n" % d_ for d_ in os.environ.get("DEFS", "").split(",") if d_) + src
src = src.replace("G18_CONFIGS", " ".join("G18_CASE(%d,%d,%d,%d)" % c for c in sorted(set(cfgs))))
mod = load_inline(name="pj_g18_" + hashlib.md5(src.encode()).hexdigest()[:10],
                  cpp_sources="void g18_gemm(torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t);",
                  cuda_sources=src, functions=["g18_gemm"], verbose=bool(os.environ.get("PTXV")), extra_cuda_cflags=["-O3"] + (["-Xptxas", "-v"] if os.environ.get("PTXV") else []) + ["-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"])


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



def pack18(w, kc, ap):
    """Stage u = (ks, 128-row tile, st) -> [8 row groups x ap atoms x 32 lanes x 16 B][8 x ap x 32 x 8 B] contiguous."""
    S, N, hdr, scale = pack16(w, kc)
    n, k = w.shape
    ks, ntr, spk = k // kc, n // 128, kc // 32 // ap
    S = S.view(ks, ntr, 8, spk, ap * 512).permute(0, 1, 3, 2, 4).reshape(ks * ntr * spk, 8 * ap * 512)
    N = N.view(torch.uint8).view(ks, ntr, 8, spk, ap * 256).permute(0, 1, 3, 2, 4).reshape(ks * ntr * spk, 8 * ap * 256)
    P = torch.cat([S, N], 1).contiguous()
    if "G18_ILV" in os.environ.get("DEFS", "") + os.environ.get("TDEFS", ""):  # stage i of block b at i * G + b
        U = P.shape[0]
        bnd = [U * b // GRID for b in range(GRID + 1)]
        L = max(bnd[b + 1] - bnd[b] for b in range(GRID))
        P2 = torch.zeros(L * GRID, P.shape[1], dtype=P.dtype, device=P.device)
        for b in range(GRID):
            P2[torch.arange(bnd[b + 1] - bnd[b], device=P.device) * GRID + b] = P[bnd[b]:bnd[b + 1]]
        P = P2
    return P, hdr, scale


SH = [("qkv", "qkv_gate", 2304), ("o", "o", 8192), ("pool", "pooler", 4096), ("unp", "unpooler", 8192)]
only = os.environ.get("SH")
GRID = int(os.environ.get("GRID", torch.cuda.get_device_properties(0).multi_processor_count))
Wr = reference.generate_weights(reference.WEIGHTS_SEED)
wsf = torch.zeros(32 * 8192, device="cuda"); cnt = torch.zeros(1024, dtype=torch.int32, device="cuda")
e = bench.empty_cost()
for nm, layer, n in SH:
    if only and nm not in only.split(","): continue
    W = Wr[layer][:n].contiguous()
    k = W.shape[1]
    packs = {}
    for M in sorted({c[0] for c in cfgs}):
        x = torch.randn(M, k, device="cuda").bfloat16()
        x[M - 1].zero_()
        XW = bool(os.environ.get("XWARM"))  # x written by a kernel right before (hot in L2, as in the layer)
        pre = (lambda: x.mul_(1)) if XW else (lambda: None)
        if XW: e = bench.time_graph(pre)
        tc = bench.time_graph(lambda: (pre(), F.linear(x, W))) - e
        line = f"{nm:5s} m{M}: cublas {tc:6.2f}"
        for (m_, kc, ap, ns) in cfgs:
            if m_ != M or k % kc: continue
            if (kc, ap) not in packs: packs[(kc, ap)] = pack18(W, kc, ap)
            P, hdr, scale = packs[(kc, ap)]
            er = (hdr[:, :, 1] != 2.0 ** -15).nonzero()
            col = int(hdr[er[0, 0], er[0, 1], 0].view(torch.int32)) if len(er) else 3
            xp_ = x.clone(); xp_[M - 1, col] = 16.0
            refp = xp_.double() @ W.double().T
            out = torch.empty(M, n, dtype=torch.bfloat16, device="cuda")
            for rep in range(1 if os.environ.get("DEFS") else 3):
                out.zero_(); wsf.fill_(1.0) if F32 else None
                mod.g18_gemm(P, hdr, scale, xp_, out, wsf, cnt, kc, ap, ns, GRID, MODE); torch.cuda.synchronize()
                o_ = wsf[:M * n].view(M, n) if F32 else out
                err = ((o_.double() - refp).norm() / refp.norm()).item()
                if not os.environ.get("DEFS"): assert err < (1e-5 if F32 else 5e-3), (err, nm, m_, kc, ap, ns)
                if not os.environ.get("DEFS"): assert torch.equal(o_[M - 1], (W[:, col].double() * 16).to(o_.dtype)), ("probe", nm, col)
            if not os.environ.get("DEFS") and not F32: assert int(cnt.abs().sum()) == 0 and float(wsf.abs().sum()) == 0, "not self-cleaned"
            if F32: wsf.zero_()
            dt = bench.time_graph(lambda: (pre(), mod.g18_gemm(P, hdr, scale, x, out, wsf, cnt, kc, ap, ns, GRID, MODE))) - e
            line += f" | KC{kc}A{ap}N{ns} {dt:6.2f}"
        print(line, flush=True)
    del W, packs
