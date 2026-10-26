# gemv13 (compressed-weight m=1 GEMV) sweep: python dev/g13.py "U,NV,MINB,PF;..." blocks [blocks ...]
# env SH=qkv,o,... selects shapes. Checks against FP64 (incl. probe-style exact columns) and times vs gemv10's config.
import os, sys, torch, torch.nn.functional as F, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.cpp_extension import load_inline
import bench, reference

ESC = 8


def pack13(w):
    """W bf16 [n, k] -> (sign|mantissa bytes [n*k] u8, codes [n*k/8] i32, header [n, ESC, 2] f32, scale) or None."""
    n, k = w.shape
    b = w.contiguous().view(torch.int16).int() & 0xFFFF
    e = (b >> 7) & 0xFF
    base = int(e.max()) - 15
    if base < 1:
        return None
    code = e - base
    esc = code < 0
    per_row = esc.sum(1)
    if int(per_row.max()) > ESC:
        return None
    code = torch.where(esc, 0, code)
    sm = torch.where(esc, 0, ((b >> 8) & 0x80) | (b & 0x7F)).to(torch.uint8).flatten()
    c8 = code.flatten().view(-1, 8).long()
    sh = torch.tensor([4 * (i // 2) + 16 * (i % 2) for i in range(8)], device=w.device)
    nib = (c8 << sh).sum(1)
    nib = torch.where(nib >= 2 ** 31, nib - 2 ** 32, nib).int()
    D = 2.0 ** (base - 127)
    hdr = torch.zeros(n, ESC, 2, dtype=torch.float32, device=w.device)
    hdr[:, :, 1] = 2.0 ** -15  # empty slot (format of sub_template._compress)
    rows, cols = esc.nonzero(as_tuple=True)
    if len(rows):
        slot = torch.zeros_like(rows)
        # slot = rank of the escape within its row
        order = torch.argsort(rows * k + cols)
        rows, cols = rows[order], cols[order]
        first = torch.searchsorted(rows, rows, right=False)
        slot = torch.arange(len(rows), device=w.device) - first
        delta = w[rows, cols].float() / 2.0 ** (base - 112)  # exact; the kernel subtracts D / scale = 2^-15
        hdr[rows, slot, 0] = cols.int().view(torch.float32)
        hdr[rows, slot, 1] = delta
    return sm, nib, hdr, 2.0 ** (base - 112)


if __name__ == "__main__":
    cfgs = [tuple(map(int, c.split(","))) for c in sys.argv[1].split(";")]
    grids = [int(g) for g in (sys.argv[2:] or ["132"])]
    SH = [("qkv", 2304, 8192), ("o", 8192, 1024), ("pool", 4096, 8192), ("unp1k", 1024, 4096), ("unp", 8192, 4096)]
    only = os.environ.get("SH")
    SH = [s for s in SH if not only or s[0] in only.split(",")]
    ks = sorted({k for _, _, k in SH})
    d = os.path.dirname(os.path.abspath(__file__))
    src = open(os.path.join(d, "gemv13.cu")).read()
    src = src.replace("G13_CONFIGS", " ".join("G13_CASE(%d,%d,%d,%d,%d)" % (k, u, v, b, p) for (u, v, b, p) in cfgs for k in ks if (u * 4096) % k == 0))
    mod = load_inline(name="pj_g13_" + hashlib.md5(src.encode()).hexdigest()[:10],
                      cpp_sources="void g13_gemv(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, int64_t, int64_t, int64_t, int64_t, int64_t);",
                      cuda_sources=src, functions=["g13_gemv"], extra_cuda_cflags=os.environ.get("XF", "").split() + ["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"])
    Wr = reference.generate_weights(reference.WEIGHTS_SEED)
    names = {"qkv": "qkv_gate", "o": "o", "pool": "pooler", "unp1k": "unpooler", "unp": "unpooler"}
    e = bench.empty_cost()
    for nm, n, k in SH:
        W = Wr[names[nm]][:n].contiguous()
        pk = pack13(W)
        assert pk is not None
        sm, nib, hdr, scale = pk
        print(nm, "escapes", int((hdr[:, :, 1] != 2.0 ** -15).sum()), "scale", scale, flush=True)
        x = torch.randn(1, k, device="cuda").bfloat16()
        ref = x.double() @ W.double().T
        tc = bench.time_graph(lambda: F.linear(x, W)) - e
        line = f"{nm:5s} m1: cublas {tc:6.2f}"
        # exactness: x = single power of two picks a column, incl. escaped columns
        er = (hdr[:, :, 1] != 2.0 ** -15).nonzero()
        probe_cols = [0, 5] + ([int(hdr[er[0, 0], er[0, 1], 0].view(torch.int32))] if len(er) else [])
        for (u, nv, mb, pf) in cfgs:
            if (u * 4096) % k: continue
            for b in grids:
                out = torch.empty(1, n, dtype=torch.bfloat16, device="cuda")
                mod.g13_gemv(sm, nib, hdr, x, out, scale, u, nv, mb, pf, b); torch.cuda.synchronize()
                err = ((out.double() - ref).norm() / ref.norm()).item()
                assert err < 5e-3, (err, nm, u, nv, mb, pf, b)
                for col in probe_cols:
                    xp = torch.zeros(1, k, device="cuda", dtype=torch.bfloat16); xp[0, col] = 32.0
                    op = torch.empty(1, n, dtype=torch.bfloat16, device="cuda")
                    mod.g13_gemv(sm, nib, hdr, xp, op, scale, u, nv, mb, pf, b)
                    exp = (W[:, col].double() * 32).bfloat16()
                    assert os.environ.get("XF") or torch.equal(op[0], exp), (nm, col, (op[0].double() - exp.double()).abs().max().item())
                dt = bench.time_graph(lambda: mod.g13_gemv(sm, nib, hdr, x, out, scale, u, nv, mb, pf, b)) - e
                line += f" | U{u}V{nv}B{mb}P{pf} g{b} {dt:6.2f}"
        print(line, flush=True)
