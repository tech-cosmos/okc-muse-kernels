# n5s13: 11-bit (3-bit exponent window, code 7 = escape) m=1 GEMV prototype vs gemv13 (12-bit DEC3).
# Timing + non-escape decode check only (escapes decode to a placeholder; no fixups yet).
# python dev/n5s13/e11.py  (cwd muse_bf16_proj)
import os, sys, torch, hashlib
sys.path.insert(0, os.getcwd())
from torch.utils.cpp_extension import load_inline
import bench, reference
import submission as _sub
CF = [(2, 4, 1, 1, 256), (2, 2, 1, 2, 256), (1, 2, 1, 1, 512), (2, 4, 1, 2, 256)]
SH = [("qkv", "qkv_gate", 2304, 8192), ("pool", "pooler", 4096, 8192), ("unp", "unpooler", 8192, 4096)]
src0 = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemv13e.cu")).read()
cases = " ".join("G13_CASE(%d,%d,%d,%d,%d,%d)" % (k, u, v, b, p, vt) for (u, v, b, p, vt) in CF for k in (4096, 8192) if (u * vt * 16) % k == 0)
src0 = src0.replace("G13_CONFIGS", cases)
decl = "void g13_gemv(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);"
def build(defs):
    s = "".join("#define %s\n" % d for d in defs) + src0
    return load_inline(name="pj_e11_" + hashlib.md5(s.encode()).hexdigest()[:10], cpp_sources=decl, cuda_sources=s,
                       functions=["g13_gemv"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])
m12, m11, m11n = build([]), build(["E11"] + os.environ.get("D11", "").split()), build(["E11", "NOESC"])

def pack11(w):
    n, k = w.shape
    b = w.contiguous().view(torch.int16).int() & 0xFFFF
    e = (b >> 7) & 0xFF
    h = torch.bincount(e.flatten(), minlength=256)
    lo = max(range(1, 240), key=lambda l: int(h[l:l + 7].sum()))
    d = e - lo
    soft = (d < 0) | (d > 6)
    v = torch.where(d >= 7, d - 7, 1 - d)
    hard = soft & ((d > 8) | (d < -14))
    v = torch.where(hard, 15, v)
    sm = torch.where(hard, 0, ((b >> 8) & 0x80) | (b & 0x7F)).to(torch.uint8).flatten()
    c = torch.where(soft, 7, d).view(-1, 16).long()
    w32 = torch.zeros(c.shape[0], dtype=torch.long, device=w.device)
    for p in range(5):
        w32 |= c[:, 2 * p] << (3 * p) | c[:, 2 * p + 1] << (16 + 3 * p)
    w32 |= (c[:, 10] & 1) << 15 | (c[:, 11] & 1) << 31
    hh = c[:, 12] | c[:, 14] << 3 | ((c[:, 10] >> 1) & 3) << 6 | (c[:, 13] | c[:, 15] << 3 | ((c[:, 11] >> 1) & 3) << 6) << 8
    w32 = torch.where(w32 >= 2 ** 31, w32 - 2 ** 32, w32).int()
    hh = torch.where(hh >= 2 ** 15, hh - 2 ** 16, hh).short()
    N = torch.cat([w32.view(torch.uint8), hh.view(torch.uint8)]).view(torch.int32)
    # nibble records: per 512 flat weights, soft escapes in flat order
    sf = soft.flatten(); vf = v.flatten()
    idx = sf.nonzero().flatten()
    unit = idx // 512
    cnt = torch.bincount(unit, minlength=n * k // 512)
    if int(cnt.max()) > 32:
        return None
    start = torch.cumsum(cnt, 0) - cnt
    rank = torch.arange(len(idx), device=w.device) - start[unit]
    nib = torch.zeros(n * k // 512, 32, dtype=torch.long, device=w.device)
    nib[unit, rank] = vf[idx].long()
    words = (nib.view(-1, 4, 8) << (4 * torch.arange(8, device=w.device))).sum(2)
    words = torch.where(words >= 2 ** 31, words - 2 ** 32, words).int().contiguous()
    # hard escapes -> per-row header (col, true / scale); empty slot = PH = 2^-21
    scale = 2.0 ** (lo - 120)
    hdr = torch.zeros(n, 8, 2, dtype=torch.float32, device=w.device); hdr[:, :, 1] = 2.0 ** -21
    rows, cols = hard.nonzero(as_tuple=True)
    if len(rows):
        if int(torch.bincount(rows).max()) > 8:
            return None
        first = torch.searchsorted(rows, rows)
        slot = torch.arange(len(rows), device=w.device) - first
        hdr[rows, slot, 0] = cols.int().view(torch.float32)
        hdr[rows, slot, 1] = w[rows, cols].float() / scale
    return sm, N, hdr, words, scale, float(soft.float().mean()), int(cnt.max()), hard.nonzero()

Wr = reference.generate_weights(reference.WEIGHTS_SEED)
e0 = bench.empty_cost()
for nm, key, n, k in SH:
    W = Wr[key][:n].contiguous()
    x = torch.randn(1, k, device="cuda").bfloat16()
    sm, nib, hdr, sc = _sub._pack13(W)
    s11, N11, h11, sec, sc11, er, cmax, hardrc = pack11(W)
    dummy = torch.zeros(16, dtype=torch.int32, device='cuda')
    ref = x.double() @ W.double().T
    line = f"{nm} esc11 {er*100:.2f}% max/unit {cmax} hard {len(hardrc)} bytes12 {(sm.numel()+nib.numel()*4)/2**20:.1f}MB bytes11 {(s11.numel()+N11.numel()*4+sec.numel()*4)/2**20:.1f}MB |"
    for (u, v, mb, pf, vt) in CF:
        if (u * vt * 16) % k: continue
        o12 = torch.empty(1, n, dtype=torch.bfloat16, device="cuda"); o11 = torch.empty_like(o12)
        m12.g13_gemv(sm, nib, hdr, dummy, x, o12, sc, u, v, mb, pf, 132, vt)
        m11.g13_gemv(s11, N11, h11, sec, x, o11, sc11, u, v, mb, pf, 132, vt)
        torch.cuda.synchronize()
        cols = [0, 5, int(hardrc[0, 1])] if len(hardrc) else [0, 5]
        soft_cols = ((W.view(torch.int16).int() >> 7 & 0xFF) < 113).nonzero()[:3, 1].tolist()
        for col in cols + soft_cols:
            xp = torch.zeros(1, k, device='cuda', dtype=torch.bfloat16); xp[0, col] = 32.0
            op = torch.empty(1, n, dtype=torch.bfloat16, device='cuda')
            m11.g13_gemv(s11, N11, h11, sec, xp, op, sc11, u, v, mb, pf, 132, vt)
            assert os.environ.get('D11') or torch.equal(op[0], (W[:, col].double() * 32).bfloat16()), (nm, col)
        e12 = ((o12.double() - ref).norm() / ref.norm()).item(); e11 = ((o11.double() - ref).norm() / ref.norm()).item()
        assert e12 < 5e-3 and (e11 < 5e-3 or os.environ.get("D11")), (e12, e11)
        t12 = bench.time_graph(lambda: m12.g13_gemv(sm, nib, hdr, dummy, x, o12, sc, u, v, mb, pf, 132, vt)) - e0
        t11 = bench.time_graph(lambda: m11.g13_gemv(s11, N11, h11, sec, x, o11, sc11, u, v, mb, pf, 132, vt)) - e0
        t11n = bench.time_graph(lambda: m11n.g13_gemv(s11, N11, h11, sec, x, o11, sc11, u, v, mb, pf, 132, vt)) - e0
        line += f" U{u}V{v}P{pf}T{vt} {t12:.2f}/{t11:.2f}/{t11n:.2f}"
    print(line, flush=True)
