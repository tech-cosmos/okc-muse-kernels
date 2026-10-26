# gemv19 interface (for r-layer): m = 16/32 GEMM over lossless 12-bit BF16 weights, FP32 output [m, n].
#   c = _compress(w)                      # sub_template._compress (bit-exact round trip; None -> keep cuBLAS)
#   P = pack19(c, kc, ap)                 # uint8 [stages, 8 * ap * 768]: one contiguous block per stage
#   lib = build([(m, kc, ap, ns), ...])   # load_inline of gemv19.cu (~40 s for 8 configs)
#   ctr = torch.zeros(1, 2, dtype=torch.int32) per call site (ctr[1] = epoch counter: never reset; a state clone is fine)
#   lib.g19_gemm(P, hdr, scale, x, out, ctr, kc, ap, ns, 1, grid, 0, 2, npre)
#     x bf16 [m, k] contiguous, out fp32 [m, n] (any contents: the kernel zeroes it, then red.adds the split-K
#     partials; exact FP32 sums of BF16 products, subnormal results flushed to 0 by red.add), grid = #SMs.
#     Launched as a PDL dependent of the previous kernel on the stream: its producer streams the first ns weight stages
#     BEFORE griddepcontrol.wait (only x / out / ctr wait), so let the predecessor trigger launch_dependents early.
#     Its own dependents: it never triggers (they launch at its end).
#   lib.g19_cvt(out, out_bf16)   # optional BF16 copy (PDL dependent, round to nearest), e.g. for qkv -> attention
# H100 (r-proj dev/g19.py, after a 3 us PDL predecessor that writes x, cold L2) us, cuBLAS F.linear in brackets:
#   m16: qkv (16,1024,4,4) 11.8 [17.1], o (16,512,4,3) 7.3 [8.5], pooler (16,1024,4,4) 20.3-20.8 [27.5],
#        unpooler (16,1024,4,4) 20.7 [25.2];  m32: qkv (32,512,4,3) 16.3 [18.2], pooler (32,512,4,4) 24.0 [28.2],
#        unpooler (32,512,4,4) 23.5 [25.7], o: keep cuBLAS (9.1-9.9 vs 8.5).
import os, hashlib, torch
from torch.utils.cpp_extension import load_inline

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemv19.cu")


def pack19(c, kc, ap):
    """(code, sm, hdr, scale) from _compress -> stage-major uint8 [k/kc * n/128 * kc/32/ap, 8 * ap * 768]:
    stage (ks, 128-row tile, st) = [8 row groups x ap atoms x 32 lanes x 16 B of sign|mantissa][same x 8 B of codes]."""
    code, sm, hdr, scale = c
    n, k = code.shape
    # [rg, h, g, ks, atom, s, hk, c, e] -> [ks, rg, atom, g, c, s, hk, h, e] (mma A fragment order per lane)
    perm = lambda t: t.view(n // 16, 2, 8, k // kc, kc // 32, 2, 2, 4, 2).permute(3, 0, 4, 2, 7, 5, 6, 1, 8).contiguous()
    S = perm(sm).to(torch.uint8).reshape(k // kc, n // 16, kc // 32, 32, 16)
    sh = torch.tensor([4 * (i // 2) + 16 * (i % 2) for i in range(8)], device=code.device)
    N = (perm(code).reshape(k // kc, n // 16, kc // 32, 32, 2, 8).long() << sh).sum(-1)
    N = torch.where(N >= 2 ** 31, N - 2 ** 32, N).int().contiguous()
    ks, ntr, spk = k // kc, n // 128, kc // 32 // ap
    S = S.view(ks, ntr, 8, spk, ap * 512).permute(0, 1, 3, 2, 4).reshape(ks * ntr * spk, 8 * ap * 512)
    N = N.view(torch.uint8).view(ks, ntr, 8, spk, ap * 256).permute(0, 1, 3, 2, 4).reshape(ks * ntr * spk, 8 * ap * 256)
    return torch.cat([S, N], 1).contiguous()


def build(cfgs, src=None):
    """cfgs: (m, kc, ap, ns) tuples; m in (16, 32), kc | k, (kc / 32) % ap == 0, smem <= 227 KB."""
    src = open(SRC).read() if src is None else src
    src = src.replace("G19_CONFIGS", " ".join("G19_CASE(%d,%d,%d,%d,1)" % tuple(c) for c in sorted(set(map(tuple, cfgs)))))
    return load_inline(name="pj_g19_" + hashlib.md5(src.encode()).hexdigest()[:10],
                       cpp_sources="void g19_gemm(torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t); void g19_cvt(torch::Tensor, torch::Tensor);",
                       cuda_sources=src, functions=["g19_gemm", "g19_cvt"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])
