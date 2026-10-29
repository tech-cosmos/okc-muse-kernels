# n7s3 boundary variants (T=256, NB = N, no PDL/RB/H32): fit 2+ row programs per SM (base: 97 regs x 512 thr = 1/SM, 2 waves)
import sys, os
import triton, triton.language as tl
sys.path.insert(0, os.path.expanduser("~/r-proj/muse_decode_layer"))
import submission as L
T, N = 256, 8192
EPS_POST, EPS = L.POST_NORM_EPS, L.RMS_EPS

@triton.jit
def _bnd_lazy(br_ptr, s_ptr, alpha_ptr, beta_ptr, bgain_ptr, gain_ptr, s_out_ptr, h_ptr,
              N: tl.constexpr, BGAIN: tl.constexpr, EPS_POST: tl.constexpr, EPS: tl.constexpr):
    """as _boundary (NB = N, SPL 1): the per-column vectors load only where used (fewer live registers)."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, N)
    s = tl.load(s_ptr + row * N + cols)
    b = tl.load(br_ptr + row * N + cols).to(tl.float32)
    if BGAIN:
        b = b * tl.load(bgain_ptr + cols)
    rb = tl.rsqrt(tl.sum(b * b, 0) / N + EPS_POST)
    s = tl.load(alpha_ptr + cols) * s + tl.load(beta_ptr + cols) * (b * rb)
    tl.store(s_out_ptr + row * N + cols, s)
    rs = tl.rsqrt(tl.sum(s * s, 0) / N + EPS)
    tl.store(h_ptr + row * N + cols, (s * rs * tl.load(gain_ptr + cols)).to(tl.bfloat16))

def base(nw, mr=None, bg=False):
    kw = {} if mr is None else dict(maxnreg=mr)
    return lambda a, s, al, be, g1, g2, so, h: L._boundary[(T,)](a, s, al, be, g2, g1, so, h, h, NB=N, N=N, BGAIN=bg,
        H32=False, EPS_POST=EPS_POST, EPS=EPS, PDL=False, RB=False, num_warps=nw, SPL=1, **kw)
def lazy(nw, mr=None, bg=False):
    kw = {} if mr is None else dict(maxnreg=mr)
    return lambda a, s, al, be, g1, g2, so, h: _bnd_lazy[(T,)](a, s, al, be, g2, g1, so, h, N=N, BGAIN=bg,
        EPS_POST=EPS_POST, EPS=EPS, num_warps=nw, **kw)
VARIANTS = {"b16r64": base(16, 64), "b32r32": base(32, 32), "b8r128": base(8, 128),
            "l16": lazy(16), "l16r64": lazy(16, 64), "l32": lazy(32), "l32r32": lazy(32, 32), "l8": lazy(8), "l8r128": lazy(8, 128),
            "l4": lazy(4), "l16r64g": lazy(16, 64, True)}
