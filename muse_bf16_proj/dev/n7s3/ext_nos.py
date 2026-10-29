# n7s3 'no stream1': bnd1 does not store the FP32 stream1 (8 MB at T=256); bnd2 recomputes it from (stream, o output a,
# res1 alpha/beta) with the same FP32 ops as bnd1 (bitwise the same s1), then does its own boundary.
import sys, os, torch
import triton, triton.language as tl
sys.path.insert(0, os.path.expanduser("~/r-proj/muse_decode_layer"))
import submission as L
T, N = 256, 8192
EPS_POST, EPS = L.POST_NORM_EPS, L.RMS_EPS

@triton.jit
def _bnd_n1(br_ptr, s_ptr, alpha_ptr, beta_ptr, gain_ptr, h_ptr, N: tl.constexpr, EPS_POST: tl.constexpr, EPS: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, N)
    alpha = tl.load(alpha_ptr + cols)
    beta = tl.load(beta_ptr + cols)
    gain = tl.load(gain_ptr + cols)
    s = tl.load(s_ptr + row * N + cols)
    b = tl.load(br_ptr + row * N + cols).to(tl.float32)
    rb = tl.rsqrt(tl.sum(b * b, 0) / N + EPS_POST)
    s = alpha * s + beta * (b * rb)
    rs = tl.rsqrt(tl.sum(s * s, 0) / N + EPS)
    tl.store(h_ptr + row * N + cols, (s * rs * gain).to(tl.bfloat16))

@triton.jit
def _bnd_r2(a_ptr, s0_ptr, a1_ptr, b1_ptr, f_ptr, alpha_ptr, beta_ptr, bgain_ptr, gain_ptr, s_out_ptr, h_ptr,
            N: tl.constexpr, EPS_POST: tl.constexpr, EPS: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, N)
    a1 = tl.load(a1_ptr + cols)
    b1 = tl.load(b1_ptr + cols)
    s = tl.load(s0_ptr + row * N + cols)
    b = tl.load(a_ptr + row * N + cols).to(tl.float32)
    rb = tl.rsqrt(tl.sum(b * b, 0) / N + EPS_POST)
    s = a1 * s + b1 * (b * rb)  # = bnd1's stream1, same ops
    alpha = tl.load(alpha_ptr + cols)
    beta = tl.load(beta_ptr + cols)
    gain = tl.load(gain_ptr + cols)
    bg = tl.load(bgain_ptr + cols)
    f = tl.load(f_ptr + row * N + cols).to(tl.float32) * bg
    rf = tl.rsqrt(tl.sum(f * f, 0) / N + EPS_POST)
    s = alpha * s + beta * (f * rf)
    tl.store(s_out_ptr + row * N + cols, s)
    rs = tl.rsqrt(tl.sum(s * s, 0) / N + EPS)
    tl.store(h_ptr + row * N + cols, (s * rs * gain).to(tl.bfloat16))

a_old = torch.randn(T, N, device="cuda").bfloat16()  # the o output, cold after the experts
def n1(nw):
    return lambda a, s, al, be, g1, g2, so, h: _bnd_n1[(T,)](a, s, al, be, g1, h, N=N, EPS_POST=EPS_POST, EPS=EPS, num_warps=nw)
def r2(nw):  # a = the (hot) unpool output here
    return lambda a, s, al, be, g1, g2, so, h: _bnd_r2[(T,)](a_old, s, al, be, a, be, al, g2, g1, so, h, N=N,
                                                            EPS_POST=EPS_POST, EPS=EPS, num_warps=nw)
VARIANTS = {"n1": n1(16), "n1w32": n1(32), "r2": r2(16), "r2w32": r2(32)}

@triton.jit
def _bnd_n1b(br_ptr, s_ptr, alpha_ptr, beta_ptr, gain_ptr, h_ptr, rb_ptr, N: tl.constexpr, EPS_POST: tl.constexpr,
             EPS: tl.constexpr):
    """n1 + stores rb (the branch's rsqrt, one FP32 per row) for bnd2."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, N)
    alpha = tl.load(alpha_ptr + cols)
    beta = tl.load(beta_ptr + cols)
    gain = tl.load(gain_ptr + cols)
    s = tl.load(s_ptr + row * N + cols)
    b = tl.load(br_ptr + row * N + cols).to(tl.float32)
    rb = tl.rsqrt(tl.sum(b * b, 0) / N + EPS_POST)
    tl.store(rb_ptr + row, rb)
    s = alpha * s + beta * (b * rb)
    rs = tl.rsqrt(tl.sum(s * s, 0) / N + EPS)
    tl.store(h_ptr + row * N + cols, (s * rs * gain).to(tl.bfloat16))

@triton.jit
def _bnd_r2b(a_ptr, rb_ptr, s0_ptr, a1_ptr, b1_ptr, f_ptr, alpha_ptr, beta_ptr, bgain_ptr, gain_ptr, s_out_ptr, h_ptr,
             N: tl.constexpr, EPS_POST: tl.constexpr, EPS: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, N)
    a1 = tl.load(a1_ptr + cols)
    b1 = tl.load(b1_ptr + cols)
    alpha = tl.load(alpha_ptr + cols)
    beta = tl.load(beta_ptr + cols)
    gain = tl.load(gain_ptr + cols)
    bg = tl.load(bgain_ptr + cols)
    s = tl.load(s0_ptr + row * N + cols)
    b = tl.load(a_ptr + row * N + cols).to(tl.float32)
    rb = tl.load(rb_ptr + row)
    s = a1 * s + b1 * (b * rb)  # = bnd1's stream1, same ops
    f = tl.load(f_ptr + row * N + cols).to(tl.float32) * bg
    rf = tl.rsqrt(tl.sum(f * f, 0) / N + EPS_POST)
    s = alpha * s + beta * (f * rf)
    tl.store(s_out_ptr + row * N + cols, s)
    rs = tl.rsqrt(tl.sum(s * s, 0) / N + EPS)
    tl.store(h_ptr + row * N + cols, (s * rs * gain).to(tl.bfloat16))

rbuf = torch.rand(T, device="cuda") + 0.5
VARIANTS["n1b"] = lambda a, s, al, be, g1, g2, so, h: _bnd_n1b[(T,)](a, s, al, be, g1, h, rbuf, N=N, EPS_POST=EPS_POST, EPS=EPS, num_warps=16)
VARIANTS["r2b"] = lambda a, s, al, be, g1, g2, so, h: _bnd_r2b[(T,)](a_old, rbuf, s, al, be, a, be, al, g2, g1, so, h, N=N,
                                                                    EPS_POST=EPS_POST, EPS=EPS, num_warps=16)
