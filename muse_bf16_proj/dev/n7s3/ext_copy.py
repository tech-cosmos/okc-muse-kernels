# copy-floor variants: same bytes as the boundary (read s FP32 + b BF16, write s' FP32 + h BF16), different grids
import triton, triton.language as tl
@triton.jit
def _copyc(b_ptr, s_ptr, so_ptr, h_ptr, N: tl.constexpr, C: tl.constexpr):
    p = tl.program_id(0).to(tl.int64); c = p * C + tl.arange(0, C)
    v = tl.load(s_ptr + c); b = tl.load(b_ptr + c).to(tl.float32)
    tl.store(so_ptr + c, v + b); tl.store(h_ptr + c, (v * b).to(tl.bfloat16))
@triton.jit
def _rdonly(b_ptr, s_ptr, so_ptr, N: tl.constexpr):
    row = tl.program_id(0).to(tl.int64); c = tl.arange(0, N)
    v = tl.load(s_ptr + row * N + c); b = tl.load(b_ptr + row * N + c).to(tl.float32)
    t = tl.sum(v * b, 0)
    tl.store(so_ptr + row * N, t)
T, N = 256, 8192
def mk(C, nw):
    return lambda a, s, al, be, g1, g2, so, h: _copyc[(T * N // C,)](a, s, so, h, N=N, C=C, num_warps=nw)
VARIANTS = {f"cp{C}w{nw}": mk(C, nw) for C, nw in [(2048, 4), (4096, 8), (1024, 4), (8192, 8), (16384, 32)]}
VARIANTS["rdonly"] = lambda a, s, al, be, g1, g2, so, h: _rdonly[(T,)](a, s, so, N=N, num_warps=16)
