"""moe n5s33 BRA candidate (CORRECT output): tk2f_consume's per-row reduce barrier is waited on only by the 128 threads that
do the epilogue (warps 0-3); warps 4-7 `bar.arrive` and go straight on to the next row's stages. gemm2's epilogue moves to
128 threads (2 tiles each) so both phases have the same split. red stays double-buffered: a warp can only rewrite buffer
rb(r) at the end of row r+2, whose last stage the producer refilled only after warps 0-3 released row r+1's stage 3,
i.e. after they had read red(r) (mbarrier release/acquire chain through the ring).
python bra.py <r-layer submission.py> <outdir> -> <outdir>/bra.py (TUNE = {}, = n5s33 base0 otherwise)."""
import os, sys
L0 = open(sys.argv[1]).read()
out = sys.argv[2]; os.makedirs(out, exist_ok=True)


def sub(s, a, b, n=1):
    assert s.count(a) == n, (a, s.count(a))
    return s.replace(a, b)


def bra(s):
    s = sub(s, "\nTUNE_BASE = {", "\nTUNE = {}  # moe n5s33 A/B: defaults only\nTUNE_BASE = {")
    i = s.index("void tk2f_consume(")
    j = s.index("void tk2g_row(", i)
    body = s[i:j]
    body = sub(body, '    asm volatile("bar.sync 1, %0;" ::"n"(NC * 32) : "memory");\n',
               '    if (tid >= 128) asm volatile("bar.arrive 1, %0;" ::"n"(NC * 32) : "memory");  // moe n5s33 BRA\n'
               '    else asm volatile("bar.sync 1, %0;" ::"n"(NC * 32) : "memory");\n')
    a = """    } else if (NC == 8 || ct < 256) {
      const int ii = ct >> 7, r = (ct >> 5) & 3;
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < NC; ++w) v += red[((w * 2 + ii) * 4 + r) * 32 + l];
      const int tok = 2 * tt + (r & 1);
      if (tok < n) {"""
    b = """    } else if (ct < 128) {
#pragma unroll
      for (int ii = 0; ii < 2; ++ii) {
      const int r = (ct >> 5) & 3;
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < NC; ++w) v += red[((w * 2 + ii) * 4 + r) * 32 + l];
      const int tok = 2 * tt + (r & 1);
      if (tok < n) {"""
    body = sub(body, a, b)
    a = """        else reinterpret_cast<float*>(Out)[(size_t)(off + tok) * NOUT + col] = v;
      }
    }
"""
    body = sub(body, a, a + "      }\n")
    return s[:i] + body + s[j:]


open(f"{out}/bra.py", "w").write(bra(L0))
print("bra ok")
