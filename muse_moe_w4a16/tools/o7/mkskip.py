"""moe o7b ORACLE (timing only, WRONG output): every CTA holding q+1 rows (q = rows / grid) skips its last gemm13 row
and its last gemm2 row (the skipped gemm13 row is still credited to its counter after phase 1, so nothing hangs)
= the row-quantization ('last wave') tail removed. A K-split of the extra rows can recover at most ~half of it."""
import sys
s = open(sys.argv[1]).read()
ik = s.find('__launch_bounds__((NC + 1 + tk2f_rel_on<NC, DYN, FN, GS, MRG, PF>()) * 32, 1) moe_tk2f(')
assert ik > 0
A = '  if constexpr ((PF & 4096) != 0) { a1 = a0; b1 = b0; }'
a = s.find(A, ik); assert a > 0
s = s[:a] + '''  int sk13 = -1;
  if constexpr (MRG == 0 && !DYN) {  // ORACLE skip
    const int q1 = n13 / (int)gridDim.x, q2 = n2 / (int)gridDim.x;
    if (q1 > 0 && a1 - a0 > q1) { a1 -= 1; sk13 = a1; }
    if (q2 > 0 && b1 - b0 > q2) b1 -= 1;
  }
''' + s[a:]
def credit():
    return '''  if (sk13 >= 0 && tid == 0) {  // ORACLE: credit the skipped gemm13 row
    int q = 0;
    while (s_rs13[q + 1] <= sk13) ++q;
    const int ss = s_rs13[q] + ((sk13 - s_rs13[q]) / 256) * 256;
    __threadfence();
    atomicAdd((PF & 32) ? cnt + 64 + ss / 256 : cnt, 1);
  }
'''
B1 = '''    tk2g_consume<256, NC, NSTG, false, KPS, SA>(slut, gs2, inter, Y, cnt, n13, b0, b1'''
b = s.find(B1, a); assert b > 0
s = s[:b] + credit() + s[b:]
B2 = '''  tk2f_consume<256, NC, NSTG, false, KPS, SA, PF>(slut, gs2, inter, Y, cnt, n13, b0, b1'''
b = s.find(B2, a); assert b > 0
s = s[:b] + credit() + s[b:]
open(sys.argv[2], "w").write(s)
print("ok")
