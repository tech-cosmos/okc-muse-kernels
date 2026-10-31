"""moe o7 LUTLATE: moe_tk2f loads its dequant LUT into a register at entry and stores it to smem at the barrier after
the routing scans, so the routing's barriers don't wait on the LUT's cold DRAM trip. Bitwise. python mklut.py in out"""
import sys
s = open(sys.argv[1]).read()
ik = s.find('__launch_bounds__((NC + 1 + tk2f_rel_on<NC, DYN, FN, GS, MRG, PF>()) * 32, 1) moe_tk2f(')
assert ik > 0
A = '  for (int i = tid; i < 256; i += (NC + 1) * 32) slut[i] = lut[i];\n'
a = s.find(A, ik); assert a > 0 and a - ik < 3000
s = s[:a] + ('  static_assert((NC + 1) * 32 >= 256, "one LUT entry per thread");\n'
             '  const uint32_t lut_v = tid < 256 ? __ldg(lut + tid) : 0u;  // moe o7: stored to smem after the routing\n') + s[a + len(A):]
B = '    if (tid == 0) { s_rs13[0] = 0; s_rs2[0] = 0; }\n  }\n  __syncthreads();\n  const int n13 = s_rs13[nact], n2 = s_rs2[nact];\n'
b = s.find(B, a); assert b > 0 and b - a < 8000
s = s[:b] + B.replace('  __syncthreads();\n', '  if (tid < 256) slut[tid] = lut_v;\n  __syncthreads();\n') + s[b + len(B):]
open(sys.argv[2], 'w').write(s)
print("ok")
