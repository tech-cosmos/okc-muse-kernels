"""fresh x12: per-CTA %globaltimer stamps of the layer's moe_tk2f (timing copy only). Stamps go to the state's cnt
buffer (grown to 128 + 132 * 32 ints) at int offset 128 + cta * 32 + 2 k (u64): 0 entry, 1 gemm13 start (past the
TK2FX wait), 5 gemm2 entry, 6 first gemm2 wait ok (CW: chunks 0..2), 7 CW chunk 3 ok, 8 end, 9 CW helper last publish.
python tools/fresh/cw/stamp.py <submission.py> <out.py>"""
import sys
s = open(sys.argv[1]).read()


def rep(a, b, start=0):
    global s
    i = s.find(a, start); assert i >= 0, a[:70]
    s = s[:i] + b + s[i + len(a):]
    return i


rep('st["cnt"] = torch.zeros(128, dtype', 'st["cnt"] = torch.zeros(128 + 132 * 32, dtype')
D = ('__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }\n'
     '#define TS(k) (reinterpret_cast<unsigned long long*>(cnt + 128 + blockIdx.x * 32))[(k)] = gtime()\n')
rep('__shared__ __align__(8) uint64_t g_tkrel[8];', D + '__shared__ __align__(8) uint64_t g_tkrel[8];')
ic = s.find('__device__ __forceinline__ void tk2f_consume(')
rep('  uint4 xf[NQ][2];\n', '  uint4 xf[NQ][2];\n  if (tid == 0) TS(GATED ? 1 : 5);\n', ic)
cw = '      if (CW && !GATED) {  // chunks 0..2 now'
if cw in s:
    rep('        cp3 = cp + 3;\n        p3 = true;\n', '        cp3 = cp + 3;\n        p3 = true;\n        if (tid == 0 && R == r0) TS(6);\n', ic)
    rep('          p3 = false;\n', '          p3 = false;\n          if (tid == 0 && R == r0) TS(7);\n', ic)
else:
    rep('      const bool v = g < n;\n      int row = 0;', '      if (!GATED && tid == 0 && R == r0) TS(6);\n      const bool v = g < n;\n      int row = 0;', ic)
# end of consume: the closing of the R loop is followed by "}\n\n// s10 GS16"
rep('\n  }\n}\n\n// s10 GS16', '\n  }\n  if (!GATED && tid == 0) TS(8);\n}\n\n// s10 GS16', ic)
if 'tk2f_cw_helper(int* cnt' in s:
    i = s.find('__device__ __forceinline__ void tk2f_cw_helper(int* cnt')
    rep('      np = 0;\n      ++k;\n', '      np = 0;\n      ++k;\n      TS(9);\n', i)
i = s.find('  int gen = 0;  // FN: this call\'s generation')
rep('  int gen = 0;', '  if (threadIdx.x == 0) (reinterpret_cast<unsigned long long*>(cnt + 128 + blockIdx.x * 32))[0] = gtime();\n  int gen = 0;', i)
open(sys.argv[2], 'w').write(s)
