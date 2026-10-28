# n6s3: port TWM_EPI (staged + rolled tw_item epilogue, rolled tw_epi; bitwise = before) into a moe_src.py copy (r-layer).
# usage: python tools/n6s3/patch_layer_epi.py <r-moe submission.py @ >= n6s3> <moe_src.py in> <moe_src.py out>
# Takes tw_epi / tw_epi2 / tw_item's epilogue block verbatim from the moe submission and sets #define TWM_EPI 1.
import sys, re
m = open(sys.argv[1]).read(); s = open(sys.argv[2]).read()
def block(t, start, end):
    a = t.index(start); b = t.index(end, a); return a, b
# 1. tw_epi (whole function) + tw_epi2: from 'template <int RT, int N, bool GATED, int NW>\n__device__ __forceinline__ void tw_epi('
#    up to the DYN comment line
A = 'template <int RT, int N, bool GATED, int NW>\n__device__ __forceinline__ void tw_epi('
E = '// DYN: one main loop for both GEMMs (moe_twf), epilogue picked by g2 at run time\n'
a, b = block(m, A, E); new = m[a:b]
a2, b2 = block(s, A, E); s = s[:a2] + new + s[b2:]
# 2. tw_item's DYN epilogue call
old = '''    if (g2) tw_epi<RT, N, false, NW>(acc, lane, warp, nv, off, rb, gsc, sroutes, Out);
    else tw_epi<RT, N, true, NW>(acc, lane, warp, nv, off, rb, gsc, sroutes, Out);
'''
a, b = block(m, '#if TWM_EPI\n    __nv_bfloat16* stg', '#endif\n')
assert s.count(old) == 1; s = s.replace(old, m[a:b + len('#endif\n')])
# 3. smem: + 8 KB for twf / hot / twm
x = '  constexpr int SMEM = NSTG * TWC<RT, true, 8>::STG + 1024;\n'
assert s.count(x) == 1, x; s = s.replace(x, x[:-2] + ' + (TWM_EPI ? 8192 : 0);\n')
hs = re.findall(r'  constexpr int SMEM = NSTG \* \((?:\(NH > 64 \? 128 : NH\)|NH) \* 128 \+ TWC<2, true, 8>::WB \+ TWC<2, true, 8>::SB\) \+ 1024;\n', s)
assert len(hs) == 1, hs; s = s.replace(hs[0], hs[0][:-2] + ' + (TWM_EPI ? 8192 : 0);\n')
i = s.index('static int64_t twm_go('); x = '  constexpr int SMEM = NSTG * TWC<4, true, 8>::STG + 1024;\n'
j = s.index(x, i); s = s[:j] + x[:-2] + ' + (TWM_EPI ? 8192 : 0);\n' + s[j + len(x):]
# 4. the define
x = '#define TK2F_L2H '
i = s.index(x); k = s.index('\n', i) + 1
s = s[:k] + '#ifndef TWM_EPI\n#define TWM_EPI 1        // n6s3 (moe a16e336+): staged, rolled tw_item / tw_epi epilogues (i-cache); bitwise\n#endif\n' + s[k:]
open(sys.argv[3], 'w').write(s)
print('ok: tw_epi/tw_epi2 replaced, tw_item epilogue, 3 SMEM sites, define')
