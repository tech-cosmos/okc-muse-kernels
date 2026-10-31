"""moe o7 ORACLE (timing only): tk2f with a FREE routing prologue. g_ormode 0 normal | 1 compute + store the plan
(every CTA: shared plan by CTA 0, own split bounds) | 2 load the stored plan (same input = same plan) instead of the
routing block + split bounds. Host: lib.moe_orset(mode)."""
import sys
src, out = sys.argv[1], sys.argv[2]
s = open(src).read()
def rep(a, b, start=0, cnt=1):
    global s
    i = s.find(a, start); assert i >= 0, a[:70]
    s = s[:i] + b + s[i + len(a):]
    return i
ik = s.find('__launch_bounds__((NC + 1 + tk2f_rel_on<NC, DYN, FN, GS, MRG, PF>()) * 32, 1) moe_tk2f(')
assert ik > 0
# globals before the kernel template line
tl = s.rfind('template <', 0, ik)
s = s[:tl] + '''__device__ int g_ormode = 0;
__device__ int g_orp[1024];       // nact, s_e[32], s_off[33], s_rs13[33], s_rs2[33], s_srt[256], s_inv[256]
__device__ int g_orab[1024 * 4];
int64_t moe_orset(int64_t m) { int v = (int)m; cudaMemcpyToSymbol(g_ormode, &v, 4); cudaDeviceSynchronize(); return 0; }
''' + s[tl:]
ik = s.find('__launch_bounds__((NC + 1 + tk2f_rel_on<NC, DYN, FN, GS, MRG, PF>()) * 32, 1) moe_tk2f(')
ANC = '  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n  {  // routing, as moe_tk2 GATED\n'
a = s.find(ANC, ik); assert a > 0
pre = """  // ORACLE: mode, plan and own split bounds in ONE round trip (all loads issued before any use)
  int o_m, o_v0 = 0, o_v1 = 0, o_v2 = 0, o_v3 = 0, o_srt = 0, o_inv = 0;
  o_m = g_ormode;
  if (tid == 0) o_v0 = g_orp[0];
  if (tid < 32) o_v1 = g_orp[1 + tid];
  if (tid < 33) { o_v2 = g_orp[33 + tid]; o_v3 = g_orp[66 + tid]; o_srt = g_orp[99 + tid]; }
  int o_s[2] = {0, 0}, o_i[2] = {0, 0};
  for (int j = 0; j < 2; ++j) { const int i = tid + j * (NC + 1) * 32; if (i < 256) { o_s[j] = g_orp[132 + i]; if (FN) o_i[j] = g_orp[388 + i]; } }
  int o_ab = tid < 4 ? g_orab[blockIdx.x * 4 + tid] : 0;
  __shared__ int s_oab[4];
  const int ormode = o_m;
  if (ormode == 2) {
    if (tid == 0) s_nact = o_v0;
    if (tid < 32) s_e[tid] = o_v1;
    if (tid < 33) { s_off[tid] = o_v2; s_rs13[tid] = o_v3; s_rs2[tid] = o_srt; }
    for (int j = 0; j < 2; ++j) { const int i = tid + j * (NC + 1) * 32; if (i < 256) { s_srt[i] = o_s[j]; if (FN) s_inv[i] = o_i[j]; } }
    if (tid < 4) s_oab[tid] = o_ab;
    goto orc_done;
  }
"""
s = s[:a] + pre + s[a:]
E1 = "  }\n  __syncthreads();\n  const int nact = s_nact;\n"
e = s.find(E1, a); assert e > 0
s = s[:e] + "  }\norc_done:\n  __syncthreads();\n  const int nact = s_nact;\n" + s[e + len(E1):]
R1 = "  if (tid < 32) {\n    const int e = s_e[tid];\n    const int ng"
r = s.find(R1, e); assert r > 0 and r - e < 400
s = s[:r] + "  if (ormode != 2 && tid < 32) {\n    const int e = s_e[tid];\n    const int ng" + s[r + len(R1):]
b = s.find('  const int n13 = s_rs13[nact], n2 = s_rs2[nact];', r); assert b > 0
st = """  if (ormode == 1 && blockIdx.x == 0) {
    if (tid == 0) g_orp[0] = s_nact;
    if (tid < 32) g_orp[1 + tid] = s_e[tid];
    if (tid < 33) { g_orp[33 + tid] = s_off[tid]; g_orp[66 + tid] = s_rs13[tid]; g_orp[99 + tid] = s_rs2[tid]; }
    for (int i = tid; i < 256; i += (NC + 1) * 32) { g_orp[132 + i] = s_srt[i]; if (FN) g_orp[388 + i] = s_inv[i]; }
  }
"""
s = s[:b] + st + s[b:]
ik = s.find('__launch_bounds__((NC + 1 + tk2f_rel_on<NC, DYN, FN, GS, MRG, PF>()) * 32, 1) moe_tk2f(')
# split bounds: skip in mode 2
x = s.find('  int b0 = (int)((long long)n2 * blockIdx.x / gridDim.x), b1 = (int)((long long)n2 * (blockIdx.x + 1) / gridDim.x);\n', ik)
assert x > 0
x = s.find('\n', x) + 1
y = s.find('  if (PF & 32) {  // moe n8 BAL', x); assert y > 0
s = s[:x] + '  if (ormode == 2) { a0 = s_oab[0]; a1 = s_oab[1]; b0 = s_oab[2]; b1 = s_oab[3]; } else {\n' + s[x:y] + '  }\n  if (ormode == 1 && tid == 0) { g_orab[blockIdx.x * 4] = a0; g_orab[blockIdx.x * 4 + 1] = a1; g_orab[blockIdx.x * 4 + 2] = b0; g_orab[blockIdx.x * 4 + 3] = b1; }\n' + s[y:]
n = s.count('"int64_t moe_set_pdl(int64_t, int64_t);"'); assert n >= 1
s = s.replace('"int64_t moe_set_pdl(int64_t, int64_t);"', '"int64_t moe_set_pdl(int64_t, int64_t);" "int64_t moe_orset(int64_t);"')
s = s.replace('functions=["moe_gemm13",', 'functions=["moe_orset", "moe_gemm13",')
open(out, "w").write(s)
print("ok", n)
