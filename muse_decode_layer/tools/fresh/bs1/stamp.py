"""fresh x11: per-CTA %globaltimer stamps in standalone muse_moe_w4a16 (moe_tk2f non-FP/non-FN path + moe_norm2/3).
Timing copy only. python tools/fresh/bs1/stamp.py <moe submission.py> <out.py>
g_td[cta*16 + k]: 0 entry, 1 routed, 2 producer first TMA issue, 4 ph1 first stage full, 5 ph1 end (gemm2 entry),
6 cnt ok (all gemm13 rows published), 7 ph2 first stage full, 8 end. Row 1000 (moe_norm2 block 0, tid 0): 0 entry,
1 past griddepcontrol.wait, 2 Y loaded, 3 end."""
import sys
src, out = sys.argv[1], sys.argv[2]
s = open(src).read()


def rep(a, b, start=0, once=True):
    global s
    i = s.find(a, start); assert i >= 0, a[:60]
    assert not once or s.find(a, i + 1) < 0, ("not unique", a[:60])
    s = s[:i] + b + s[i + len(a):]
    return i


D = ('__device__ unsigned long long g_td[1024 * 16];\n__device__ __forceinline__ unsigned long long gtime() '
     '{ unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }\n'
     '#define TD(k) g_td[blockIdx.x * 16 + (k)] = gtime()\n'
     'int64_t moe_tdbg(torch::Tensor o) { cudaMemcpyFromSymbol(o.data_ptr(), g_td, 1024 * 16 * 8); return 0; }\n')
a = "template <bool SLOT = false>\n__global__ void __launch_bounds__(512) moe_norm2("
n0 = rep(a, D + a)
rep('  asm volatile("griddepcontrol.wait;" ::: "memory");  // PDL secondary of gemm2\n',
    '  if (tok == 0 && tid == 0) g_td[1000 * 16 + 0] = gtime();\n  asm volatile("griddepcontrol.wait;" ::: "memory");  // PDL secondary of gemm2\n  if (tok == 0 && tid == 0) g_td[1000 * 16 + 1] = gtime();\n', n0, False)
rep('  __syncthreads();\n  float acc[8];', '  if (tok == 0 && tid == 0) g_td[1000 * 16 + 2] = gtime();\n  __syncthreads();\n  float acc[8];', n0, False)
rep('  *reinterpret_cast<uint4*>(out + (size_t)tok * NOUT + tid * 8) = o;\n}',
    '  *reinterpret_cast<uint4*>(out + (size_t)tok * NOUT + tid * 8) = o;\n  if (tok == 0 && tid == 0) g_td[1000 * 16 + 3] = gtime();\n}', n0, False)
ic = s.find('__device__ __forceinline__ void tk2f_consume(const uint32_t* slut'); assert ic > 0
rep('''  uint4 xf[NQ][2];
  int a = 0, seg_end = -1, seg_start = 0, off = 0, n = 0, ci = 0;''', '''  uint4 xf[NQ][2];
  if (!GATED && tid == 0) TD(5);
  int a = 0, seg_end = -1, seg_start = 0, off = 0, n = 0, ci = 0;''', ic, False)
rep('''      const bool v = g < n;
      int row = 0;''', '''      if (!GATED && tid == 0 && R == r0) TD(6);
      const bool v = g < n;
      int row = 0;''', ic, False)
rep('''      mbar_wait(&full[st], ph);
      const uint4* sw4 = wsm + st * (SB / 16);''', '''      mbar_wait(&full[st], ph);
      if (tid == 0 && R == r0 && q == 0) TD(GATED ? 4 : 7);
      const uint4* sw4 = wsm + st * (SB / 16);''', ic, False)
ik = s.find('__launch_bounds__((NC + 1 + tk2f_rel_on<NC, DYN, FN, GS, MRG, PF>()) * 32, 1) moe_tk2f('); assert ik > 0
rep('''  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  constexpr bool FP = (PF & 16) != 0;''', '''  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) TD(0);
  constexpr bool FP = (PF & 16) != 0;''', ik, False)
rep('''  const int n13 = s_rs13[nact], n2 = s_rs2[nact];''', '''  const int n13 = s_rs13[nact], n2 = s_rs2[nact];
  if (tid == 0) TD(1);''', ik)
rep('''      tk2f_produce<512, NC, NSTG, KPS, SQ, S6, S6P>(Wq13, S13, a0, a1, s_rs13, s_e, smem, full, empty, st, ph, i, nullptr, 0, l2hv, nullptr, PFD);''',
    '''      TD(2);
      tk2f_produce<512, NC, NSTG, KPS, SQ, S6, S6P>(Wq13, S13, a0, a1, s_rs13, s_e, smem, full, empty, st, ph, i, nullptr, 0, l2hv, nullptr, PFD);''', ik)
rep('''  if constexpr (FN) {
    s_rt[tid] = rt;''', '''  if (tid == 0) TD(8);
  if constexpr (FN) {
    s_rt[tid] = rt;''', ik)
n = s.count('"int64_t moe_set_pdl(int64_t, int64_t);"'); assert n >= 1, n
s = s.replace('"int64_t moe_set_pdl(int64_t, int64_t);"', '"int64_t moe_set_pdl(int64_t, int64_t);" "int64_t moe_tdbg(torch::Tensor);"')
assert s.count('functions=["moe_gemm13",') >= 1
s = s.replace('functions=["moe_gemm13",', 'functions=["moe_tdbg", "moe_gemm13",')
open(out, "w").write(s)
print("ok", out)
