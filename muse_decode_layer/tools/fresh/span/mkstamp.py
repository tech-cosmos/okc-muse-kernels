"""fresh x8 (b): per-CTA %globaltimer stamps in the layer's moe_tk2f (in-layer, TK2FX path at T=1/8).
python tools/fresh/span/mkstamp.py [in] [out]  (defaults submission.py -> /tmp/fx8/sub_st.py). Timing copy only.
g_td[cta*16 + k]: 0 entry, 1 routed, 2 producer first TMA issue, 3 consumer past the TK2FX wait (gemm13 entry),
4 ph1 first stage full, 5 ph1 end (gemm2 entry), 6 cnt ok (all gemm13 rows published), 7 ph2 first stage full, 8 end."""
import os, sys
src = sys.argv[1] if len(sys.argv) > 1 else "submission.py"
out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/fx8/sub_st.py"
os.makedirs(os.path.dirname(out), exist_ok=True)
s = open(src).read()


def rep(a, b, start=0):
    global s
    i = s.find(a, start); assert i >= 0, a[:60]
    assert s.find(a, i + 1) < 0 or start, ("not unique", a[:60])
    s = s[:i] + b + s[i + len(a):]
    return i


rep('template <typename... KA, typename... AA>\nstatic int launch_k(', '''__device__ unsigned long long g_td[1024 * 16];
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define TD(k) g_td[blockIdx.x * 16 + (k)] = gtime()
int64_t moe_tdbg(torch::Tensor o) { cudaMemcpyFromSymbol(o.data_ptr(), g_td, 1024 * 16 * 8); return 0; }
template <typename... KA, typename... AA>
static int launch_k(''')
ic = s.find('__device__ __forceinline__ void tk2f_consume(const uint32_t* slut')
assert ic > 0
rep('''  uint4 xf[NQ][2];
  int a = 0, seg_end = -1, seg_start = 0, off = 0, n = 0, ci = 0;''', '''  uint4 xf[NQ][2];
  if (tid == 0) TD(GATED ? 3 : 5);
  int a = 0, seg_end = -1, seg_start = 0, off = 0, n = 0, ci = 0;''', start=ic)
rep('''      const bool v = g < n;
      int row = 0;''', '''      if (!GATED && tid == 0 && R == r0) TD(6);
      const bool v = g < n;
      int row = 0;''', start=ic)
rep('''      mbar_wait(&full[st], ph);
      const uint4* sw4 = wsm + st * (SB / 16);''', '''      mbar_wait(&full[st], ph);
      if (tid == 0 && R == r0 && q == 0) TD(GATED ? 4 : 7);
      const uint4* sw4 = wsm + st * (SB / 16);''', start=ic)
ik = s.find('__launch_bounds__((NC + 1 + tk2f_rel_on<NC, DYN, FN, GS, MRG, PF>()) * 32, 1) moe_tk2f(')
assert ik > 0
rep('''  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  int gen = 0;''', '''  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) TD(0);
  int gen = 0;''' + ('''\n  if (warp == NC && lane == 0) { asm volatile("griddepcontrol.wait;" ::: "memory"); TD(9); }''' if os.environ.get("WAITST") else ""), start=ik)
rep('''  const int n13 = s_rs13[nact], n2 = s_rs2[nact];
  if constexpr (DYN) {''', '''  const int n13 = s_rs13[nact], n2 = s_rs2[nact];
  if (tid == 0) TD(1);
  if constexpr (DYN) {''', start=ik)
rep('''      tk2f_produce<512, NC, NSTG, KPS, SQ, S6, (PF & 2048) != 0>(Wq13''', '''      TD(2);
      tk2f_produce<512, NC, NSTG, KPS, SQ, S6, (PF & 2048) != 0>(Wq13''', start=ik)
rep('''  if constexpr (FN) {
    s_rt[tid] = rt;''', '''  if (tid == 0) TD(8);
  if constexpr (FN) {
    s_rt[tid] = rt;''', start=ik)
# export moe_tdbg from every _moe__lib build
n = s.count('"int64_t moe_set_pdl(int64_t, int64_t);"')
assert n >= 1, n
s = s.replace('"int64_t moe_set_pdl(int64_t, int64_t);"', '"int64_t moe_set_pdl(int64_t, int64_t);" "int64_t moe_tdbg(torch::Tensor);"')
n = s.count('functions=["moe_gemm13",')
s = s.replace('functions=["moe_gemm13",', 'functions=["moe_tdbg", "moe_gemm13",')
open(out, "w").write(s)
print("ok", out, n)

# pr1 (T=1 pooler+router, g10 lib): g_pt[b*4 + 0] block entry, [b*4 + 1] thread 0's end, g_pt[4000] routing written
s = open(out).read()
a = s.index("_GPR1_SRC = r'''"); b = s.index("'''", a + 20)
k = s[a:b]
def kr(o, n):
    global k
    assert k.count(o) == 1, o[:50]; k = k.replace(o, n)
kr("namespace pr1 {\n", "namespace pr1 {\n__device__ unsigned long long g_pt[4096];\n__device__ __forceinline__ unsigned long long ptime() { unsigned long long t; asm volatile(\"mov.u64 %0, %%globaltimer;\" : \"=l\"(t)); return t; }\n")
kr("    if (lane == 0) *a.cnt = 0;\n", "    if (lane == 0) { *a.cnt = 0; g_pt[4000] = ptime(); }\n")
kr("""    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if ((int)blockIdx.x < RB) router<R, 256 * NV>(a, blockIdx.x, RB);
    else pooler<U, NV, 8192, PF>(a, blockIdx.x - RB, gridDim.x - RB);
""", """    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if (threadIdx.x == 0) g_pt[blockIdx.x * 4] = ptime();
    if ((int)blockIdx.x < RB) router<R, 256 * NV>(a, blockIdx.x, RB);
    else pooler<U, NV, 8192, PF>(a, blockIdx.x - RB, gridDim.x - RB);
    if (threadIdx.x == 0) { g_pt[blockIdx.x * 4 + 1] = ptime(); g_pt[blockIdx.x * 4 + 2] = (int)blockIdx.x < RB; }
""")
kr("    // ticket: the last router block does the routing\n", "    // ticket: the last router block does the routing\n    if (threadIdx.x == 0) g_pt[blockIdx.x * 4 + 3] = ptime();\n")
kr("    if (!last || wp) return;\n    __threadfence();\n", "    if (!last || wp) return;\n    if (lane == 0) g_pt[4001] = ptime();\n    __threadfence();\n    if (lane == 0) g_pt[4002] = ptime();\n")
kr("    int myid = 0; float mysc = 0.f, sum = 0.f;\n", "    if (lane == 0) g_pt[4003] = ptime();\n    int myid = 0; float mysc = 0.f, sum = 0.f;\n")
kr("    if (lane < TOPK) {\n        a.ids[lane] = myid;", "    if (lane == 0) g_pt[4004] = ptime();\n    if (lane < TOPK) {\n        a.ids[lane] = myid;")
k += "\nint64_t pr1_tdbg(torch::Tensor o) { cudaMemcpyFromSymbol(o.data_ptr(), pr1::g_pt, 4096 * 8); return 0; }\n"
s = s[:a] + k + s[b:]
assert s.count('+ "void bnd2_launch(') == 1
s = s.replace('+ "void bnd2_launch(', '+ "int64_t pr1_tdbg(torch::Tensor);\\n" + "void bnd2_launch(')
s = s.replace('"g19_gemm", "pr1_launch"]', '"g19_gemm", "pr1_launch", "pr1_tdbg"]')
open(out, "w").write(s)
print("pr1 stamps ok")

# ORACLE (runtime, default off = mode 0): bit 0 hoist bias / ls into smem before pr1's griddepcontrol.wait (routing's
# 8 us DRAM trip under the pooler stream); mode >> 1 = P: the routing warp L2-prefetches the first P w13 rows (gate
# 32 KB + up 32 KB + scales 8 KB, non-SQ T=1 layout) of every tk2f CTA's even split right after writing ids.
s = open(out).read()
a = s.index("_GPR1_SRC = r'''"); b = s.index("'''", a + 20)
k = s[a:b]
kr("__device__ unsigned long long g_pt[4096];\n", "__device__ unsigned long long g_pt[4096];\n__device__ int g_pfm = 0;\n__device__ const char* g_pw13 = nullptr;\n__device__ const char* g_ps2 = nullptr;\n")
kr("""    if (g < NG) load(g);
    pf(g + RB);
    asm volatile("griddepcontrol.wait;" ::: "memory");
""", """    if (g < NG) load(g);
    pf(g + RB);
    __shared__ float sbias[E];
    __shared__ int sls;
    const int pfm = g_pfm;
    if (pfm & 1) {
        if (t < E) sbias[t] = __ldg(a.bias + t);
        if (t == E) sls = __ldg(a.ls);
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
""")
kr("        key[i] = sc[i] + __ldg(a.bias + e);\n", "        key[i] = sc[i] + ((pfm & 1) ? sbias[e] : __ldg(a.bias + e));\n")
kr("        int loc = (myid - __ldg(a.ls) + E) % E;\n", "        int loc = (myid - ((pfm & 1) ? sls : __ldg(a.ls)) + E) % E;\n")
kr("    if (lane == 0) { *a.cnt = 0; g_pt[4000] = ptime(); }\n", """    if (lane == 0) { *a.cnt = 0; g_pt[4000] = ptime(); }
    if (pfm >> 1) {
        const int P = pfm >> 1;
        int loc = -1;
        if (lane < TOPK) { const int l = (myid - ((pfm & 1) ? sls : __ldg(a.ls)) + E) % E; loc = l < a.local ? l : -1; }
        const unsigned vm = __ballot_sync(0xffffffffu, loc >= 0);
        const int ne = __popc(vm);
        int rank = 0;
        for (int j = 0; j < TOPK; ++j) { const int o = __shfl_sync(0xffffffffu, loc, j); rank += (loc >= 0 && o >= 0 && o < loc); }
        int es[TOPK];
#pragma unroll
        for (int q = 0; q < TOPK; ++q) {
            const unsigned m = __ballot_sync(0xffffffffu, loc >= 0 && rank == q);
            es[q] = m ? __shfl_sync(0xffffffffu, loc, __ffs(m) - 1) : 0;
        }
        const int n13 = ne * 256;
        for (int cb = lane; cb < 132 && ne; cb += 32) {
            const int r0 = (int)((long long)n13 * cb / 132), r1 = (int)((long long)n13 * (cb + 1) / 132);
            for (int R = r0; R < r1 && R < r0 + P; ++R) {
                int e = es[0];
#pragma unroll
                for (int q = 1; q < TOPK; ++q) if (R / 256 == q) e = es[q];
                const int lr = R % 256;
                const char* gp = g_pw13 + ((size_t)e * 512 + lr) * 32768;
                const char* sp = g_ps2 + ((size_t)e * 256 + lr) * 8192;
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(gp), "r"(32768) : "memory");
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(gp + (size_t)256 * 32768), "r"(32768) : "memory");
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(sp), "r"(8192) : "memory");
            }
        }
    }
""")
k += "\nint64_t pr1_set_pf(int64_t m, int64_t w13, int64_t s2) { int mm = (int)m; const char* p = (const char*)w13; const char* q = (const char*)s2;\n  cudaMemcpyToSymbol(pr1::g_pfm, &mm, 4); cudaMemcpyToSymbol(pr1::g_pw13, &p, 8); cudaMemcpyToSymbol(pr1::g_ps2, &q, 8); return 0; }\n"
s = s[:a] + k + s[b:]
s = s.replace('+ "int64_t pr1_tdbg(torch::Tensor);\\n"', '+ "int64_t pr1_tdbg(torch::Tensor); int64_t pr1_set_pf(int64_t, int64_t, int64_t);\\n"')
s = s.replace('"pr1_launch", "pr1_tdbg"]', '"pr1_launch", "pr1_tdbg", "pr1_set_pf"]')
open(out, "w").write(s)
print("oracle ok")

# ORACLE: N2U_NOIDF (timing diagnostic of layer n7s8: n2u keeps its base routing path under IDF)
s = open(out).read()
o = 'list(bnd) + ([state["idf"]] if idf else [])'
assert s.count(o) == 1
s = s.replace(o, 'list(bnd) + ([state["idf"]] if idf and not N2U_NOIDF else [])')
s = s.replace("\nIDF_DEFAULT = False\n", "\nIDF_DEFAULT = False\nN2U_NOIDF = False\n", 1)
open(out, "w").write(s)
print("n2u diag ok")
