s = open('/tmp/moe16/sub_g.py').read()
def rep(a, b, start=0):
    global s
    i = s.find(a, start); assert i >= 0, a
    s = s[:i] + b + s[i + len(a):]
rep('template <typename... KA, typename... AA>\nstatic int launch_k(', '''__device__ unsigned long long g_td[1024 * 8];
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define TD(k) g_td[blockIdx.x * 8 + (k)] = gtime()
int64_t moe_tdbg(torch::Tensor o) { cudaMemcpyFromSymbol(o.data_ptr(), g_td, 1024 * 8 * 8); return 0; }
template <typename... KA, typename... AA>
static int launch_k(''')
i = s.find('__device__ __forceinline__ void tk2h_consume(')
rep('''  for (int p = 0; p < 2; ++p) {
    const bool gated = p == 0;''', '''  for (int p = 0; p < 2; ++p) {
    if (tid == 0) TD(p ? 2 : 0);
    const bool gated = p == 0;''', i)
rep('''          } while (c < nd);''', '''          } while (c < nd);
          if (tid == 0 && R == r0) TD(3);''', i)
rep('''      if (GS == 16 && n > 8)
        tk2g_row''', '''      if (tid == 0 && !gated && R == r0) TD(4);
      if (GS == 16 && n > 8)
        tk2g_row''', i)
j = s.find('tk2h_consume<NC, NSTG, KPS, SA, GS, MRG == 2>(')
rep('''    return;
  }
  if constexpr (GS == 16) {''', '''    if (tid == 0) TD(5);
    return;
  }
  if constexpr (GS == 16) {''', j)
rep('"int64_t moe_set_pdl(int64_t, int64_t);"', '"int64_t moe_set_pdl(int64_t, int64_t);" "int64_t moe_tdbg(torch::Tensor);"')
rep('functions=["moe_gemm13",', 'functions=["moe_tdbg", "moe_gemm13",')
open('/tmp/moe16/sub_t.py', 'w').write(s)
