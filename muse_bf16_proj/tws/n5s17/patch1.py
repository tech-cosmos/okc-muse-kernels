import sys
p = sys.argv[1]; s = open(p).read()
def rep(a, b, n=1):
    global s
    c = s.count(a)
    assert c == n, (c, a[:80])
    s = s.replace(a, b)
# ---- tw_item
i0 = s.index("__device__ __forceinline__ void tw_item(")
i1 = s.index("template <int RT, int KG, int NSTG, bool GATED, int NW>\n__global__ void __launch_bounds__(TWC<RT, GATED, NW>::THREADS, 1) moe_tw(")
t = s[i0:i1]
t = t.replace("__device__ __forceinline__ void tw_item(", "__device__ __forceinline__ bool tw_item(", 1)
a = "                                        int* s_flag = nullptr, int pofs = 0) {"
assert t.count(a) == 1
t = t.replace(a, """                                        int* s_flag = nullptr, int pofs = 0,
                                        // n5s17 TWS (work stealing): s_lastp != null = dynamic K range, the producer
                                        // marks each stage (s_lastp: 0 more, 1 last, 2 last + item was stolen from,
                                        // 3 last of a stolen piece; s_pcp: piece id | slot << 4); sw_* = the item's
                                        // state word, slot map, partial slots. Returns false = partial only (not the finisher).
                                        const int* s_lastp = nullptr, const int* s_pcp = nullptr,
                                        unsigned long long* sw_word = nullptr, int* sw_smap = nullptr,
                                        float* sw_part = nullptr) {""")
a = """  for (int kc = 0; kc < nkc; ++kc) {
    const char* sb = smem + st * (STRIDE ? STRIDE : C::STG);
    if (kc) mbar_wait(&full[st], ph);
"""
assert t.count(a) == 1
t = t.replace(a, """  int lastv = 0, pcv = 0;
  for (int kc = 0; s_lastp ? !lastv : kc < nkc; ++kc) {
    const char* sb = smem + st * (STRIDE ? STRIDE : C::STG);
    if (kc) mbar_wait(&full[st], ph);
    if (s_lastp) { lastv = s_lastp[st]; pcv = s_pcp[st]; }
""")
t = t.replace("    asm volatile(\"bar.sync %0, 64;\" ::\"r\"(1 + (warp >> 1)) : \"memory\");\n    return;\n  }",
              "    asm volatile(\"bar.sync %0, 64;\" ::\"r\"(1 + (warp >> 1)) : \"memory\");\n    return true;\n  }", 1)
a = """        if (ct == 0) asm volatile("st.release.gpu.global.b32 [%0], %1;" ::"l"(tick + 512), "r"(1) : "memory");
        return;"""
assert t.count(a) == 1
t = t.replace(a, a.replace("return;", "return false;"))
a = """  if constexpr (DYN) {
    if (part) {"""
assert t.count(a) == 1
t = t.replace(a, """  if constexpr (DYN) {
    if (s_lastp && lastv >= 2) {  // n5s17 TWS: this piece's FP32 partial -> its slot; the last finisher sums all pieces
      const int ct = warp * 32 + lane, pid = pcv & 15, slot = pcv >> 4;
      float* my = sw_part + (size_t)slot * 16384;
      if (ct == 0) sw_smap[pid] = slot;
#pragma unroll
      for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int i = 0; i < NA; ++i) __stcg(my + (r * NA + i) * 256 + ct, acc[r][i]);
      __threadfence();
      asm volatile("bar.sync 1, 256;" ::: "memory");
      if (ct == 0) {
        const unsigned long long o = atomicAdd(sw_word, 1ull << 24);
        const int lo = o & 255, hi = (o >> 8) & 255, np = (o >> 16) & 255, nd = (o >> 24) & 255;
        const int fin = lo >= hi && nd + 1 == np;
        if (fin) *sw_word = 0;  // last user: clean for the next call
        __threadfence();
        *s_flag = fin ? np : 0;
      }
      asm volatile("bar.sync 1, 256;" ::: "memory");
      const int npc = *s_flag;
      asm volatile("bar.sync 1, 256;" ::: "memory");  // s_flag free again
      if (!npc) return false;
#pragma unroll
      for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int i = 0; i < NA; ++i) acc[r][i] = 0.f;
      for (int q = 0; q < npc; ++q) {  // piece order: deterministic for a given split
        const float* src = sw_part + (size_t)__ldcg(sw_smap + q) * 16384;
#pragma unroll
        for (int r = 0; r < RT; ++r)
#pragma unroll
          for (int i = 0; i < NA; ++i) acc[r][i] += __ldcg(src + (r * NA + i) * 256 + ct);
      }
    }
    if (part) {""")
# final close of tw_item: add return true before the last closing brace
k = t.rstrip().rfind("}")
t = t[:k] + "  return true;\n}\n\n"
s = s[:i0] + t + s[i1:]
open(p, "w").write(s)
print("ok")
