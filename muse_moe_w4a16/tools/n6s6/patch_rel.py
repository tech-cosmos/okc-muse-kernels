"""n6s6 TK2F_REL: tk2f/tk2h gemm13 publishes by a helper warp (NC + 1), as twm's TWM_REL.
Usage: python patch_rel.py <file> [define value, default 0] (in place; works on moe submission.py and the layer's embedded moe source)."""
import sys, re
p = sys.argv[1]
s = open(p).read()
n0 = len(s)

def rep(old, new, cnt=1):  # cnt None: every occurrence (>= 1; the layer's python anchors repeat CUDA lines)
    global s
    c = s.count(old)
    assert c == cnt if cnt is not None else c >= 1, (c, cnt, old[:120])
    s = s.replace(old, new)

# 1. define
v = sys.argv[2] if len(sys.argv) > 2 else "0"  # layer copies: hard-coded value (python TK2F_REL in moe's submission.py)
m = re.findall(r"\n#define TWM_REL [01] ", s)
assert len(m) == 1, m
s = s.replace(m[0], f"\n#define TK2F_REL {v}       // n6s6: tk2f/tk2h gemm13 publishes by a helper warp NC + 1 (python TK2F_REL; 2 = tk2h only)" + m[0])

# 2. shared barriers + helper, before tk2f_consume
helper = r'''// n6s6 TK2F_REL: the consumers' gemm13 publishes (bar.sync 2 + thread-0 gpu fence + atomicAdd at each segment end)
// move to a helper warp (NC + 1): consumers arrive on g_tkrel[k & 7] (release.cta) and go on with the next row. gemm13
// segments are 256 rows on 256-aligned row-space boundaries (tk2f GS 8 and tk2h), so the helper derives the k-th publish
// (segment a0 / 256 + k, its row count, its counter) from the CTA's rows [a0, a1) alone. Its mbarrier wait (acquire)
// + gpu fence (cumulative) order every consumer's inter stores before the count, as bar.sync + thread-0 fence did.
__shared__ __align__(8) uint64_t g_tkrel[8];
template <int NC, int DYN, int FN, int GS, int MRG, int PF>
__host__ __device__ constexpr bool tk2f_rel_on() {
  return TK2F_REL && (TK2F_REL != 2 || MRG != 0) && NC == 8 && !DYN && !FN && !(PF & 16) && !(PF & 524288) && !(PF & 65536) && !(PF & 256) &&
         (MRG == 1 || MRG == 2 || (GS == 8 && MRG == 0));
}
__device__ __forceinline__ void tk2f_rel_arrive(int seg_start, int r0) { mbar_arrive(&g_tkrel[((seg_start >> 8) - (r0 >> 8)) & 7]); }
template <bool SEGC>
__device__ __forceinline__ void tk2f_rel_helper(int* cnt, int a0, int a1) {
  if (a0 >= a1) return;
  const int s0 = a0 >> 8, s1 = (a1 - 1) >> 8;
  for (int s = s0; s <= s1; ++s) {
    const int k = s - s0;
    mbar_wait(&g_tkrel[k & 7], (k >> 3) & 1);
    __threadfence();
    atomicAdd(SEGC ? cnt + 64 + s : cnt, min(a1, (s + 1) * 256) - max(a0, s * 256));
  }
}

template <int NRT, int NC, int NSTG, bool GATED, int KPS, bool SA = false, int PF = 0, bool REL = false>
__device__ __forceinline__ void tk2f_consume('''
rep("template <int NRT, int NC, int NSTG, bool GATED, int KPS, bool SA = false, int PF = 0>\n__device__ __forceinline__ void tk2f_consume(", helper)
# tk2f_consume publish
rep("""      if (R + 1 == seg_end || R + 1 == r1) {  // publish this CTA's rows of the segment
        asm volatile("bar.sync 2, %0;" ::"n"(NC * 32) : "memory");
        if (tid == 0) {
          __threadfence();
          atomicAdd(BAL ? cnt + 64 + seg_start / RPE : cnt, R + 1 - max(seg_start, r0));
        }
      }""", """      if (R + 1 == seg_end || R + 1 == r1) {  // publish this CTA's rows of the segment
        if constexpr (REL) {
          tk2f_rel_arrive(seg_start, r0);
        } else {
        asm volatile("bar.sync 2, %0;" ::"n"(NC * 32) : "memory");
        if (tid == 0) {
          __threadfence();
          atomicAdd(BAL ? cnt + 64 + seg_start / RPE : cnt, R + 1 - max(seg_start, r0));
        }
        }
      }""")
# tk2h
rep("template <int NC, int NSTG, int KPS, bool SA, int GS, bool PSC = false, bool SLOT = false, bool S6 = false>\n__device__ __forceinline__ void tk2h_consume(",
    "template <int NC, int NSTG, int KPS, bool SA, int GS, bool PSC = false, bool SLOT = false, bool S6 = false, bool REL = false>\n__device__ __forceinline__ void tk2h_consume(")
rep("""      if (gated && (R + 1 == seg_end || R + 1 == r1)) {  // publish this CTA's rows of the segment
        asm volatile("bar.sync 2, %0;" ::"n"(NC * 32) : "memory");
        if (tid == 0) {
          __threadfence();
          atomicAdd(PSC ? cnt + 64 + seg_start / 256 : cnt, R + 1 - max(seg_start, r0));
        }
      }""", """      if (gated && (R + 1 == seg_end || R + 1 == r1)) {  // publish this CTA's rows of the segment
        if constexpr (REL) {
          tk2f_rel_arrive(seg_start, r0);
        } else {
        asm volatile("bar.sync 2, %0;" ::"n"(NC * 32) : "memory");
        if (tid == 0) {
          __threadfence();
          atomicAdd(PSC ? cnt + 64 + seg_start / 256 : cnt, R + 1 - max(seg_start, r0));
        }
        }
      }""")
# kernel launch bounds
rep("__global__ void __launch_bounds__((NC + 1) * 32, 1) moe_tk2f(",
    "__global__ void __launch_bounds__((NC + 1 + tk2f_rel_on<NC, DYN, FN, GS, MRG, PF>()) * 32, 1) moe_tk2f(")
# init
rep("""    for (int i = 0; i < 4; ++i) mbar_init(&ridb[i], 1);""", """    for (int i = 0; i < 4; ++i) mbar_init(&ridb[i], 1);
    if constexpr (tk2f_rel_on<NC, DYN, FN, GS, MRG, PF>())
      for (int i = 0; i < 8; ++i) mbar_init(&g_tkrel[i], NC * 32);""")
# helper branch before the producer branch (static path)
rep("""  int st = 0, ph = 0;
  if (warp == NC) {
    if (lane == 0) {
      int i = 0;
      constexpr bool SQ""", """  constexpr bool REL = tk2f_rel_on<NC, DYN, FN, GS, MRG, PF>();
  if constexpr (REL) {
    if (warp == NC + 1) {  // n6s6 TK2F_REL helper
      if (lane == 0) tk2f_rel_helper<MRG ? MRG == 2 : (PF & 32) != 0>(cnt, a0, a1);
      return;
    }
  }
  int st = 0, ph = 0;
  if (warp == NC) {
    if (lane == 0) {
      int i = 0;
      constexpr bool SQ""")
# consumer call sites
rep("tk2h_consume<NC, NSTG, KPS, SA, GS, MRG == 2, (PF & 4) != 0, (PF & 1024) != 0>(",
    "tk2h_consume<NC, NSTG, KPS, SA, GS, MRG == 2, (PF & 4) != 0, (PF & 1024) != 0, REL>(", None)
rep("tk2f_consume<512, NC, NSTG, true, KPS, SA, PF>(slut, gs13, X, inter, cnt, n13, a0, a1,",
    "tk2f_consume<512, NC, NSTG, true, KPS, SA, PF, REL>(slut, gs13, X, inter, cnt, n13, a0, a1,", None)
# launch dims
rep("return launch_k(moe_tk2f<NC, NSTG, KPS, DYN, FN, SA, GS, MRG, PF>, grid, (NC + 1) * 32, SMEM,",
    "return launch_k(moe_tk2f<NC, NSTG, KPS, DYN, FN, SA, GS, MRG, PF>, grid, (NC + 1 + tk2f_rel_on<NC, DYN, FN, GS, MRG, PF>()) * 32, SMEM,", None)
open(p, "w").write(s)
print("patched", p, len(s) - n0)
