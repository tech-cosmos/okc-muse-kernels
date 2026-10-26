# n6: GS16 merged consumer (tk2h) with PF4 (slot-ordered pre-scaled y32 -> moe_norm3) + the n4s5 warp-parallel weighted
# split, for a LAYER moe_src.py (r-layer 69ab934's copy; layer.py's own TK2FX/TK2FJ patches unchanged):
#   python patch_layer_h16.py <moe_src.py>        (edits the file in place)
# Adds: #define TK2H_P4 1 (only the SA tk2h PF4 kernels are built: GS8 cfg 130004, GS16 cfg 14w004), python TK2H_MIN /
# TK2H_PF4, griddepcontrol.wait before tk2h_consume (TK2FX/TK2FJ: tk2f is a PDL secondary; no-op otherwise) and __ldcg
# for its x loads (x = the PDL predecessor's output). Enable per call (e.g. a layer TUNE option at T = 16/32):
#   _moe_TK2H_M[T] = 3 (= GS16, 2-group weight 11/8; clear() restores the default); or globally:
#   _moe_TK2H = 16; _moe_TK2H_W = 3; _moe_TK2H_MIN = 9; _moe_TK2H_PF4 = 1    (TK2H = 0 restores the default path)
import sys
p = sys.argv[1]; s = open(p).read()
def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (old[:100], s.count(old)); s = s.replace(old, new)
BOUND = "// n4s5: tk2g_split by warp 0 in 32-bit lane-parallel math (one lane per active expert; same bounds as tk2g_split).\n// tk2g_split ran 64-bit loops in every thread: consumers started 1.8 (median) / 4.1 (max) us late at d16 (G88).\n__device__ __forceinline__ int tk2g_bound(int b, int nact, int wt, int rps, const int* s_e, const int* s_off) {\n  const int lane = threadIdx.x & 31;\n  const int c = lane < nact ? s_off[s_e[lane] + 1] - s_off[s_e[lane]] : 0;\n  int w = 0;  // this expert's weight / rps\n  for (int o = 0; o < c; o += 16) w += c - o > 8 ? wt : 8;\n  const int ng = (c + 15) >> 4;\n  int incl = w * rps, rows = ng * rps;\n#pragma unroll\n  for (int o = 1; o < 32; o <<= 1) {\n    const int v = __shfl_up_sync(0xffffffffu, incl, o), r = __shfl_up_sync(0xffffffffu, rows, o);\n    if (lane >= o) { incl += v; rows += r; }\n  }\n  const int W = __shfl_sync(0xffffffffu, incl, 31);\n  const int p = (int)((long long)W * b / gridDim.x);\n  int cum = incl - w * rps, row = rows - ng * rps;\n  int res = 0x7fffffff;\n  if (ng && p >= cum && p < incl) {\n    for (int o = 0; o < c; o += 16) {\n      const int sw = c - o > 8 ? wt : 8;\n      if (p < cum + rps * sw) { res = row + (p - cum + sw - 1) / sw; break; }\n      cum += rps * sw;\n      row += rps;\n    }\n  }\n  if (p >= W) res = __shfl_sync(0xffffffffu, rows, 31);\n  return __reduce_min_sync(0xffffffffu, res);\n}\n\n"
rep("// s12 WEIGHTED static split (GS 8)", BOUND + "// s12 WEIGHTED static split (GS 8)")
rep("""    a0 = tk2g_split(blockIdx.x, nact, D, 256, s_e, s_off);
    a1 = tk2g_split(blockIdx.x + 1, nact, D, 256, s_e, s_off);
    b0 = tk2g_split(blockIdx.x, nact, D, 128, s_e, s_off);
    b1 = tk2g_split(blockIdx.x + 1, nact, D, 128, s_e, s_off);
""", """    __shared__ int s_gb[4];  // n4s5: warp 0, lane-parallel (tk2g_bound)
    if (warp == 0) {
      const int v0 = tk2g_bound(blockIdx.x, nact, D, 256, s_e, s_off), v1 = tk2g_bound(blockIdx.x + 1, nact, D, 256, s_e, s_off);
      const int v2 = tk2g_bound(blockIdx.x, nact, D, 128, s_e, s_off), v3 = tk2g_bound(blockIdx.x + 1, nact, D, 128, s_e, s_off);
      if (lane == 0) { s_gb[0] = v0; s_gb[1] = v1; s_gb[2] = v2; s_gb[3] = v3; }
    }
    __syncthreads();
    a0 = s_gb[0]; a1 = s_gb[1]; b0 = s_gb[2]; b1 = s_gb[3];
""")
rep("""template <int NRT, int NC, int NSTG, bool GATED_, int KPS, bool SA, bool TWO, bool RTG = false>
__device__ __forceinline__ void tk2g_row(const uint32_t* slut, const uint4 (&xf)[64 / NC][2], const uint4* xs, void* Out,
                                         int rp, int off, int n, float gsc, float* s_red, uint4* smem, uint64_t* full,
                                         uint64_t* empty, int& st, int& ph, int& rb, bool gated_rt = false) {""",
"""template <int NRT, int NC, int NSTG, bool GATED_, int KPS, bool SA, bool TWO, bool RTG = false, bool SLOT = false>
__device__ __forceinline__ void tk2g_row(const uint32_t* slut, const uint4 (&xf)[64 / NC][2], const uint4* xs, void* Out,
                                         int rp, int off, int n, float gsc, float* s_red, uint4* smem, uint64_t* full,
                                         uint64_t* empty, int& st, int& ph, int& rb, bool gated_rt = false,
                                         const int* s_srt = nullptr) {
  // SLOT (moe n6, = tk2f PF & 4): gemm2 Y row = route slot s_srt[off + tok], pre-scaled by gsc (= gs2 * 256): moe_norm3""")
rep("""      for (int w = 0; w < NC; ++w) v += red[(((h * NC + w) * 2 + ii) * 4 + r) * 32 + l];
      const int tok = h * 8 + 2 * tt + (r & 1);
      if (tok < n) {
        const int col = (rp + ii * RPE) * 16 + gg + ((r & 2) ? 8 : 0);
        reinterpret_cast<float*>(Out)[(size_t)(off + tok) * NOUT + col] = v;""",
"""      for (int w = 0; w < NC; ++w) v += red[(((h * NC + w) * 2 + ii) * 4 + r) * 32 + l];
      const int tok = h * 8 + 2 * tt + (r & 1);
      if (tok < n) {
        const int col = (rp + ii * RPE) * 16 + gg + ((r & 2) ? 8 : 0);
        if (SLOT) reinterpret_cast<float*>(Out)[(size_t)s_srt[off + tok] * NOUT + col] = v * gsc;
        else reinterpret_cast<float*>(Out)[(size_t)(off + tok) * NOUT + col] = v;""")
rep("""template <int NC, int NSTG, int KPS, bool SA, int GS, bool PSC = false>
__device__ __forceinline__ void tk2h_consume(""", """template <int NC, int NSTG, int KPS, bool SA, int GS, bool PSC = false, bool SLOT = false>
__device__ __forceinline__ void tk2h_consume(""")
rep("""            if (v) { x0 = __ldg(xr + kc * 8); x1 = __ldg(xr + kc * 8 + 1); }""",
"""            if (v && gated) { x0 = __ldcg(xr + kc * 8); x1 = __ldcg(xr + kc * 8 + 1); }  // x: PDL predecessor's output
            else if (v) { x0 = __ldg(xr + kc * 8); x1 = __ldg(xr + kc * 8 + 1); }""")
rep("""        tk2g_row<0, NC, NSTG, false, KPS, SA, true, true>(slut, xf, xs, Out, R - seg_start, off, n, gsc, s_red, smem, full, empty, st, ph, rb, gated);
      else
        tk2g_row<0, NC, NSTG, false, KPS, SA, false, true>(slut, xf, xs, Out, R - seg_start, off, n, gsc, s_red, smem, full, empty, st, ph, rb, gated);""",
"""        tk2g_row<0, NC, NSTG, false, KPS, SA, true, true, SLOT>(slut, xf, xs, Out, R - seg_start, off, n, gsc, s_red, smem, full, empty, st, ph, rb, gated, s_srt);
      else
        tk2g_row<0, NC, NSTG, false, KPS, SA, false, true, SLOT>(slut, xf, xs, Out, R - seg_start, off, n, gsc, s_red, smem, full, empty, st, ph, rb, gated, s_srt);""")
rep("""    tk2h_consume<NC, NSTG, KPS, SA, GS, MRG == 2>(""",
"""    asm volatile("griddepcontrol.wait;" ::: "memory");  // moe n6: as layer TK2FX (no-op without the PDL attribute)
    tk2h_consume<NC, NSTG, KPS, SA, GS, MRG == 2, (PF & 4) != 0>(""")
rep("""#define TK2_EXP 0 """, """#define TK2H_P4 1        // moe n6: build the SA tk2h PF4 kernels (cfg 1x3x004 GS8 / 1x4w004 GS16) without TK2_EXP
#define TK2_EXP 0 """)
rep("""#if TK2_EXP
  if (cfg >= 30000) {""", """#if TK2H_P4 && !TK2_EXP
  if (sa && cfg >= 30000 && cfg < 50000 && cfg % 1000 == 4) {  // moe n6: tk2h PF4 only (SA): GS 8 / GS 16 with weight wt
    if (cfg < 40000) return tk2f_go<8, 4, 2, false, false, true, 8, 1, 4>(0, TK2F_ARGS0);
    return tk2f_go<8, 4, 2, false, false, true, 16, 1, 4>(8 + (int)(cfg / 1000) % 10, TK2F_ARGS0);
  }
#endif
#if !TK2_EXP
  TORCH_CHECK(cfg < 30000 || cfg >= 80000, "moe_tk2f: cfg ", cfg, " needs TK2_EXP (or TK2H_P4 + SA)");
#else
  if (cfg >= 30000) {""")
rep("""TK2H_W = 0  # GS16""", """TK2H_MIN = 1  # moe n6/n4s5: TK2H applies from this M (below: the default tk2f cfg)
TK2H_M = {}  # moe n6: per-M GS16 tk2h PF4 weight W; layer TUNE option "tk2h" = W + 1
TK2H_PF4 = 0  # moe n6: 1 = tk2h writes y32 like tk2f PF 4 (slot order, pre-scaled) -> moe_norm3 (cfg + 4)
TK2H_W = 0  # GS16""")
rep("""def _tk2f_cfg(M, st):
    sa = 100000 if TK2SA and st["sa"] else 0
""", """def _tk2f_cfg(M, st):
    sa = 100000 if TK2SA and st["sa"] else 0
    if M in TK2H_M:  # moe n6: {M: W}: GS16 tk2h + PF4 (moe_norm3), 2-group row weight (8 + W) / 8 (SA only: TK2H_P4)
        return sa + 40000 + TK2H_M[M] * 1000 + 4
""")
rep("""    if TK2H:
        return sa + (40000 + TK2H_W * 1000 if TK2H == 16 and M >= TK2H_MIN16 else 30000) + 20000 * TK2H_PSC""",
"""    if TK2H and M >= TK2H_MIN:
        return sa + (40000 + TK2H_W * 1000 if TK2H == 16 and M >= TK2H_MIN16 else 30000) + 20000 * TK2H_PSC + 4 * (TK2H_PF4 and not TK2H_PSC)""")
rep("""            if 80000 < c % 100000 < 100000 and (c % 100000 - 80000) // 1000 & 4:  # s13 slot-ordered y32""",
"""            if (80000 < c % 100000 < 100000 and (c % 100000 - 80000) // 1000 & 4) or (30000 <= c % 100000 < 50000 and c % 1000 == 4):  # s13 slot-ordered y32 (moe n6: tk2h + 4)""")
open(p, 'w').write(s)
print("patched", p)
