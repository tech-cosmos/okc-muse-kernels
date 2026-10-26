"""moe n8: port moe n7 BAL into a layer muse_decode_layer dir (moe_src.py + layer.py), in place.

BAL (tk2f PF & 32): per-segment gemm13 -> gemm2 counters cnt[64 + j] (gemm2 segment j waits only for gemm13 segment j)
and the rounding rows of gemm13 (n13 % G) go to the LAST CTAs, gemm2's (n2 % G) to the FIRST: no CTA has an extra
row in both phases. Bitwise = the non-BAL kernel. cnt[64..127] is re-zeroed by every tk2f follower (moe_norm2/3,
n2u, nu8 all zero cnt[0..127]).
Layer side: TUNE option "bal" (bool) at T = 8/16/32 -> moe_src TK2F_BAL_M (set of M) -> cfg + 1000000 in _tk2f_cfg
(launcher: SA builds only; pf 4 -> PF 36, pf 5 (NU8, T = 8) -> PF 52; anything else ignores the BAL bit).
usage: python patch_layer_bal.py <dir containing moe_src.py and layer.py>
"""
import os
import sys

d = sys.argv[1]
p = os.path.join(d, "moe_src.py")
s = open(p).read()


def rep(s, a, b, n=1):
    assert s.count(a) == n, (a[:80], s.count(a))
    return s.replace(a, b)


# 1. tk2f_consume: per-segment counters under BAL
a = "  constexpr int RPE = NRT / 2, NQ = 64 / NC, NS = NQ / KPS, SW = NC * 32, SB = NC * KPS * 1152;\n  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;\n  uint4 xf[NQ][2];\n  int a = 0, seg_end = -1, seg_start = 0, off = 0, n = 0, ci = 0;"
s = rep(s, a, a.replace("  const int tid", "  constexpr bool BAL = (PF & 32) != 0;  // moe n8 BAL: per-segment counters cnt[64 + j]\n  const int tid"))
a = "      if (!GATED && !((PF & 2) && R != r0)) {  // all gemm13 rows written"
s = rep(s, a, """      if (BAL && !GATED) {
        const int* cp = cnt + 64 + seg_start / RPE;
        int c;
        do {
          asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(c) : "l"(cp) : "memory");
        } while (c < 2 * RPE);
      } else""" + a[5:])
a = "          atomicAdd(cnt, R + 1 - max(seg_start, r0));"
s = rep(s, a, "          atomicAdd(BAL ? cnt + 64 + seg_start / RPE : cnt, R + 1 - max(seg_start, r0));")
# 2. moe_tk2f: balanced split
a = "  int st = 0, ph = 0;\n  if (warp == NC) {\n    if (lane == 0) {\n      int i = 0;\n      tk2f_produce<512, NC, NSTG, KPS>(Wq13"
s = rep(s, a, """  if (PF & 32) {  // moe n8 BAL: gemm13's extra rows (n13 % G) on the LAST CTAs, gemm2's (n2 % G) on the FIRST ones
    const int G = gridDim.x, b = blockIdx.x, q1 = n13 / G, e1 = G - n13 % G, q2 = n2 / G, e2 = n2 % G;
    a0 = b * q1 + max(0, b - e1); a1 = a0 + q1 + (b >= e1);
    b0 = b * q2 + min(b, e2); b1 = b0 + q2 + (b < e2);
  }
""" + a)
# 3. launcher
a = "  const bool sa = cfg >= 100000;  // + 100000: SA"
s = rep(s, a, "  const bool bal = cfg >= 1000000;  // moe n8: + 1000000 = BAL (PF | 32; SA pf 4 / 5 only)\n  cfg %= 1000000;\n" + a)
a = "    if (pf == 12) return sa ?"
s = rep(s, a, """    if (bal && sa && pf == 4) return tk2f_go<8, 4, 2, false, false, true, 8, 0, 36>(0, TK2F_ARGS0);
#ifdef MOE_BAL_NU8
    if (bal && sa && pf == 5) {
      TORCH_CHECK(nr == 64 && y_.size(0) >= 65 && g_ssw1 != nullptr, "tk2f PF 52: shape");
      return tk2f_go<8, 4, 2, false, false, true, 8, 0, 52>(0, TK2F_ARGS0, nullptr, g_ssw1);
    }
#endif
""" + a)
# 4. python: TK2F_BAL_M
a = "def _tk2f_cfg(M, st):\n    sa = 100000 if TK2SA and st[\"sa\"] else 0\n"
s = rep(s, a, "TK2F_BAL_M = set()  # moe n8 BAL: M values whose tk2f (84204 / NU8 85204, SA) runs balanced (cfg + 1000000)\n\n\n"
        + a.replace("def _tk2f_cfg(M, st):", "def _tk2f_cfg(M, st):\n    return _tk2f_cfg0(M, st) + (1000000 if M in TK2F_BAL_M and TK2SA and st[\"sa\"] else 0)\n\n\ndef _tk2f_cfg0(M, st):"))
a = "            c5 = c % 100000\n"
s = rep(s, a, "            c5 = c % 1000000 % 100000\n")
open(p, "w").write(s)

p = os.path.join(d, "layer.py")
s = open(p).read()
a = "        _moe_TK2H_M.clear()\n"
s = rep(s, a, "        _moe_TK2F_BAL_M.clear()\n        if _opt(state, m, \"bal\", TK2F_BAL_DEFAULT):  # moe n8 BAL (balanced tk2f phases, per-segment counters) at T=8/16/32\n            _moe_TK2F_BAL_M.add(m)\n" + a)
a = "    row 64 (the slots' sums of squares) was zeroed by the pre-expert _rmsnorm_gain.\"\"\"\n    st = state[\"moe\"]\n"
s = rep(s, a, a + "    _moe_TK2F_BAL_M.discard(8)\n    if _opt(state, 8, \"bal\", TK2F_BAL_DEFAULT):  # moe n8 BAL\n        _moe_TK2F_BAL_M.add(8)\n")
a = "def _nu8_d(state, tokens):\n"
s = rep(s, a, "TK2F_BAL_DEFAULT = False  # moe n8 BAL at T = 8/16/32 (TUNE option \"bal\")\n\n\n" + a)
open(p, "w").write(s)
print("patched", d)
