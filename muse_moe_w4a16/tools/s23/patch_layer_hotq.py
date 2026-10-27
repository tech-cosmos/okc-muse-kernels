# s23 layer port: moe_hot<NH 64> beside moe_twq (33..64-token experts in one N 64 pass) into the layer's moe_src.py.
# python tools/s23/patch_layer_hotq.py <path/to/moe_src.py>   then TUNE option at T=128 e.g. {"twq": 73988 | 32 << 24}
# (= 536944644: twq + n21 33-item half split + moe_hot<64> on 32 CTAs). Needs n21's twq (K split) already in moe_src.py.
import sys
p = sys.argv[1]; s = open(p).read()
def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (s.count(a), a[:100]); s = s.replace(a, b)
rep("""template <int NSTG, bool SA>
__global__ void __launch_bounds__(384, 1) moe_hot(""", """// s23 NH 64: the experts with 33..64 tokens in one N 64 pass (layer T = 128), with moe_twq (flags bit 20) skipping them.
template <int NSTG, bool SA, int NH = 128>
__global__ void __launch_bounds__(384, 1) moe_hot(""")
rep("  constexpr int XB = 128 * 128, STRIDE = XB + HG::WB + HG::SB;", "  constexpr int XB = NH * 128, STRIDE = XB + HG::WB + HG::SB;")
rep("    const bool h = nt > 64 && nt <= 128;\n    const unsigned b = __ballot_sync(", "    const bool h = nt > NH / 2 && nt <= NH;\n    const unsigned b = __ballot_sync(")
rep("      const CUtensorMap* map = g2 ? &maps2.m[5] : &maps13.m[5];",
    "      const CUtensorMap* map = g2 ? &maps2.m[NH == 128 ? 5 : 3] : &maps13.m[NH == 128 ? 5 : 3];")
rep("    tw_item<2, 1, 128, NSTG, true, 8, true, SA, STRIDE, XB>(", "    tw_item<2, 1, NH, NSTG, true, 8, true, SA, STRIDE, XB>(")
rep("template <bool SA>\nstatic int64_t hot_go(", "template <bool SA, int NH>\nstatic int64_t hot_go(")
rep("  constexpr int SMEM = NSTG * (128 * 128 + TWC<2, true, 8>::WB + TWC<2, true, 8>::SB) + 1024;",
    "  constexpr int SMEM = NSTG * (NH * 128 + TWC<2, true, 8>::WB + TWC<2, true, 8>::SB) + 1024;")
rep("    cudaFuncSetAttribute(moe_hot<NSTG, SA>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);",
    "    cudaFuncSetAttribute(moe_hot<NSTG, SA, NH>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);")
rep("  return launch_k(moe_hot<NSTG, SA>, grid, 384,", "  return launch_k(moe_hot<NSTG, SA, NH>, grid, 384,")
rep("  return (g_sa & 1) ? hot_go<true>(HOTA) : hot_go<false>(HOTA);",
    "  if (g_hot & 4) return (g_sa & 1) ? hot_go<true, 64>(HOTA) : hot_go<false, 64>(HOTA);  // s23: 33..64-token experts\n"
    "  return (g_sa & 1) ? hot_go<true, 128>(HOTA) : hot_go<false, 128>(HOTA);")
# moe_twq: hx skip (with or without n19 WBW)
k0 = s.index("__global__ void __launch_bounds__(640, 1) moe_twq("); k1 = s.index("static int64_t twq_go(", k0)
K = s[k0:k1]
HX = ("  // s23 hx (flags bit 20): moe_hot<NH 64> (the PDL primary) runs the experts with 33..64 tokens: skip them, no wait at\n"
      "  // entry (moe_hot triggers after its own wait: the gather is complete), wait at the end (moe_norm sees moe_hot's Y)\n"
      "  const int hx = (flags >> 20) & 1;\n")
if "  const int wbw = flags & 16;\n  if (!wbw) asm volatile(\"griddepcontrol.wait;\" ::: \"memory\");" in K:
    K = K.replace("  const int wbw = flags & 16;\n  if (!wbw) asm volatile(\"griddepcontrol.wait;\" ::: \"memory\");",
                  HX + "  const int wbw = hx ? 0 : flags & 16;\n  if (!wbw && !hx) asm volatile(\"griddepcontrol.wait;\" ::: \"memory\");")
else:
    assert K.count("  asm volatile(\"griddepcontrol.wait;\" ::: \"memory\");\n  asm volatile(\"griddepcontrol.launch_dependents;\"") == 1
    K = K.replace("  asm volatile(\"griddepcontrol.wait;\" ::: \"memory\");\n  asm volatile(\"griddepcontrol.launch_dependents;\"",
                  HX + "  if (!hx) asm volatile(\"griddepcontrol.wait;\" ::: \"memory\");\n  asm volatile(\"griddepcontrol.launch_dependents;\"")
old = "    const int nt = tid < nact ? s_off[e + 1] - s_off[e] : 0, big = 0;  // s17: 64-token passes do not fit (registers)"
assert K.count(old) == 1
K = K.replace(old, "    int nt = tid < nact ? s_off[e + 1] - s_off[e] : 0;\n    if (hx && nt > 32 && nt <= 64) nt = 0;\n"
              "    const int big = 0;  // s17: 64-token passes do not fit (registers)")
end = "    ++ntl;\n#endif\n  }\n}\n"
assert K.count(end) == 1
K = K.replace(end, "    ++ntl;\n#endif\n  }\n  if (hx && tid == 0) asm volatile(\"griddepcontrol.wait;\" ::: \"memory\");\n}\n")
s = s[:k0] + K + s[k1:]
# python: hot grid from the twq flags (bits 24..30)
rep("""                if DIAG_TL and TWM_ORDER_M.get(M, TWM_ORDER) & 2: st["_tl"] = (part, meta); q |= 2
                lib.moe_twq_launch(""", """                if DIAG_TL and TWM_ORDER_M.get(M, TWM_ORDER) & 2: st["_tl"] = (part, meta); q |= 2
                hq = (q >> 24) & 127  # moe s23: q | grid << 24 = moe_hot<NH 64> on grid CTAs first; moe_twq skips its experts
                q &= (1 << 24) - 1
                if hq:
                    lib.moe_set_hot(4)
                    lib.moe_hot_launch(st["w13b2"], st["w13sb2"], st["gs13"], st["w2b2"], st["w2sb2"], st["gs2"], st["lut"],
                                       xg, meta, sroutes, inter, y, hq)
                    q |= 1 << 20
                lib.moe_twq_launch(""")
open(p, "w").write(s)
print("patched", p)
