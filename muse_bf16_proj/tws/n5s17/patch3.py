import sys, re
p = sys.argv[1]; s = open(p).read()
def rep(a, b, n=1):
    global s
    c = s.count(a); assert c == n, (c, a[:90]); s = s.replace(a, b)
rep("            if (lo < hi) kend = min(lo + SB_, hi);  // next batch [lo, kend) follows contiguously? no: lo == kend here",
    "            if (lo < hi) kend = min(lo + SB_, hi);  // next batch [lo = kend, ..): only the owner advances lo")
rep("""        kend = SB_;
        atomicExch(sw, (unsigned long long)SB_ | (64ull << 8) | (1ull << 16));""",
    """        kend = SB_;
        __threadfence();  // release: done[e] (acquired above) before a thief's TMA reads Inter
        atomicExch(sw, (unsigned long long)SB_ | (64ull << 8) | (1ull << 16));""")
rep("""            if (atomicCAS(tws + best, bw, nw) != bw) continue;""",
    """            if (atomicCAS(tws + best, bw, nw) != bw) continue;
            __threadfence();  // acquire (pairs with the owner's fence before its atomicExch)""")
# TWI -> fin
rep("#define TWI(RT, NN) tw_item<RT, 1, NN, NSTG, true, 8, true, SA, STRIDE>(smem, full, empty, st, ph, slut, warp, lane, nv, off, rb, gsc, sroutes, out, g2, \\\n                                                        h < 0 ? 64 : 32, pt, tick + j, &s_flag, h ? -16384 : 16384, \\\n                                                        dyn",
    "    bool fin = true;\n#define TWI(RT, NN) fin = tw_item<RT, 1, NN, NSTG, true, 8, true, SA, STRIDE>(smem, full, empty, st, ph, slut, warp, lane, nv, off, rb, gsc, sroutes, out, g2, \\\n                                                        h < 0 ? 64 : 32, pt, tick + j, &s_flag, h ? -16384 : 16384, \\\n                                                        dyn")
i0 = s.index("__global__ void __launch_bounds__(384, 1) moe_twm(")
i1 = s.index("// ---- s29 moe_hot", i0)
t = s[i0:i1]
a = """#if NORM_EARLY
    else {  // n4s1"""
assert t.count(a) == 1
t = t.replace(a, """#if NORM_EARLY
    else if (fin) {  // n5s17: a TWS partial-only piece does not count; n4s1""")
s = s[:i0] + t + s[i1:]
# host
rep("""                      torch::Tensor y_, torch::Tensor tk_, torch::Tensor part_, int64_t ns) {
  constexpr int NSTG = 8;
  constexpr int SMEM = NSTG * TWC<4, true, 8>::STG + 1024;""", """                      torch::Tensor y_, torch::Tensor tk_, torch::Tensor part_, int64_t ns, torch::Tensor tws_, int64_t stw) {
  constexpr int NSTG = 8;
  constexpr int SMEM = NSTG * TWC<4, true, 8>::STG + 1024;""")
rep("""      g_hot & 1);
}
int64_t moe_twm_launch(""", """      g_hot & 1, stw ? (unsigned long long*)tws_.data_ptr() : nullptr, (int)stw);
}
int64_t moe_twm_launch(""")
rep("""                       torch::Tensor inter_, torch::Tensor y_, torch::Tensor tk_, torch::Tensor part_, int64_t ns) {
#define TWMA w13_, s13_, gs13_, w2_, s2_, gs2_, w13h_, s13h_, w2h_, s2h_, lut_, xg_, meta_, sr_, inter_, y_, tk_, part_, ns""",
"""                       torch::Tensor inter_, torch::Tensor y_, torch::Tensor tk_, torch::Tensor part_, int64_t ns,
                       torch::Tensor tws_, int64_t stw) {
  if (stw) {  // n5s17 TWS: NV words + 8 NV ints; partial slots past 67 * 32768 floats: NV + 132 * 4
    const int64_t nv = stw & 1023;
    TORCH_CHECK(!(g_hot & 1) && ns == 0, "TWS: no hot / K split");
    TORCH_CHECK(tws_.numel() * tws_.element_size() >= nv * 40, "TWS words");
    TORCH_CHECK(part_.numel() >= 67 * 32768 + (nv + 132 * 4) * 16384, "TWS part");
  }
#define TWMA w13_, s13_, gs13_, w2_, s2_, gs2_, w13h_, s13h_, w2h_, s2h_, lut_, xg_, meta_, sr_, inter_, y_, tk_, part_, ns, tws_, stw""")
open(p, "w").write(s)
print("ok")
