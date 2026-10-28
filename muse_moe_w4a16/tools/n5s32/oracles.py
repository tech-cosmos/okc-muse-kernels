"""moe n5s32: in-layer H200 oracles for the layer's tk2f experts (T <= 32: d1/v1/d16/d32). WRONG OUTPUT, timing only; never submit.
python oracles.py <r-layer submission.py> <outdir> -> base0.py (TUNE = {}), orc.py (tk2f/tk2g consumers: no dequant/mma, the
ring + x loads + barriers kept), orw.py (tk2f weights re-read from L2: each producer streams its first 2 rows over and over),
both.py (orw + orc: ring at L2 rate, no math), orz.py (tk2f with NO rows: routing + launch + norm2/n2u only),\norn.py (producer issues no copies: zero-byte zero-latency ring, math kept), ornc.py (orn + orc: rows' structure only). Then:
python ~/r-moe/tools/h200.py muse_decode_layer base=<outdir>/base0.py cand=<outdir>/both.py ...  (run from ~/r-layer)
Pattern = proj n5s14 (dev/n5s14/oracles.py). Written against r-layer da2bcd9 (asserts on the patched lines)."""
import os, sys
L0 = open(sys.argv[1]).read()
out = sys.argv[2]; os.makedirs(out, exist_ok=True)


def sub(s, a, b, n=1):
    assert s.count(a) == n, (a, s.count(a))
    return s.replace(a, b)


def tune_off(s):
    return sub(s, "\nTUNE_BASE = {", "\nTUNE = {}  # moe n5s32 oracle A/B: defaults only\nTUNE_BASE = {")


def flags(s, orc, orw, orz=0, orn=0):
    return sub(s, "\n#define TK2F_DIAG 0 ", f"\n#define TK2F_ORC {orc}\n#define TK2F_ORW {orw}\n#define TK2F_ORZ {orz}\n#define TK2F_ORN {orn}\n#define TK2F_DIAG 0 ")


def no_copy(s):  # tk2f_produce: no bulk copies; a plain arrive completes each stage (zero-latency, zero-byte stream)
    i = s.index("void tk2f_produce(")
    a = "      mbar_expect_tx(&full[st], SB);\n"
    j = s.index(a, i); assert j - i < 3000, j - i
    b = "      if (TK2F_ORN) { mbar_arrive(&full[st]); if (++st == NSTG) { st = 0; ph ^= 1; } continue; }\n" + a
    return s[:j] + b + s[j + len(a):]


def no_rows(s):  # moe_tk2f: every CTA gets empty gemm13/gemm2 row ranges (routing, launch, barriers' setup, norm2 kept)
    a = "  int st = 0, ph = 0;\n  if (warp == NC) {\n    if (lane == 0) {\n      int i = 0;\n      constexpr bool SQ = (PF & 512) != 0;  // layer s68"
    return sub(s, a, "  if (TK2F_ORZ) { a1 = a0; b1 = b0; }\n" + a)


def no_math(s):  # tk2f_consume: reuse layer s81's PF 256 probe body; tk2g_row (GS16 tk2h, T=32): same skip
    s = sub(s, "if (PF & 256) {  // layer s81", "if ((PF & 256) || TK2F_ORC) {  // layer s81")
    i = s.index("void tk2g_row(")
    a = "      const int qq = q * KPS + h;\n#pragma unroll\n      for (int ii = 0; ii < 2; ++ii) {"
    j = s.index(a, i); assert j - i < 6000, j - i
    b = ("      const int qq = q * KPS + h;\n      if (TK2F_ORC) { acc[0][0][0] += __uint_as_float(sc[0] ^ wv0[h].x ^ wv1[h].y ^ xf[qq][0].x); continue; }\n"
         "#pragma unroll\n      for (int ii = 0; ii < 2; ++ii) {")
    return s[:j] + b + s[j + len(a):]


def w_l2(s):  # tk2f_produce only: rows R -> r0 + (R - r0) % 2 (same expert lookup path; ~144 KB per CTA, L2-resident)
    i = s.index("void tk2f_produce(")
    a = "  for (int R = r0; R < r1; ++R) {\n    while (s_rs[a + 1] <= R) ++a;"
    j = s.index(a, i); assert j - i < 3000, j - i
    b = ("  for (int R0 = r0; R0 < r1; ++R0) {\n    const int R = TK2F_ORW ? r0 + ((R0 - r0) & 1) : R0;\n"
         "    while (s_rs[a + 1] <= R) ++a;")
    return s[:j] + b + s[j + len(a):]


for name, orc, orw, orz, orn in (("base0", 0, 0, 0, 0), ("orc", 1, 0, 0, 0), ("orw", 0, 1, 0, 0), ("both", 1, 1, 0, 0),
                                 ("orz", 0, 0, 1, 0), ("orn", 0, 0, 0, 1), ("ornc", 1, 0, 0, 1)):
    s = no_copy(no_rows(w_l2(no_math(flags(tune_off(L0), orc, orw, orz, orn)))))
    open(f"{out}/{name}.py", "w").write(s)
    print(name, "ok")
