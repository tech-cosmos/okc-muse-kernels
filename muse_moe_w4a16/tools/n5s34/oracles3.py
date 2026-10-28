"""moe n5s34: in-layer CONSUMER-INSTRUCTION oracles (lead 2026-10-28 08:50Z). WRONG OUTPUT, timing only; never submit.
python oracles3.py <r-layer submission.py> <outdir> -> base0.py (TUNE = {}) and, all with TUNE = {}:
  orq  = dq() free: the 4 A registers are the raw q word (no shift/mask/IMAD), the 4 scale HMUL2 kept      -> (a)
  ors  = scale multiply free: dq()'s 4 bf16x2 = the masked codes, no HMUL2 (scale decode then dead code)   -> (b)
  orqs = both (A = raw q, no scale work at all)                                                         -> (a)+(b)
  orm  = tk2f_consume/tk2g_row mma.sync replaced by an integer xor into acc (dq + scales kept)            -> (c)
dq() is patched globally (tk2f, tk2g/tk2h and twm/twf all call it): v16/v32 rows read as the twm (T>=128... T=16/32
verify) analogue. Then, from ~/r-layer:  python ~/r-moe/tools/h200.py muse_decode_layer base=<out>/base0.py cand=<out>/orqs.py
orq=<out>/orq.py ors=<out>/ors.py orm=<out>/orm.py --cmd "python bench.py bench 2>&1 | grep -E ' us |geomean|rror'"
Written against r-layer 2caa722 (asserts on the patched lines)."""
import os, sys
L0 = open(sys.argv[1]).read()
out = sys.argv[2]; os.makedirs(out, exist_ok=True)


def sub(s, a, b, n=1):
    assert s.count(a) == n, (a, s.count(a))
    return s.replace(a, b)


def tune_off(s):
    return sub(s, "\nTUNE_BASE = {", "\nTUNE = {}  # moe n5s34 oracle A/B: defaults only\nTUNE_BASE = {")


DQ = """  a[0] = bmul(o0, s0);
  a[1] = bmul(o1, s1);
  a[2] = bmul(o2, s0);
  a[3] = bmul(o3, s1);
}"""


def dq_free(s, q, sc):
    if q and sc:
        body = "  a[0] = q; a[1] = q ^ 1u; a[2] = q ^ 2u; a[3] = q ^ 3u; (void)M; (void)o0; (void)o1; (void)o2; (void)o3;\n}"
    elif q:
        body = "  a[0] = bmul(q, s0); a[1] = bmul(q, s1); a[2] = bmul(q ^ 1u, s0); a[3] = bmul(q ^ 1u, s1);\n  (void)o0; (void)o1; (void)o2; (void)o3;\n}"
    else:
        body = "  a[0] = o0; a[1] = o1; a[2] = o2; a[3] = o3; (void)s0; (void)s1;\n}"
    return sub(s, DQ, body)


MMA = "__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {"
MMAO = ("__device__ __forceinline__ void mmao(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {  // moe n5s34 orm\n"
        "  c[0] = __uint_as_float(__float_as_uint(c[0]) ^ a[0] ^ a[1] ^ a[2] ^ a[3] ^ b0 ^ b1);\n}\n")


def mma_free(s):
    s = sub(s, MMA, MMAO + MMA)
    i = s.index("void tk2f_consume(")
    j = s.index("void tk2g_consume(", i)
    mid = s[i:j]
    k = mid.index("if (PF & 256)")  # keep the probe body as is; patch the real math after it
    n = mid[k:].count("mma16816(")
    assert n == 3, n  # tk2f_consume 1 + tk2g_row 2
    return s[:i] + mid[:k] + mid[k:].replace("mma16816(", "mmao(") + s[j:]


for name in ("base0", "orq", "ors", "orqs", "orm"):
    s = tune_off(L0)
    if name in ("orq", "ors", "orqs"):
        s = dq_free(s, "q" in name[2:], "s" in name[2:])
    if name == "orm":
        s = mma_free(s)
    open(f"{out}/{name}.py", "w").write(s)
    print(name, "ok")
