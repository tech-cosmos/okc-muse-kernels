# s35: source transform DEC4 -> DEC3 decode (see dec3.py for the layout). Every
#   lop_aob(prmt(X, 0x9180u|0xB3A2u), 0x807f807fu, expo<E>(NW))
# becomes decw<E>(X, NW, sel4, sel3); decw decodes DEC3 (default) or the old layout with -DPJ_DEC4 (A/B builds).
# usage: python dev/s35/dec3_src.py file.cu [...]   (in place; idempotent)
import re, sys
SEL = {"0x9180u": "0x4140u", "0xB3A2u": "0x4342u"}
DECW = """template <int P>  // s35 DEC3: one prmt (exponent constant 0x38 from a register) + one lop3 per BF16 pair
__device__ __forceinline__ unsigned decw(unsigned sw, unsigned nw, unsigned sel4, unsigned sel3) {
#ifdef PJ_DEC4  // s34 layout (sign in the byte, code nibbles): for A/B builds only
    return lop_aob(prmt(sw, sel4), 0x807f807fu, expo<P>(nw));
#else
    const unsigned sh = P == 0 ? nw : P == 1 ? nw << 4 : P == 2 ? nw << 8 : __funnelshift_r(nw, nw, 4);
    const unsigned k38 = 0x38383838u >> (gridDim.y >> 16);  // opaque register (== the constant)
    unsigned p;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(p) : "r"(sw), "r"(k38), "r"(sel3));
    return lop_aob(sh, 0x87008700u, p);
#endif
}
"""
pat = re.compile(r"lop_aob\(prmt\(([^()]*(?:\([^()]*\))?[^()]*), (0x9180u|0xB3A2u)\), 0x807f807fu, expo<([^>]+)>\(([\w.\[\]]+)\)\)")
for p in sys.argv[1:]:
    s = open(p).read()
    if "s35 DEC3" in s:
        print(p, "already"); continue
    s, n = pat.subn(lambda m: "decw<%s>(%s, %s, %s, %s)" % (m.group(3), m.group(1), m.group(4), m.group(2), SEL[m.group(2)]), s)
    assert n >= 2, (p, n)
    i = s.index("template <int P>\n__device__ __forceinline__ unsigned expo")
    j = s.index("}\n", i) + 2
    s = s[:j] + DECW + s[j:]
    open(p, "w").write(s)
    print(p, "ok", n)
