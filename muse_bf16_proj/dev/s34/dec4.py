# s34: 12-bit decode with 4 instead of 5 instructions per BF16 pair. The exponent word (s & 0x07800780) | 0x38003800
# needs two immediates, so ptxas emits 2 LOP3s; with both constants in registers it is ONE 3-register LOP3 (and the
# final (prmt & 0x807f807f) | e one more). The constants are made opaque to ptxas by >> (gridDim.y >> 16) (always
# 0: gridDim.y <= 65535), i.e. bit-identical for every launch shape.
# usage: python dev/s34/dec4.py file.cu [...]   (in place; idempotent; fails if the pattern is missing)
import sys
OLD_E = "    return (s & 0x07800780u) | 0x38003800u;\n}"
NEW_E = ("    const unsigned z = gridDim.y >> 16, m2 = 0x07800780u >> z, cc = 0x38003800u >> z;  // s34 DEC4: opaque (== the immediates)\n"
         "    unsigned r;\n    asm(\"lop3.b32 %0, %1, %2, %3, 0xEA;\" : \"=r\"(r) : \"r\"(s), \"r\"(m2), \"r\"(cc));  // (s & m2) | cc\n"
         "    return r;\n}")
import re
for p in sys.argv[1:]:
    s = open(p).read()
    if "s34 DEC4" in s:
        print(p, "already"); continue
    assert s.count(OLD_E) == 1, (p, s.count(OLD_E))
    s = s.replace(OLD_E, NEW_E)
    # (prmt(a, sel) & 0x807f807fu) | expo<i>(nw)  ->  one lop3 (prmt & 0x807f807f) | expo
    pat = re.compile(r"\(prmt\(([^()]*(?:\([^()]*\))?[^()]*), (0x[0-9A-Fa-f]+u)\) & 0x807f807fu\) \| (expo<[\w +]+>\(\w+(?:\.\w+)?\))")
    s, n = pat.subn(r"lop_aob(prmt(\1, \2), 0x807f807fu, \3)", s)
    assert n >= 2, (p, n)
    LOP = ("__device__ __forceinline__ unsigned lop_aob(unsigned a, unsigned b, unsigned c) {  // s34: (a & b) | c\n"
           "    unsigned r;\n    asm(\"lop3.b32 %0, %1, %2, %3, 0xEA;\" : \"=r\"(r) : \"r\"(a), \"r\"(b), \"r\"(c));\n    return r;\n}\n")
    i = s.index("template <int P>\n__device__ __forceinline__ unsigned expo")
    s = s[:i] + LOP + s[i:]
    open(p, "w").write(s)
    print(p, "ok", n, "words")
