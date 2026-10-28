"""proj n5s20: moe-board DIAG generator (print only, never submit): H200 dq-free ceiling of twm || moe_hot on the
layer's v16/v32 routings. python mk.py <r-moe submission.py> <out.py>. A second library (DQO 1 = orq: A = raw q word,
scale HMUL2 kept; DQO 2 = orqs: A = raw q, no scale work) is built only inside the DIAG; the timed path is moe's main."""
import sys
s = open(sys.argv[1]).read()


def sub(a, b, n=1):
    global s
    assert s.count(a) == n, (a, s.count(a))
    s = s.replace(a, b)


DQ = """  a[0] = bmul(o0, s0);
  a[1] = bmul(o1, s1);
  a[2] = bmul(o2, s0);
  a[3] = bmul(o3, s1);
}"""
sub(DQ, """#if DQO == 2
  a[0] = q; a[1] = q ^ 1u; a[2] = q ^ 2u; a[3] = q ^ 3u; (void)M; (void)o0; (void)o1; (void)o2; (void)o3; (void)s0; (void)s1;
#elif DQO == 1
  a[0] = bmul(q, s0); a[1] = bmul(q, s1); a[2] = bmul(q ^ 1u, s0); a[3] = bmul(q ^ 1u, s1);
  (void)o0; (void)o1; (void)o2; (void)o3;
#else
""" + DQ[:-2] + "\n#endif\n}")
sub("\n#define DQW 0 ", "\n#define DQO 0  // proj n5s20 DIAG oracle (WRONG output): 1 = dq() rotations/masks free, 2 = all of dq() free\n#define DQW 0 ")
sub("    _LIB = _LIBX.get(int(DQW))", "    _LIB = _LIBX.get((int(DQW), int(DQO)))")
sub("        _LIBX[int(DQW)] = _LIB", "        _LIBX[(int(DQW), int(DQO))] = _LIB")
sub('        src = src.replace("#define DQW 0 ", f"#define DQW {int(DQW)} ")',
    '        src = src.replace("#define DQW 0 ", f"#define DQW {int(DQW)} ").replace("#define DQO 0 ", f"#define DQO {int(DQO)} ")')
sub("        if DIAG:  # n4s5", "        if DIAG and DIAG_XB:  # n4s5")
sub("\nDIAG = 0\n", "\nDIAG = 1\nDIAG_XB = False  # proj n5s20: no DIAG-only tk2f builds\nDQO = 0\n")
sub("    for load, seed, T in _DIAG_HDR:\n", "    for load, seed, T in _DIAG_HDR:\n        if T < 128: p += T * 8; continue  # proj n5s20: big-M only\n")
i = s.index("DIAG_VARIANTS = {"); j = s.index("\n}\n", i) + 3
P = '{"HOT_M": {256: 32}, "TWF_SPLIT_M": {256: 66}'
s = s[:i] + ("DIAG_VARIANTS = {  # proj n5s20: layer pin (hot 32 + twf_split 66 at T=256) vs dq-free oracles; moe default twm control\n"
             '    "dec": [],\n'
             f'    "big": [("pin", {P}}}), ("orq", {P}, "DQO": 1}}), ("orqs", {P}, "DQO": 2}}), ("twm", {{}}),\n'
             '            ("twmorqs", {"DQO": 2}), ("pinb", ' + P + '})],\n}\n') + s[j:]
open(sys.argv[2], "w").write(s)
print("ok")
# n5s20 v2: one oracle library (DQO 2), compiled in a thread beside prepare()'s main build (judge prepare limit 300 s)
s = open(sys.argv[2]).read()
i = s.index("def _lib():"); j = s.index("# ---- prepare-time repacking")
body = s[i:j]
head = "def _lib():\n    global _LIB\n"
assert body.startswith(head)
rest = body[len(head):].replace("_LIB = _LIBX.get((int(DQW), int(DQO)))", "L = _LIBX.get((int(DQW), dqv))")
rest = rest.replace("if _LIB is None:", "if L is None:").replace("_LIB = load_inline(", "L = load_inline(")
rest = rest.replace("_LIBX[(int(DQW), int(DQO))] = _LIB", "_LIBX[(int(DQW), dqv)] = L").replace("    return _LIB\n", "    if dqo is None:\n        _LIB = L\n    return L\n")
rest = rest.replace('f"#define DQO {int(DQO)} "', 'f"#define DQO {dqv} "')
assert "_LIB " not in rest.replace("_LIB = L", ""), rest
body = "def _lib(dqo=None):  # n5s20: dqo = build that oracle library only (no global _LIB change; thread-safe)\n    global _LIB\n    dqv = int(DQO) if dqo is None else dqo\n" + rest
s = s[:i] + body + s[j:]
s = s.replace("def prepare(weights):\n    lib = _lib()",
              "def prepare(weights):\n    import threading\n    th = threading.Thread(target=_lib, args=(2,)) if DIAG else None\n"
              "    if th: th.start()\n    lib = _lib()\n    if th: th.join()", 1)
s = s.replace('("orq", {"HOT_M": {256: 32}, "TWF_SPLIT_M": {256: 66}, "DQO": 1}), ', "")
open(sys.argv[2], "w").write(s)
print("ok v2")
