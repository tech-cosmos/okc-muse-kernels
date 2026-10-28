"""moe n5s33: DECOMPOSITION of n5s32's ornc oracle (tk2f rows' structure: zero bytes, zero math). WRONG OUTPUT, timing only.
python oracles2.py <r-layer submission.py> <outdir> -> base0/orz/ornc (as n5s32) + ornc with one structure piece removed, in
tk2f_consume only (the T=1/T=8 path; d16/d32's tk2g rows keep theirs):
  ornb = no gemm13 -> gemm2 phase barrier (cnt spin), ornx = no x/inter LDGs (xf = 0), ornr = no bar.sync 1 + no row output
  stores, ornall = all three (should approach orz if the decomposition is complete)."""
import importlib.util, os, sys
here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("o1", os.path.join(here, "..", "n5s32", "oracles.py"))
src = open(spec.origin).read()
src = src[:src.index("\nfor name, orc")]  # helpers only
o1 = {}
sys_argv = sys.argv
exec(compile(src.replace("L0 = open(sys.argv[1]).read()\nout = sys.argv[2]; os.makedirs(out, exist_ok=True)\n", ""), spec.origin, "exec"), o1)
L0 = open(sys.argv[1]).read()
out = sys.argv[2]; os.makedirs(out, exist_ok=True)


def piece(s, b, x, r):
    s = o1["sub"](s, "\n#define TK2F_DIAG 0 ", f"\n#define TK2F_ORB {b}\n#define TK2F_ORX {x}\n#define TK2F_ORR {r}\n#define TK2F_DIAG 0 ")
    i = s.index("void tk2f_consume(")
    j = s.index("void tk2g_row(", i)
    body = s[i:j]
    a = "        } while (c < need);"
    assert body.count(a) == 1
    body = body.replace("        do {\n          asm volatile(\"ld.acquire.gpu.global.b32 %0, [%1];\" : \"=r\"(c) : \"l\"(cnt) : \"memory\");\n        } while (c < need);",
                        "        if (!TK2F_ORB) do {\n          asm volatile(\"ld.acquire.gpu.global.b32 %0, [%1];\" : \"=r\"(c) : \"l\"(cnt) : \"memory\");\n        } while (c < need);")
    assert "if (!TK2F_ORB) do" in body
    assert body.count("      const bool v = g < n;\n") == 1  # (the x LDG lines are patched again by the layer at import)
    body = body.replace("      const bool v = g < n;\n", "      const bool v = g < n && !TK2F_ORX;\n")
    a = '    asm volatile("bar.sync 1, %0;" ::"n"(NC * 32) : "memory");\n'
    assert body.count(a) == 1
    body = body.replace(a, '    if (!TK2F_ORR) asm volatile("bar.sync 1, %0;" ::"n"(NC * 32) : "memory");\n')
    assert body.count("        if (tok < n) {") == 1 and body.count("      if (tok < n) {") == 2, (body.count("        if (tok < n) {"), body.count("      if (tok < n) {"))
    body = body.replace("if (tok < n) {", "if (!TK2F_ORR && tok < n) {")
    return s[:i] + body + s[j:]


def gen(orc, orz, orn, b, x, r):
    s = o1["tune_off"](L0)
    s = o1["flags"](s, orc, 0, orz, orn)
    s = o1["no_copy"](o1["no_rows"](o1["w_l2"](o1["no_math"](s))))
    return piece(s, b, x, r)


for name, args in (("base0", (0, 0, 0, 0, 0, 0)), ("orz", (0, 1, 0, 0, 0, 0)), ("ornc", (1, 0, 1, 0, 0, 0)),
                   ("ornb", (1, 0, 1, 1, 0, 0)), ("ornx", (1, 0, 1, 0, 1, 0)), ("ornr", (1, 0, 1, 0, 0, 1)),
                   ("ornall", (1, 0, 1, 1, 1, 1))):
    open(f"{out}/{name}.py", "w").write(gen(*args))
    print(name, "ok")
