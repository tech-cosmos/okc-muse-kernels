"""fresh x35: csl (d1 sliding) as a PDL secondary of the qkv gemv13 with a LATE trigger in g13 (prototype on submission.py).
s62 tried CSL_PW at d1 with g13 triggering at START (csl CTAs resident during the whole W stream: +1.3..1.7 us). Here g13
(qkv launch only: pdl bit 1) triggers per CTA when it issues its LAST W chunk (mode 'issue') or after computing its last
chunk (mode 'last'); csl's page table / cos-sin / KV gathers then go out during the qkv drain, the qkv rows after the wait.
python tools/fresh/x35/late.py <submission.py> <out.py> issue|last|start"""
import sys
s = open(sys.argv[1]).read()
mode = sys.argv[3]


def rep(a, b, cnt=1):
    global s
    assert s.count(a) == cnt, (s.count(a), a[:90])
    s = s.replace(a, b)


# runtime src13 edits: start trigger only without bit 1; secondary flags use bit 0
rep("""                    src13 = src13.replace(anchor, '    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\\n' + anchor)""",
    """                    src13 = src13.replace(anchor, ('    if (!(pdl & 2)) ' if 'vg' in anchor else '    ') + 'asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\\n' + anchor)""")
rep('"        if (i < KK / 8 && !pdl) xv[q]', '"        if (i < KK / 8 && !(pdl & 1)) xv[q]')
rep('"    if (pdl) {  // s33', '"    if (pdl & 1) {  // s33')
i = s.find("        g13::gemv<U_, V_, B_, K_, P_><<<blocks, 256 * V_, 0, st>>>(")
j = s.find("lc.numAttrs = pdl ? 1 : 0;", i)
s = s[:j] + "lc.numAttrs = (pdl & 1) ? 1 : 0;" + s[j + len("lc.numAttrs = pdl ? 1 : 0;"):]
# late trigger inside the g13 kernel (_G13_SRC only)
a = s.find("_G13_SRC = r'''")
e = s.find("'''", a + 20)
seg = s[a:e]
TR = 'asm volatile("griddepcontrol.launch_dependents;" ::: "memory");'
if mode == "issue":
    old = "                if (cn < nch) load(p, cn);\n"
    assert seg.count(old) == 1
    seg = seg.replace(old, "                if (cn < nch) { load(p, cn); if ((pdl & 2) && cn + PF * G >= nch) " + TR + " }\n")
    old = "    __syncthreads();\n    int par = 0, j = 0;\n"
    assert seg.count(old) == 1
    seg = seg.replace(old, "    if ((pdl & 2) && c + PF * G >= nch) " + TR + "\n" + old)
elif mode == "last":
    old = "                if (cn < nch) load(p, cn);\n"
    assert seg.count(old) == 1
    seg = seg.replace(old, "                if (cn < nch) load(p, cn); else if (pdl & 2) " + TR + "\n")
s = s[:a] + seg + s[e:]
# layer: qkv gemv13 at T=1 gets bit 1 when csl runs PW; csl PW at T=1
rep("                              int(o13 and UNPOOL_PDL and key in GEMV_PDL_SECONDARY.get((1, sliding), ())))",
    "                              int(o13 and UNPOOL_PDL and key in GEMV_PDL_SECONDARY.get((1, sliding), ())) + 2 * int(key == \"qkv_gate\" and sliding and X35_LATE))")
rep("CSL_PW_TOKENS = (16,)", "CSL_PW_TOKENS = (16, 1)\nX35_LATE = %s" % (mode != "start"))
open(sys.argv[2], "w").write(s)
