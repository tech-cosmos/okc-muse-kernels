"""proj n9s1: in-layer H200 oracles for the layer g30 qkv + o GEMMs (adapted from n5s14) (WRONG OUTPUT, timing only; never submit).
python oracles.py <r-layer submission.py> <outdir>  -> base0.py (TUNE = {}), noesc.py, orw.py (pooler+unpool W from L2),
orc.py (pooler+unpool no decode/wgmma), both.py (orw + orc). Then:
python ~/r-proj/tools/h200.py muse_decode_layer base=<outdir>/base0.py cand=<outdir>/both.py --cmd "python bench.py bench 2>&1 | grep -i -E 'fail|error| us |geomean'"
Written against r-layer 671cbdd (asserts on the patched lines; adjust if the layer's g30 source moves)."""
import os, sys
L0 = open(sys.argv[1]).read().split("\n")
out = sys.argv[2]; os.makedirs(out, exist_ok=True)
SEL = "N == 2304 || (N == 8192 && KS == 16)"  # qkv (N 2304) or o (K 1024) in g30_kernel; g30k untouched


def tune_off(L):
    i = next(i for i, l in enumerate(L) if l.startswith("TUNE_BASE = {"))
    L.insert(i, "TUNE = {}  # proj n5s14 oracle A/B: defaults only")


def checks_off(L):  # the _g30*_prepare checks vs F.linear: accept garbage
    cur = ""
    for i, l in enumerate(L):
        if l.startswith("def "):
            cur = l
        if cur.startswith("def _g30") and "_prepare(" in cur and "< 1e-2" in l and "ref.norm()" in l:
            L[i] = l.replace("< 1e-2", "< 1e30")


def w_l2(L):
    idx = [i for i, l in enumerate(L) if "bulk1d(ws + st * G30_WST, src + (size_t)i * G30_WST" in l]
    assert len(idx) == 2, idx
    L[idx[0]] = L[idx[0]].replace("(size_t)i *", f"(size_t)(({SEL}) ? (i & 7) : i) *")


def no_math(L):
    for mac, cond in (("#define G30_STEP(a, pa)", SEL),):
        a = next(i for i, l in enumerate(L) if l.startswith(mac)) + 6
        assert "const uint4 s01" in L[a], L[a]
        L[a] = L[a].replace("const uint4 s01", f"if (!({cond})) {{ const uint4 s01", 1)
        c = next(i for i in range(a, a + 12) if "wgmma.commit_group" in L[i])
        L.insert(c + 1, "    }" + " " * 100 + "\\")


def no_esc(L):
    i = next(i for i, l in enumerate(L) if l == "#define G30_ESC 8"); L[i] = "#define G30_ESC 0"
    i = next(i for i, l in enumerate(L) if l.startswith("#define G30_GXD 0")); L[i] = L[i].replace("G30_GXD 0", "G30_GXD 3")


for name, fs in (("base0", []), ("both", [checks_off, w_l2, no_math])):
    L = list(L0); tune_off(L)
    for f in fs: f(L)
    open(f"{out}/{name}.py", "w").write("\n".join(L))
    print(name, "ok")
