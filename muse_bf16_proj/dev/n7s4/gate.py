"""proj n7s4: lead 06:00Z step-2 GATE oracles (WRONG OUTPUT where noted, timing only; never submit).
Ceiling of a persistent 'HBM-never-idle' kernel for the d16/d32 piece o -> bnd1 -> pooler || router -> routing -> rmsnorm.
python gate.py <r-layer submission.py> <outdir> -> base0.py (TUNE = {}), R.py, P.py, F.py:
  R = router GEMM + routing free (layer free_oracle 'r': kept outputs, same routing) + pooler W read from L2 (g30k, d32 only)
      ~ 'all piece weights already on chip, consumers unchanged'
  P = R + pooler no decode/wgmma (g30k) = pooler and router both free (d32; d16 pooler is g24: untouched)
  F = P + o (g30_kernel N 8192 KS 16) W from L2 + no math; bnd1 stays REAL (a kept-output b1 lets the expert chain skip the
      attention wait: H100 d32 full -35 us artifact, n7s4) = the piece except bnd1/rmsnorm/launches (d32; d16 o is g27: untouched)
Run on the H200: dir with these + run.sh, `python tools/h200.py muse_decode_layer a=<dir> --cmd 'bash run.sh'`."""
import os, sys
here = os.path.dirname(os.path.abspath(__file__))
L0 = open(sys.argv[1]).read().split("\n")
out = sys.argv[2]; os.makedirs(out, exist_ok=True)
FO = open(os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])), "tools/free_oracle.py")).read()
HELPER = FO.split("HELPER = '''")[1].split("'''")[0]
SEL_W = "(N == 4096)"  # g30_kernel: none at d32 for the pooler (g30k), o added in F


def find(L, s, start=0):
    return next(i for i in range(start, len(L)) if L[i].startswith(s))


def sub(L, s, old, new):
    i = find(L, s); assert old in L[i], (s, L[i]); L[i] = L[i].replace(old, new, 1)


def helper(L):
    k = find(L, "def _kernel_fn(data, state):")
    L[k:k] = HELPER.split("\n")
    sub(L, "    _SLIDING[0] = cfg[\"window\"] > 0", "_SLIDING[0] =", "_ORC_T[0] = cfg[\"tokens\"]; _SLIDING[0] =")


def free_r(L):
    sub(L, "            topk_ids, local_ids, topk_weights = _route(state, h, h32,", "_route(", '_orc("r", _route)(')


def free_b1(L):
    sub(L, "      _boundary[(tokens * spl + npf,)](a,", "_boundary[(tokens * spl + npf,)](",
        'stream1, h, h32 = _orc("b1", _boundary[(tokens * spl + npf,)], (6, 7, 8))(')


def checks_off(L):
    cur = ""
    for i, l in enumerate(L):
        if l.startswith("def "):
            cur = l
        if cur.startswith("def _g30") and "_prepare(" in cur and "< 1e-2" in l and "ref.norm()" in l:
            L[i] = l.replace("< 1e-2", "< 1e30")


def w_l2(L, o=False):
    idx = [i for i, l in enumerate(L) if "bulk1d(ws + st * G30_WST, src + (size_t)i * G30_WST" in l]
    assert len(idx) == 2, idx
    if o:
        L[idx[0]] = L[idx[0]].replace("(size_t)i *", "(size_t)((N == 8192 && KS == 16) ? (i & 7) : i) *")
    L[idx[1]] = L[idx[1]].replace("(size_t)i *", "(size_t)(i & 7) *")  # g30k = pooler only


def no_math(L, o=False):
    mm = [("#define G30K_STEP(a, pa)", "1")] + ([("#define G30_STEP(a, pa)", "N == 8192 && KS == 16")] if o else [])
    for mac, cond in mm:
        a = find(L, mac) + 6
        assert "const uint4 s01" in L[a], L[a]
        L[a] = L[a].replace("const uint4 s01", f"if (!({cond})) {{ const uint4 s01", 1)
        c = next(i for i in range(a, a + 12) if "wgmma.commit_group" in L[i])
        L.insert(c + 1, "    }" + " " * 100 + "\\")


V = {"base0": [], "R": [helper, free_r, checks_off, w_l2], "P": [helper, free_r, checks_off, w_l2, no_math],
     "F": [helper, free_r, checks_off, lambda L: w_l2(L, True), lambda L: no_math(L, True)]}
for name, fs in V.items():
    L = list(L0)
    L.insert(find(L, "TUNE_BASE = {"), "TUNE = {}  # proj n7s4 oracle: defaults only")
    for f in fs: f(L)
    open(f"{out}/{name}.py", "w").write("\n".join(L)); print(name, "ok")
