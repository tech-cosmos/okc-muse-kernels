"""n10s1 oracle variants of a layer submission.py: python mkvar.py <submission.py> <outdir>. Writes (all TUNE = {}):
nt_base, nt_nobnd (second boundary skipped), nt_hop (second boundary returns right after its wait), nt_nfree (moe_norm2
returns after its wait + n2u reads no y32), nt_q4 (moe_norm2 loads only a quarter of y32: same CTAs). Wrong outputs: timing only."""
import sys, os
src, out = sys.argv[1], sys.argv[2]
s = open(src).read().replace("\nTUNE_REPS = 40\n", "\nTUNE = {}\nTUNE_REPS = 40\n", 1)
def put(name, t):
    open(os.path.join(out, name + ".py"), "w").write(t)
put("nt_base", s)
a = "    _boundary[(tokens * spl,)](f, stream1 if nos"
assert s.count(a) == 1
skip = "class _BSK:\n    def __getitem__(self, g):\n        return lambda *a, **k: None\n_BNDSKIP = _BSK()\n"
put("nt_nobnd", s.replace(a, a.replace("_boundary", "_BNDSKIP")).replace("\ndef kernel_fn(data, state):", "\n" + skip + "\ndef kernel_fn(data, state):", 1))
h = s.replace("NOS: tl.constexpr = 0, rst_ptr=None, RST: tl.constexpr = False):", "NOS: tl.constexpr = 0, rst_ptr=None, RST: tl.constexpr = False, HOP: tl.constexpr = False):")
for w in ("        if PDL:\n            gdc_wait()\n        if RST:", "    if PDL:\n        gdc_wait()\n    if RST:\n"):
    assert h.count(w) == 1
    ind = w[:w.index("if")]
    h = h.replace(w, w.replace("gdc_wait()\n", "gdc_wait()\n" + ind + "if HOP:\n" + ind + "    return\n", 1))
c = "**({} if rst is None else dict(rst_ptr=rst, RST=True)))\n    return hidden, stream2"
assert h.count(c) == 1
put("nt_hop", h.replace(c, c.replace(")))\n", ")), HOP=True)\n")))
b = "  if (tok == 0 && tid < 128) zc[tid] = 0;  // moe_tk2f's segment counters: zero again for the next call\n"
c = "            const float4 c = p[0], d = p[1];\n"
assert s.count(b) == 1 and s.count(c) == 1
put("nt_nfree", s.replace(b, b + "  return;  // ORACLE norm2 free\n").replace(c, "            const float4 c = make_float4(wv[0], wv[1], wv[2], wv[3]), d = c;  // ORACLE y free\n"))
q = "    loc[s] = e >= 0 && e < count;\n    ss[s] = 0.f;\n    if (loc[s]) {\n      const float4* p = reinterpret_cast<const float4*>(Y + (size_t)(SLOT ? tok * 8 + s"
assert s.count(q) == 1
put("nt_q4", s.replace(q, q.replace("e < count;", "e < count && tid < 128;  // ORACLE quarter loads")))
