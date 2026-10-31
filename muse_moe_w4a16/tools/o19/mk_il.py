"""proj n22 HOT_IL: moe_hot claims the slices of one >128-token expert rb-major / slice-minor (item it of a slice group
-> rb = loc / ns, slice = loc % ns), so the ns items sharing a W row block run back to back (2nd W read = L2 hit) instead of
nrb items apart under evict_first. Same items, same math: bitwise. python mk_il.py in.py out.py"""
import sys
s = open(sys.argv[1]).read()
i0 = s.index("__global__ void __launch_bounds__(384, 1) moe_hot(")
i1 = s.index("template <bool SA, int NH>", i0)
k = s[i0:i1]
old = "const int a = it / nrb, rb = it - a * nrb, e = s_e[a]"
import os
if os.environ.get("IL", "1") == "2":  # gated: s_g0[a] = first slice of a's expert (smem, no divides); ns 1 = old decode
    new = ("const int a0 = it / nrb, ns0 = s_ns[a0], g0 = s_g0[a0], loc = it - g0 * nrb, rb = ns0 > 1 ? loc / ns0 : it - a0 * nrb, "
           "a = ns0 > 1 ? g0 + loc - rb * ns0 : a0, e = s_e[a]")
    for o, n in (("s_sz[64], s_off[33]", "s_sz[64], s_g0[64], s_off[33]"),
                 ("if (h) { s_e[p] = e; s_so[p] = 0; s_ns[p] = 1; s_sz[p] = NH; }", "if (h) { s_e[p] = e; s_so[p] = 0; s_ns[p] = 1; s_sz[p] = NH; s_g0[p] = p; }"),
                 ("s_sz[pos - ns + q] = sz; }", "s_sz[pos - ns + q] = sz; s_g0[pos - ns + q] = pos - ns; }")):
        assert k.count(o) == 1, o
        k = k.replace(o, n)
else:
    new = ("const int a0 = it / nrb, ns0 = s_ns[a0], g0 = a0 - s_so[a0] / s_sz[a0], loc = it - g0 * nrb, rb = loc / ns0, "
           "a = g0 + loc - rb * ns0, e = s_e[a]")  # proj n22 HOT_IL
assert k.count(old) == 2, k.count(old)
k = k.replace(old, new)
open(sys.argv[2], "w").write(s[:i0] + k + s[i1:])
