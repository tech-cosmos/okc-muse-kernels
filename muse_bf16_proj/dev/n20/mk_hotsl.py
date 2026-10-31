"""proj n20 HOT_SL: moe_hot (NH 128) also takes >128-token experts as 128-token slices; moe_twm skips every >64-token
expert when moe_hot is on. python mk_hotsl.py <layer submission.py> <out.py>"""
import sys
s = open(sys.argv[1]).read()
def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (a, s.count(a))
    s = s.replace(a, b)
rep("    if (hx && nt > 64 && nt <= 128) nt = 0;", "    if (hx && nt > 64) nt = 0;  // proj n20 HOT_SL: moe_hot takes every >64-token expert (128-token slices)")
rep("  __shared__ int s_e[32], s_off[33], s_item[NSTG], s_nh;",
    "  __shared__ int s_e[64], s_so[64], s_ns[64], s_off[33], s_item[NSTG], s_nh;  // proj n20: slice table (expert, token offset, slices)")
rep("""    const bool h = nt > NH / 2 && nt <= NH;
    const unsigned b = __ballot_sync(0xffffffffu, h);
    if (h) s_e[__popc(b & ((1u << tid) - 1))] = e;
    if (tid == 0) s_nh = __popc(b);""",
"""    const bool h = nt > NH / 2 && (NH == 128 || nt <= NH);
    const int ns = h ? (nt + NH - 1) / NH : 0;  // proj n20 HOT_SL: >128-token experts as ceil(nt / 128) slices (NH 128 only)
    int pos = ns;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) { const int y = __shfl_up_sync(0xffffffffu, pos, o); if (tid >= o) pos += y; }
    for (int q = 0; q < ns; ++q) { s_e[pos - ns + q] = e; s_so[pos - ns + q] = q * NH; s_ns[pos - ns + q] = ns; }
    if (tid == 31) s_nh = pos;""")
rep("      const int a = it / nrb, rb = it - a * nrb, e = s_e[a], off = s_off[e];",
    "      const int a = it / nrb, rb = it - a * nrb, e = s_e[a], off = s_off[e] + s_so[a];")
rep("          if (v >= HG::NRB * NEEDK) break;", "          if (v >= HG::NRB * NEEDK * s_ns[a]) break;")
rep("    const int a = it / nrb, rb = it - a * nrb, e = s_e[a], off = s_off[e], nv = s_off[e + 1] - off;",
    "    const int a = it / nrb, rb = it - a * nrb, e = s_e[a], off = s_off[e] + s_so[a], nv = min(NH, s_off[e + 1] - off);")
rep("        if (atomicAdd(done + 32 + e, 1) == HN::NRB - 1) atomicAdd(done + 32 + e, 1 << 20);",
    "        if (atomicAdd(done + 32 + e, 1) == HN::NRB * s_ns[a] - 1) atomicAdd(done + 32 + e, 1 << 20);")
open(sys.argv[2], "w").write(s)
print("ok")
