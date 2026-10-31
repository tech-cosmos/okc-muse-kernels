"""proj n21 HOT_SL for the moe board (muse_moe_w4a16/submission.py): port of dev/n20/mk_hotsl.py (layer) to the moe-board
moe_hot (h condition uses NX/64 there). python mk_hotsl_moe.py <moe submission.py> <out.py> (FP=1 env: ballot fast path, n21b)"""
import sys
s = open(sys.argv[1]).read()
def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (a, s.count(a))
    s = s.replace(a, b)
rep("    if (hx && nt > 64 && nt <= 128) nt = 0;", "    if (hx && nt > 64) nt = 0;  // proj n21 HOT_SL: moe_hot takes every >64-token expert (128-token slices)")
rep("  __shared__ int s_e[32], s_off[33], s_item[NSTG], s_nh;",
    "  __shared__ int s_e[64], s_so[64], s_ns[64], s_sz[64], s_off[33], s_item[NSTG], s_nh;  // proj n21: slice table (expert, token offset, slices)")
rep("""    const bool h = nt > (NH > 64 ? 64 : NH / 2) && nt <= NH;
    const unsigned b = __ballot_sync(0xffffffffu, h);
    if (h) s_e[__popc(b & ((1u << tid) - 1))] = e;
    if (tid == 0) s_nh = __popc(b);""",
"""    const bool h = nt > (NH > 64 ? 64 : NH / 2) && (NH == 128 || nt <= NH);
    const int ns = h ? (nt + NH - 1) / NH : 0;  // proj n21 HOT_SL: >128-token experts as ceil(nt / 128) slices (NH 128 only)
    const unsigned b = __ballot_sync(0xffffffffu, h), bm = __ballot_sync(0xffffffffu, ns > 1);
    if (!bm) {  // n21 fast path (no expert > 128 tokens): the old ballot compaction
      const int p = __popc(b & ((1u << tid) - 1));
      if (h) { s_e[p] = e; s_so[p] = 0; s_ns[p] = 1; s_sz[p] = NH; }
      if (tid == 0) s_nh = __popc(b);
    } else {
      int pos = ns;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) { const int y = __shfl_up_sync(0xffffffffu, pos, o); if (tid >= o) pos += y; }
      const int sz = ns > 1 ? (((nt + ns - 1) / ns + 15) & ~15) : NH;  // n21c: balanced slices (167 -> 2 x 96 rows, not 128 + 39)
      for (int q = 0; q < ns; ++q) { s_e[pos - ns + q] = e; s_so[pos - ns + q] = q * sz; s_ns[pos - ns + q] = ns; s_sz[pos - ns + q] = sz; }
      if (tid == 31) s_nh = pos;
    }""")
rep("      const int a = it / nrb, rb = it - a * nrb, e = s_e[a], off = s_off[e];",
    "      const int a = it / nrb, rb = it - a * nrb, e = s_e[a], off = s_off[e] + s_so[a];")
rep("          if (v >= HG::NRB * NEEDK) break;", "          if (v >= HG::NRB * NEEDK * s_ns[a]) break;")
rep("    const int a = it / nrb, rb = it - a * nrb, e = s_e[a], off = s_off[e], nv = s_off[e + 1] - off;",
    "    const int a = it / nrb, rb = it - a * nrb, e = s_e[a], off = s_off[e] + s_so[a], nv = min(s_sz[a], s_off[e + 1] - off);")
rep("        if (atomicAdd(done + 32 + e, 1) == HN::NRB - 1) atomicAdd(done + 32 + e, 1 << 20);",
    "        if (atomicAdd(done + 32 + e, 1) == HN::NRB * s_ns[a] - 1) atomicAdd(done + 32 + e, 1 << 20);")
open(sys.argv[2], "w").write(s)
print("ok")
