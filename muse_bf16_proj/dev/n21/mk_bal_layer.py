"""proj n21: on top of a HOT_SL layer submission (dev/n20/mk_hotsl.py output): ballot fast path when no expert has > 128
tokens + balanced slices (ceil(nt / ns) rounded up to 16 rows: 167 -> 2 x 96, not 128 + 39).
python mk_bal_layer.py <hotsl submission.py> <out.py>"""
import sys
s = open(sys.argv[1]).read()
def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (a, s.count(a))
    s = s.replace(a, b)
rep("s_ns[64], s_off[33]", "s_ns[64], s_sz[64], s_off[33]")
rep("""    int pos = ns;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) { const int y = __shfl_up_sync(0xffffffffu, pos, o); if (tid >= o) pos += y; }
    for (int q = 0; q < ns; ++q) { s_e[pos - ns + q] = e; s_so[pos - ns + q] = q * NH; s_ns[pos - ns + q] = ns; }
    if (tid == 31) s_nh = pos;""",
"""    const unsigned b = __ballot_sync(0xffffffffu, h), bm = __ballot_sync(0xffffffffu, ns > 1);
    if (!bm) {  // proj n21 fast path (no expert > 128 tokens): the old ballot compaction
      const int p = __popc(b & ((1u << tid) - 1));
      if (h) { s_e[p] = e; s_so[p] = 0; s_ns[p] = 1; s_sz[p] = NH; }
      if (tid == 0) s_nh = __popc(b);
    } else {
      int pos = ns;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) { const int y = __shfl_up_sync(0xffffffffu, pos, o); if (tid >= o) pos += y; }
      const int sz = ns > 1 ? (((nt + ns - 1) / ns + 15) & ~15) : NH;  // proj n21: balanced slices (167 -> 2 x 96 rows, not 128 + 39)
      for (int q = 0; q < ns; ++q) { s_e[pos - ns + q] = e; s_so[pos - ns + q] = q * sz; s_ns[pos - ns + q] = ns; s_sz[pos - ns + q] = sz; }
      if (tid == 31) s_nh = pos;
    }""")
rep("nv = min(NH, s_off[e + 1] - off);", "nv = min(s_sz[a], s_off[e + 1] - off);")
open(sys.argv[2], "w").write(s)
print("ok")
