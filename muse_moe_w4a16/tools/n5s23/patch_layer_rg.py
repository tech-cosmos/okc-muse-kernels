"""n5s23: port moe's routegather latency fix into r-layer's moe_routegathern (layer.py). Usage: python patch_layer_rg.py <layer.py>
(1) the token's x row (+ sum of squares) and its own 8 ids are loaded right after griddepcontrol, BEFORE the id scan;
(2) the scan loads RGU=8 ids per thread in one batch before the __match_any_sync loop (old: one serial L2 trip per 256 ids);
(3) repeated-id fixup uses __shfl_sync of the preloaded ids (old: up to 8 serial __ldcg). Integer counts: bitwise-identical output."""
import sys
p = sys.argv[1]; s = open(p).read()
a = s.index("__global__ void __launch_bounds__(256) moe_routegathern("); b = s.index("int64_t moe_routegathern_launch", a); k = s[a:b]
scan_old = k[k.index("  const int lim = tok * 8;\n  for (int i = tid;"):k.index("  // this token's row: load + sum of squares")]
row_old = k[k.index("  // this token's row: load + sum of squares"):k.index("#pragma unroll\n  for (int o = 16; o; o >>= 1) ss +=")]
scan_new = '''  for (int b = 0; b < nroutes; b += 256 * 8) {  // uniform trip count: __match_any_sync below
    int ev[8];
#pragma unroll
    for (int u = 0; u < 8; ++u) { const int i = b + 256 * u + tid; ev[u] = i < nroutes ? __ldcg(ids + i) : -1; }
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      const int i = b + 256 * u + tid, e = ev[u];
      const bool v = e >= 0 && e < count;
      const unsigned m = __match_any_sync(0xffffffffu, v ? e : -1);
      const unsigned mp = __match_any_sync(0xffffffffu, (v && i < lim) ? e : -1);
      if (v && (__ffs(m) - 1) == lane) atomicAdd(&cnt[e], __popc(m));
      if (v && i < lim && (__ffs(mp) - 1) == lane) atomicAdd(&pre[e], __popc(mp));
    }
  }
'''
k = k.replace(row_old, "", 1)
k = k.replace(scan_old, "  const int lim = tok * 8;\n  const int mye = tid < 8 ? __ldcg(ids + lim + tid) : -1;\n" + row_old + scan_new, 1)
tail_old = k[k.index("  if (tid < 8) {\n    const int e = __ldcg(ids + lim + tid);"):k.index("  float tot = 0.f;")]
tail_new = '''  if (tid < 32) {
    const int e = mye;
    int pos = -1, rep = 0;
#pragma unroll
    for (int j = 0; j < 7; ++j) { const int ej = __shfl_sync(0xffffffffu, mye, j); rep += j < tid && ej == e; }
    if (tid < 8 && e >= 0 && e < count) { pos = pre[e] + rep; sroutes[pos] = lim + tid; }
    if (tid < 8) s_pos[tid] = pos;
  }
'''
k = k.replace(tail_old, tail_new, 1)
s = s[:a] + k + s[b:]
open(p, "w").write(s); print("patched", p)
