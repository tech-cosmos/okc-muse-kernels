# s24 layer port: NORM_EARLY for moe_twq (moe_norme per-token waits on per-expert gemm2 flags, so the post-expert norm runs in
# twq's drain). python tools/s24/patch_layer_ne.py <path/to/moe_src.py>   (needs n21's twq K split; s23 HOTQ optional)
# Then: twq flags | 1 << 21 turns it on, e.g. TUNE[128] {"twq": 536953348 | 1 << 21} (= 539050500) or {"twq": 73988 | 1 << 21}
# (= 2171140). Bitwise = without it (r-moe tools/s24/ne_t.py 0/78). moe_src.py must have NORM_EARLY 1 (layer: yes).
import sys
p = sys.argv[1]; s = open(p).read()
def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (s.count(a), a[:100]); s = s.replace(a, b)
k0 = s.index("__global__ void __launch_bounds__(640, 1) moe_twq("); k1 = s.index("static int64_t twq_go(", k0)
K = s[k0:k1]
for a, P in (("  const int ns = n2 - n2f, nitems = n13 + n2f + npt * ns;\n", "npt"),
             ("  const int ns = n2 - n2f, nitems = n13 + n2f + 2 * ns;\n", "2")):  # moe n21 general / layer's halves-only copy
    if K.count(a) == 1: break
assert K.count(a) == 1, "twq nitems anchor"
K = K.replace(a, a + """#if NORM_EARLY
  __shared__ int s_nd2[32];  // s24: gemm2 item ends of slot tid (K-split items count once per part), as moe_twm's
  if (tid < 32) {
    const int c0 = s_cum[1][tid], c1 = s_cum[1][tid + 1];
    s_nd2[tid] = (c1 - c0) + (%s - 1) * max(0, c1 - max(c0, n2f));
  }
  __syncthreads();
#endif
""" % P)
s = s[:k0] + K + s[k1:]
b = """    if (!g2) {
      asm volatile("fence.proxy.async.global;" ::: "memory");
      asm volatile("bar.sync 1, 512;" ::: "memory");
      if (tid == 0) {
        __threadfence();
        atomicAdd(done + e, 1);
      }
    }
"""
rep(b, b + """#if NORM_EARLY
    else {  // s24: Y rows of expert e complete after its last gemm2 item end (a split's last part writes Y before it counts)
      asm volatile("bar.sync 1, 512;" ::: "memory");
      if (tid == 0) {
        __threadfence();
        if (atomicAdd(done + 32 + e, 1) == s_nd2[a] - 1) atomicAdd(done + 32 + e, 1 << 20);
      }
    }
#endif
""")
c = """                lib.moe_twq_launch(st["w13b4"], st["w13sb4"], st["gs13"], st["w2b4"], st["w2sb4"], st["gs2"], st["lut"], xg,
                                   meta, sroutes, inter, y, part, st["tk"], q)
"""
k = s.index(c); i = s.rindex("            if TWQ_M.get(M, TWQ):\n", 0, k)
blk = s[i:k]
d = "                q = TWQ_M.get(M, TWQ)\n"
assert blk.count(d) == 1
blk = blk.replace(d, d + "                ne = (q >> 21) & 1  # s24: twq flags | 1 << 21 = moe_norme after twq\n"
                           "                q &= ~(7 << 21)\n")
s = s[:i] + blk + c + """                if NORM_EARLY and ne:  # s24: moe_norm waits per token on twq's (and moe_hot's) expert flags
                    lib.moe_norme_launch(y, ids, tw, st["w1"], st["start"], st["count"], POST_NORM_EPS, out, meta)
                    return out
""" + s[k + len(c):]
open(p, "w").write(s)
print("patched", p)
