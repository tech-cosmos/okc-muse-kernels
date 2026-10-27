# n21 layer port: moe_twq K-split tail into the layer's moe_src.py (works with or without n19's WBW code).
# python tools/n21/patch_layer_twq.py <path/to/moe_src.py>   then TUNE option e.g. {"twq": 4 | 33 << 8 | 1 << 16} at T=128
import sys, re
p = sys.argv[1]; s = open(p).read()
src = open(__file__.replace("patch_layer_twq.py", "../../submission.py")).read()
def rep(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (s.count(old), old[:100]); s = s.replace(old, new)
def blk(a, b):  # text of r-moe submission.py from anchor a through anchor b (inclusive)
    i = src.index(a); j = src.index(b, i) + len(b); return src[i:j]
# 1) tq_item: whole function from r-moe (it has no WBW parts)
a, b = "template <int N, int NSTG, bool SA>\n__device__ __forceinline__ void tq_item(", "  else tw_epi<2, N, true, 16>(acc, lane, warp, nv, off, rb, gsc, sroutes, Out);\n}"
i = s.index(a); j = s.index(b, i) + len(b); s = s[:i] + blk(a, b) + s[j:]
# 2) kernel signature / shared / split setup
rep("    __nv_bfloat16* __restrict__ Y, float* __restrict__ part, int flags) {\n  using C = TWC<4, true, 8>;",
    "    __nv_bfloat16* __restrict__ Y, float* __restrict__ part, int flags, int* __restrict__ tick) {\n  using C = TWC<4, true, 8>;")
rep("  __shared__ int s_e[32], s_off[33], s_cum[2][33], s_item[NSTG], s_big[32];\n",
    "  __shared__ int s_e[32], s_off[33], s_cum[2][33], s_item[NSTG], s_big[32], s_flag;\n")
rep("  for (int i = tid; i < 256; i += 640) slut[i] = lut[i];\n",
    "  // n21 K split: the last ~(flags >> 8 & 255) gemm2 items (from an expert boundary) run as 2^(flags >> 16 & 3) K parts\n"
    "  const int kq = (flags >> 16) & 3, npt = 1 << kq;\n  int nsplit = kq ? (flags >> 8) & 255 : 0;\n"
    "  for (int i = tid; i < 256; i += 640) slut[i] = lut[i];\n")
a, b = "  const int n13 = s_cum[0][32], n2 = s_cum[1][32];\n  int n2f = n2;", "    const int* cum = s_cum[g2];"
rep("  const int n13 = s_cum[0][32], nitems = n13 + s_cum[1][32];\n  auto decode = [&](int item, int& g2, int& a, int& p, int& rb) {\n"
    "    g2 = item >= n13;\n    const int it = g2 ? item - n13 : item;\n    const int* cum = s_cum[g2];", blk(a, b))
# 3) producer + consumer, inside the moe_twq kernel only
k0 = s.index("__global__ void __launch_bounds__(640, 1) moe_twq("); k1 = s.index("static int64_t twq_go(", k0)
K = s[k0:k1]
def krep(old, new, cnt=1):
    global K
    assert K.count(old) == cnt, (K.count(old), old[:100]); K = K.replace(old, new)
krep("      int g2, a, p, rb;\n      decode(item, g2, a, p, rb);\n", "      int g2, a, p, rb, h, j;\n      decode(item, g2, a, p, rb, h, j);\n")
krep("      const int mi = nv <= 8 ? 0 : nv <= 16 ? 1 : nv <= 32 ? 2 : 3;\n",
     "      const int k0 = h < 0 ? 0 : h * (64 >> kq), k1 = h < 0 ? 64 : k0 + (64 >> kq);\n"
     "      const int mi = nv <= 8 ? 0 : nv <= 16 ? 1 : nv <= 32 ? 2 : 3;\n")
krep("      for (int kc = 0; kc < 64; ++kc, ++i) {\n        if (kc && i >= NSTG)", "      for (int kc = k0; kc < k1; ++kc, ++i) {\n        if (kc != k0 && i >= NSTG)")
krep("        if (!kc) s_item[st] = item;", "        if (kc == k0) s_item[st] = item;")
krep("    int g2, a, p, rb;\n    decode(item, g2, a, p, rb);\n", "    int g2, a, p, rb, h, j;\n    decode(item, g2, a, p, rb, h, j);\n")
krep("    __nv_bfloat16* out = g2 ? Y : Inter;\n", "    __nv_bfloat16* out = g2 ? Y : Inter;\n    float* pt = h < 0 ? nullptr : part + (size_t)j * npt * 16384;\n")
krep("sroutes, out, g2)\n", "sroutes, out, g2, \\\n                                      h < 0 ? 64 : 64 >> kq, pt, tick + j, &s_flag, h, npt)\n")
s = s[:k0] + K + s[k1:]
# 4) host: flags clamp + tick argument, binding, python
a, b = "// n21: clamp the K-split", "  return (f & ~((int64_t)0x3FF00)) | (m << 8) | (m ? kq << 16 : 0);\n}\n"
rep("template <bool SA>\nstatic int64_t twq_go(", blk(a, b) + "template <bool SA>\nstatic int64_t twq_go(")
rep("torch::Tensor inter_, torch::Tensor y_, torch::Tensor part_, int64_t flags) {", "torch::Tensor inter_, torch::Tensor y_, torch::Tensor part_, torch::Tensor tk_, int64_t flags) {", 2)
rep("(float*)part_.data_ptr(), (int)flags);", "(float*)part_.data_ptr(), (int)twq_flags(flags, part_.numel(), tk_.numel()), (int*)tk_.data_ptr());")
rep("inter_, y_, part_, flags\n", "inter_, y_, part_, tk_, flags\n")
T13 = '"torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t);"'
i = s.index('"int64_t moe_twq_launch('); j = s.index(T13, i)
s = s[:j] + T13.replace('torch::Tensor, int64_t', 'torch::Tensor, torch::Tensor, int64_t') + s[j + len(T13):]
rep("                if DIAG_TL and TWM_ORDER_M.get(M, TWM_ORDER) & 2: st[\"_tl\"] = (part, meta); q |= 2\n",
    "                qs, qk = (q >> 8) & 255, (q >> 16) & 3  # n21: K-split tail in the flags: q | items << 8 | log2 parts << 16\n"
    "                if qs and qk and not DIAG_TL:\n"
    "                    part = torch.empty((qs << qk) * 16384, dtype=torch.float32, device=dev)\n"
    "                if DIAG_TL and TWM_ORDER_M.get(M, TWM_ORDER) & 2: st[\"_tl\"] = (part, meta); q |= 2\n")
rep("meta, sroutes, inter, y, part, q)", "meta, sroutes, inter, y, part, st[\"tk\"], q)")
open(p, "w").write(s)
print("patched", p)
