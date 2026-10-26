"""Build submission.py = attention kernels (attn_src.py, from ~/attn submission.py) + the layer (layer.py).
The judge takes one file, so the attention module is inlined with its kernel_fn renamed attn_kernel_fn."""
import re, sys, pathlib
import ast, io, tokenize


def moe_namespaced(src):
    """The moe agent's submission with every top-level Python name prefixed _moe_ (tokens only: strings,
    including the CUDA source, and attributes stay as they are)."""
    src = src.split("\n", 1)[1]
    tree = ast.parse(src)
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    out, prev = [], None
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.NAME and tok.string in names and not (prev and prev.string == "."):
            tok = tok._replace(string="_moe_" + tok.string)
        out.append(tok)
        if tok.type not in (tokenize.NL, tokenize.COMMENT):
            prev = tok
    return tokenize.untokenize(out)


here = pathlib.Path(__file__).parent
attn = (here / "attn_src.py").read_text()
layer = (here / "layer.py").read_text()
attn = attn.split("\n", 1)[1]  # drop its "# okc: problem=" line
assert attn.count("\ndef kernel_fn(") == 1
attn = attn.replace("\ndef kernel_fn(", "\ndef attn_kernel_fn(")
# In the judge image SGLang's import leaves triton.next_power_of_2 a plain function, which a @triton.jit body
# can't call (check 7e3e63a1): use our own constexpr function inside kernels.
jit_np2 = "PF_N: tl.constexpr = triton.next_power_of_2("
assert attn.count(jit_np2) == 1
attn = attn.replace(jit_np2, "PF_N: tl.constexpr = _np2_jit(")
attn = attn.replace("\nclass Config(TypedDict):", """
from triton.runtime.jit import constexpr_function as _constexpr_function


@_constexpr_function
def _np2_jit(n):
    return 1 << max(0, int(n) - 1).bit_length()


class Config(TypedDict):""", 1)
# csl::decode (decode 1 sliding) triggers its dependents at start: the o GEMV (a PDL secondary) issues its first
# weight loads during attention.
csl_anchor = "    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3;\n    const int h = (blockIdx.x >> 4) & 1, wk = warp;\n"
assert attn.count(csl_anchor) == 2  # csl and its verify twin (no PDL dependent there)
attn = attn.replace(csl_anchor, csl_anchor + '    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n')
moe = moe_namespaced((here / "moe_src.py").read_text())
# s12 UPF: moe_tk2f's producer warp L2-prefetches its 1/grid share of a buffer (the unpool slice) right after it has
# issued its last weight stage, so the unpool GEMV after the experts reads warm L2. moe_set_upf(t, bytes) sets it
# (host globals, captured as kernel args at launch); bytes 0 = off (the default: moe's own calls are unchanged).
def _patch(src, a, b, n=1):
    assert src.count(a) == n, a
    return src.replace(a, b)
moe = _patch(moe, "int64_t moe_set_pdl(int64_t v, int64_t twpf) {",
             "static const char* g_upf = nullptr;\nstatic int g_upfb = 0;\n"
             "int64_t moe_set_upf(torch::Tensor t, int64_t bytes) { g_upf = bytes ? (const char*)t.data_ptr() : nullptr; g_upfb = (int)bytes; return 0; }\n"
             "int64_t moe_set_pdl(int64_t v, int64_t twpf) {")
moe = _patch(moe, "__nv_bfloat16* __restrict__ out, int M, float eps) {\n  static_assert(NC % 8 == 0,",
             "__nv_bfloat16* __restrict__ out, int M, float eps, const char* __restrict__ upf, int upfb) {\n  static_assert(NC % 8 == 0,")
_a = """  tk2f_consume<256, NC, NSTG, false, KPS, SA>(slut, gs2, inter, Y, cnt, n13, b0, b1, s_rs2, s_e, s_off, s_srt, s_red, smem, full,
                                     empty, st, ph, rb);
"""
moe = _patch(moe, _a, _a + """  if (upfb && tid == 0) {
    const int c = (upfb / (int)gridDim.x + 15) & ~15, o = (int)blockIdx.x * c, n = min(c, upfb - o);
    if (n > 0) prefetch_l2(upf + o, n);
  }
""")
moe = _patch(moe, "(__nv_bfloat16*)out, nr / 8, eps);", "(__nv_bfloat16*)out, nr / 8, eps, g_upf, g_upfb);")
# s12 XN: the pre-expert rmsnorm (x r (1 + w), BF16, as _rmsnorm_gain) inside moe_tk2f: the consumer warps compute r of
# every token from the raw BF16 pooler output and copy the gain to smem in the kernel prologue (while the producer's
# first weight copies are in flight); gemm13's x loads apply it. No rmsnorm launch. moe_set_xn(gain f32 [4096], eps,
# on) (host globals; off by default).
moe = _patch(moe, "static const char* g_upf = nullptr;", "static const float* g_xng = nullptr;\nstatic float g_xeps = 0.f;\n"
             "int64_t moe_set_xn(torch::Tensor g, double eps, int64_t on) { g_xng = on ? (const float*)g.data_ptr() : nullptr; g_xeps = (float)eps; return 0; }\n"
             "static const char* g_upf = nullptr;")
moe = _patch(moe, "float eps, const char* __restrict__ upf, int upfb) {", "float eps, const char* __restrict__ upf, int upfb, const float* __restrict__ xng, float xeps) {")
moe = _patch(moe, "eps, g_upf, g_upfb);", "eps, g_upf, g_upfb, g_xng, g_xeps);")
moe = _patch(moe, """                                             float* s_red, uint4* smem, uint64_t* full, uint64_t* empty, int& st,
                                             int& ph, int& rb) {
""", """                                             float* s_red, uint4* smem, uint64_t* full, uint64_t* empty, int& st,
                                             int& ph, int& rb, const float* xg_s = nullptr, const float* xr_s = nullptr) {
""")
_a = """          xf[qq][1] = v ? __ldg(xr + kc * 8 + 1) : make_uint4(0, 0, 0, 0);
        }
      }
"""
moe = _patch(moe, _a, _a + """      if (GATED && xg_s != nullptr) {  // XN: x r[token] (1 + w), rounded to BF16 (r and the gain: kernel prologue, smem)
        const float r = v ? xr_s[row] : 0.f;
#pragma unroll
        for (int qq = 0; qq < NQ; ++qq) {
          const int kc = qq * NC + warp;
#pragma unroll
          for (int hh = 0; hh < 2; ++hh) {
            const float* gv = xg_s + kc * 64 + 16 * t + 8 * hh;
            uint32_t* u = reinterpret_cast<uint32_t*>(&xf[qq][hh]);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              const float lo = __uint_as_float(u[j] << 16) * r * gv[2 * j];
              const float hi = __uint_as_float(u[j] & 0xFFFF0000u) * r * gv[2 * j + 1];
              const __nv_bfloat162 b = __floats2bfloat162_rn(lo, hi);
              u[j] = *reinterpret_cast<const uint32_t*>(&b);
            }
          }
        }
      }
""")
_a = """  float rt = 0.f, rg = 0.f;  // FN: route tid's topk weight and gs2 * 256 (0: not local), used by tk2f_norm
"""
moe = _patch(moe, _a, """  __shared__ __align__(16) float s_xg[4096];
  __shared__ float s_xr[32];
  if (NC == 8 && xng != nullptr) {  // all NC warps: thread tid sums elements [8 tid, 8 tid + 8) and [8 (tid + 256), ...) of a token
    float4 gq[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) gq[i] = __ldg(reinterpret_cast<const float4*>(xng) + tid + i * NC * 32);
    float* s_xp = s_red;  // [NC][32] partial sums (s_red is free until the first row's reduction)
    for (int t0 = 0; t0 < M; t0 += 8) {
      uint4 q[8][2];
#pragma unroll
      for (int k = 0; k < 8; ++k)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh)
          q[k][hh] = t0 + k < M ? __ldg(reinterpret_cast<const uint4*>(X + (size_t)(t0 + k) * KD) + tid + hh * NC * 32)
                                : make_uint4(0, 0, 0, 0);
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        float ss = 0.f;
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const uint32_t u[4] = {q[k][hh].x, q[k][hh].y, q[k][hh].z, q[k][hh].w};
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const float lo = __uint_as_float(u[j] << 16), hi = __uint_as_float(u[j] & 0xFFFF0000u);
            ss = fmaf(lo, lo, ss);
            ss = fmaf(hi, hi, ss);
          }
        }
#pragma unroll
        for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        if (lane == 0) s_xp[warp * 32 + t0 + k] = ss;
      }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) reinterpret_cast<float4*>(s_xg)[tid + i * NC * 32] = gq[i];
    asm volatile("bar.sync 2, %0;" ::"n"(NC * 32) : "memory");
    if (tid < M) {
      float tot = 0.f;
#pragma unroll
      for (int w = 0; w < NC; ++w) tot += s_xp[w * 32 + tid];
      s_xr[tid] = rsqrtf(tot / (float)KD + xeps);
    }
    asm volatile("bar.sync 2, %0;" ::"n"(NC * 32) : "memory");
  }
""" + _a)
moe = _patch(moe, """  tk2f_consume<512, NC, NSTG, true, KPS, SA>(slut, gs13, X, inter, cnt, n13, a0, a1, s_rs13, s_e, s_off, s_srt, s_red, smem, full,
                                    empty, st, ph, rb);""", """  tk2f_consume<512, NC, NSTG, true, KPS, SA>(slut, gs13, X, inter, cnt, n13, a0, a1, s_rs13, s_e, s_off, s_srt, s_red, smem, full,
                                    empty, st, ph, rb, NC == 8 && xng ? s_xg : nullptr, s_xr);""")
moe = _patch(moe, '"int64_t moe_set_sa(int64_t);")', '"int64_t moe_set_sa(int64_t);"\n        "int64_t moe_set_upf(torch::Tensor, int64_t);"\n        "int64_t moe_set_xn(torch::Tensor, double, int64_t);")')
moe = _patch(moe, '"moe_set_blk", "moe_set_sa"]', '"moe_set_blk", "moe_set_sa", "moe_set_upf"]')
moe = _patch(moe, '"moe_set_sa", "moe_set_upf"]', '"moe_set_sa", "moe_set_upf", "moe_set_xn"]')
layer = layer.replace("# @@MOE@@\n", "# ---- experts (moe_src.py, names prefixed _moe_) ----\n" + moe + "\n# ---- layer ----\n")
g10 = (here / "g10_src.cu").read_text()  # proj's gemv10 CUDA source, as a raw string literal in the layer
Q3 = "'" * 3
assert Q3 not in g10 and layer.count("_G10_SRC = None  # @@G10@@\n") == 1
layer = layer.replace("_G10_SRC = None  # @@G10@@\n", "_G10_SRC = r" + Q3 + g10 + Q3 + "\n")
g12 = (here / "g12_src.cu").read_text()  # proj's gemv12 (m = 8)
assert Q3 not in g12 and layer.count("_G12_SRC = None  # @@G12@@\n") == 1
layer = layer.replace("_G12_SRC = None  # @@G12@@\n", "_G12_SRC = r" + Q3 + g12 + Q3 + "\n")
for _n in ("13", "14", "19", "PR3", "PR1", "PR8"):  # proj's gemv13 / gemv14 (compressed BF16 weights), router3 (hi/lo, T 9..128)
    _s = (here / (f"g{_n}_src.cu" if _n.isdigit() else f"{_n.lower()}_src.cu")).read_text()
    assert Q3 not in _s and layer.count(f"_G{_n}_SRC = None  # @@G{_n}@@\n") == 1
    layer = layer.replace(f"_G{_n}_SRC = None  # @@G{_n}@@\n", f"_G{_n}_SRC = r" + Q3 + _s + Q3 + "\n")
lt = (here / "lt_src.cu").read_text()  # proj's cuBLASLt GEMM with an explicit algo config
assert Q3 not in lt and layer.count("_LT_SRC = None  # @@LT@@\n") == 1
layer = layer.replace("_LT_SRC = None  # @@LT@@\n", "_LT_SRC = r" + Q3 + lt + Q3 + "\n")
out = "# okc: problem=muse_decode_layer\n" + layer.replace("# @@ATTENTION@@\n", "# ---- attention (attn_src.py) ----\n" + attn + "\n# ---- layer ----\n")
assert "@@ATTENTION@@" not in out and "@@MOE@@" not in out and "attn_kernel_fn" in out
(here / (sys.argv[1] if len(sys.argv) > 1 else "submission.py")).write_text(out)
