"""n75 QLO (human-approved 11-01 23:45Z): two-term E4M3 queries in vjx's q_len-8 kernels (verify full layers):
q = e4m3(q) + e4m3(q - e4m3(q)), QK^T = q_hi K^T + q_lo K^T, both FP32-accumulated wgmmas into the same scores
(fresh x16's nvs/GW scheme ported to vjx). Decode (q_len 1) runs FP16 queries and is untouched.
- per-request kernels (QDIR, v1): the lo plane is a second 64x64 e4m3 A tile per KV head in shared memory (SS wgmma);
- flat kernel (WS, v16/v32): no shared memory left at 4 stages (2 CTAs/SM at 2 x 114688 B): ss3 (default) cuts its
  pipeline to 3 stages (H100 standalone: 3 vs 4 stages tie, base3) and keeps the lo plane in shared memory too; rs keeps
  the lo A fragments in registers (RS wgmma; spills 36 B, v16/v32 full +9/+12 us: not used).
  H100 standalone (dev/n75/ab.py, seeds 99/7, 4 rounds): v1/v16/v32 full ss3 vs base +0.0/+0.0/+0.1 us; attention
  output rel. error vs the BF16-query reference 7.4-7.7e-2 -> 1.6-2.0e-2.
patch(src) -> patched CUDA source (apply after vjx_dyn.patch)."""


def _rep(s, a, b, n=1):
    assert s.count(a) == n, (s.count(a), a[:80])
    return s.replace(a, b)


def patch(S, flat="ss3"):
    """flat: "rs" = lo A fragments in registers (RS wgmma); "ss3" = lo plane in shared memory like the per-request
    kernels, the flat pipeline cut from 4 to 3 stages to make room; "base3" = no QLO, only the 3-stage flat pipeline (A/B)."""
    if flat in ("ss3", "base3"):
        S = _rep(S, "  static constexpr int STAGES = F ? ((L == 1) ? 3 : 4) : 3;\n",
                 "  static constexpr int STAGES = 3;   // (QLO ss3: the flat q_len-8 pipeline 4 -> 3 stages)\n")
    if flat == "base3":
        return S
    ss = flat == "ss3"
    S = _rep(S, "  alignas(1024) uint8_t q8s[(L == 8) ? 8192 : 16];\n",
             "  alignas(1024) uint8_t q8s[(L == 8) ? (F ? %s : 16384) : 16];   // (QLO: + the lo plane)\n" % (16384 if ss else 8192))
    # helper: e4m3 bytes of the residuals x - e4m3(x) (hi = the e4m3 bytes of x0..x3, x0 in the low byte)
    S = _rep(S, "// four f16 (two f16x2 words, low first) -> four e4m3 bytes (low byte first)\n",
             "// QLO: four floats and their e4m3 bytes -> the e4m3 bytes of the residuals x - e4m3(x)\n"
             "__device__ __forceinline__ uint32_t e4m3x4_res(float x0, float x1, float x2, float x3, uint32_t hi) {\n"
             "  const uint32_t d01 = cvt2(hi), d23 = cvt2h(hi);\n"
             "  const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&d01));\n"
             "  const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&d23));\n"
             "  return e4m3x4f(x0 - a.x, x1 - a.y, x2 - b.x, x3 - b.y);\n"
             "}\n"
             "// four f16 (two f16x2 words, low first) -> four e4m3 bytes (low byte first)\n")
    # per-request kernels (QDIR): the lo plane at q8s + 8192 (same [head][row][dim] swizzled layout)
    a = """      *reinterpret_cast<uint4*>(S.q8s + hk4 * 4096 + r4 * 64 + (((kb >> 4) ^ ((r4 >> 1) & 3)) << 4)) =
          make_uint4(e4m3x4f(x[0], x[1], x[2], x[3]), e4m3x4f(x[4], x[5], x[6], x[7]),
                     e4m3x4f(x[8], x[9], x[10], x[11]), e4m3x4f(x[12], x[13], x[14], x[15]));
"""
    S = _rep(S, a, """      const uint4 qh8 = make_uint4(e4m3x4f(x[0], x[1], x[2], x[3]), e4m3x4f(x[4], x[5], x[6], x[7]),
                                   e4m3x4f(x[8], x[9], x[10], x[11]), e4m3x4f(x[12], x[13], x[14], x[15]));
      *reinterpret_cast<uint4*>(S.q8s + hk4 * 4096 + r4 * 64 + (((kb >> 4) ^ ((r4 >> 1) & 3)) << 4)) = qh8;
      *reinterpret_cast<uint4*>(S.q8s + 8192 + hk4 * 4096 + r4 * 64 + (((kb >> 4) ^ ((r4 >> 1) & 3)) << 4)) =
          make_uint4(e4m3x4_res(x[0], x[1], x[2], x[3], qh8.x), e4m3x4_res(x[4], x[5], x[6], x[7], qh8.y),
                     e4m3x4_res(x[8], x[9], x[10], x[11], qh8.z), e4m3x4_res(x[12], x[13], x[14], x[15], qh8.w));
""")
    # flat kernel (WS): lo A fragments in registers, residuals of the f16 staging (exact in f16) rounded to e4m3
    S = _rep(S, "  uint32_t qa8[2][4];   // e4m3 A fragments, k-steps of 32 dims\n",
             "  uint32_t qa8[2][4];   // e4m3 A fragments, k-steps of 32 dims\n"
             "  uint32_t ql8[2][4];   // (QLO, flat kernel) the lo plane's A fragments: e4m3(q - e4m3(q))\n")
    a = """        qa8[ks][2 * hf] = pack_e4m3x4(lo.x, lo.y);
        qa8[ks][2 * hf + 1] = pack_e4m3x4(hi.x, hi.y);
"""
    S = _rep(S, a, a + """        {
          const uint32_t a0 = qa8[ks][2 * hf], a1 = qa8[ks][2 * hf + 1];
          uint32_t d0 = cvt2(a0), d1 = cvt2h(a0), d2 = cvt2(a1), d3 = cvt2h(a1);
          const __half2 r0 = __hsub2(*reinterpret_cast<const __half2*>(&lo.x), *reinterpret_cast<const __half2*>(&d0));
          const __half2 r1 = __hsub2(*reinterpret_cast<const __half2*>(&lo.y), *reinterpret_cast<const __half2*>(&d1));
          const __half2 r2 = __hsub2(*reinterpret_cast<const __half2*>(&hi.x), *reinterpret_cast<const __half2*>(&d2));
          const __half2 r3 = __hsub2(*reinterpret_cast<const __half2*>(&hi.y), *reinterpret_cast<const __half2*>(&d3));
          ql8[ks][2 * hf] = pack_e4m3x4(*reinterpret_cast<const uint32_t*>(&r0), *reinterpret_cast<const uint32_t*>(&r1));
          ql8[ks][2 * hf + 1] = pack_e4m3x4(*reinterpret_cast<const uint32_t*>(&r2), *reinterpret_cast<const uint32_t*>(&r3));
        }
""")
    if ss:
        S = _rep(S, LOSTORE_A, LOSTORE_A + LOSTORE_B)
    a = """       wgmma_m64n64k32_e4m3_ss(s, qdesc0, kd, 0);
      wgmma_m64n64k32_e4m3_ss(s, qd2, kd2, 1);
"""
    S = _rep(S, a, a + """      if constexpr (%s) {   // QLO: + q_lo K^T (lo plane 8192 bytes up: +512 in the descriptor's address field)
        wgmma_m64n64k32_e4m3_ss(s, qdesc0 + 512u, kd, 1);
        wgmma_m64n64k32_e4m3_ss(s, qd2 + 512u, kd2, 1);
      } else {
        wgmma_m64n64k32_e4m3(s, ql8[0][0], ql8[0][1], ql8[0][2], ql8[0][3], kd, 1);
        wgmma_m64n64k32_e4m3(s, ql8[1][0], ql8[1][1], ql8[1][2], ql8[1][3], kd2, 1);
      }
""" % ("true" if ss else "QDIR"))
    if "#include <cuda_fp16.h>" not in S:
        S = "#include <cuda_fp16.h>\n" + S
    return S


LOSTORE_A = """          *reinterpret_cast<uint32_t*>(S.q8s + hk * 4096 + r1 * 64 + (((kb >> 4) ^ ((r1 >> 1) & 3)) << 4) + (kb & 15)) =
              qa8[ks][2 * hf + 1];
"""
LOSTORE_B = """          *reinterpret_cast<uint32_t*>(S.q8s + 8192 + hk * 4096 + r0 * 64 + (((kb >> 4) ^ ((r0 >> 1) & 3)) << 4) + (kb & 15)) =
              ql8[ks][2 * hf];
          *reinterpret_cast<uint32_t*>(S.q8s + 8192 + hk * 4096 + r1 * 64 + (((kb >> 4) ^ ((r1 >> 1) & 3)) << 4) + (kb & 15)) =
              ql8[ks][2 * hf + 1];
"""
