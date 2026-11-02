"""n78 OVL (lead 11-02 01:35Z: make QLO ~free): vjx flat q_len-8 kernel (WS, v16/v32 full) issues QK^T as two
key halves (keys 0-31 / 32-63: m64n32k32 wgmmas, hi + lo pairs each, FP32 accumulate, one commit group per half),
so on the common tiles (interior, after the segment's first tile) the softmax + P pack + PV of half 0 run while the
tensor cores still compute half 1's scores. Uses vjx's per-half PV path (offset check per half; the rare offset move
waits for half 0's PV before rescaling). First / masked / new-row tiles keep the whole-tile path.
patch(src) -> patched CUDA source (apply after vjx_qlo.patch(..., "ss3"))."""
import re


def _rep(s, a, b, n=1):
    """replace a by b; a's lines match with any trailing blanks (the source keeps blank-only lines)"""
    rx = re.compile("\n".join(re.escape(x.rstrip(" ")) + "[ ]*" for x in a.split("\n")))
    m = rx.findall(s)
    assert len(m) == n, (len(m), a[:80])
    return rx.sub(lambda _: b, s)


N32 = r"""// D[64x32] (+)= A[64x32] (e4m3, shared memory descriptor) * B[32x32]^T (e4m3, shared memory descriptor)
__device__ __forceinline__ void wgmma_m64n32k32_e4m3_ss(float* d, uint64_t da, uint64_t db, int scale_d) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %18, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, "
      "%16, %17, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
      : "l"(da), "l"(db), "r"(scale_d));
}
"""


def patch(S, test_hook=False, fast=True):
    S = _rep(S, "// K-major operand with 64-byte rows, 64B swizzle (8-row atoms of 512 bytes)\n",
             N32 + "// K-major operand with 64-byte rows, 64B swizzle (8-row atoms of 512 bytes)\n")
    # QK^T in two key halves (K rows 32..63 = 4 swizzle atoms of 1024 B up: +256 in the descriptor address)
    a = """       wgmma_m64n64k32_e4m3_ss(s, qdesc0, kd, 0);
      wgmma_m64n64k32_e4m3_ss(s, qd2, kd2, 1);
      if constexpr (true) {   // QLO: + q_lo K^T (lo plane 8192 bytes up: +512 in the descriptor's address field)
        wgmma_m64n64k32_e4m3_ss(s, qdesc0 + 512u, kd, 1);
        wgmma_m64n64k32_e4m3_ss(s, qd2 + 512u, kd2, 1);
      } else {
        wgmma_m64n64k32_e4m3(s, ql8[0][0], ql8[0][1], ql8[0][2], ql8[0][3], kd, 1);
        wgmma_m64n64k32_e4m3(s, ql8[1][0], ql8[1][1], ql8[1][2], ql8[1][3], kd2, 1);
      }

    wgmma_commit();
"""
    b = """    if constexpr (WS) {   // (OVL) two key halves, one commit group each (hi + lo q planes, FP32 accumulate)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        wgmma_m64n32k32_e4m3_ss(s + 16 * h, qdesc0, kd + 256u * h, 0);
        wgmma_m64n32k32_e4m3_ss(s + 16 * h, qd2, kd2 + 256u * h, 1);
        wgmma_m64n32k32_e4m3_ss(s + 16 * h, qdesc0 + 512u, kd + 256u * h, 1);
        wgmma_m64n32k32_e4m3_ss(s + 16 * h, qd2 + 512u, kd2 + 256u * h, 1);
        wgmma_commit();
      }
    } else {
""" + a + "    }\n"
    S = _rep(S, a, b)
    # common tiles: their own copy of the rest of the tile (wait for half 0 only; per-half pack + PV; half 1's wait
    # releases the stage). A separate copy, not a runtime switch inside one body: ptxas merges the in-flight wgmma
    # state of both paths at a join and then serializes every wgmma of the kernel (C7514).
    a0 = """    wgmma_wait<0>();
#pragma unroll
    for (int e = 0; e < 32; ++e) reg_fence(s[e]);
"""
    rx = re.compile(re.escape(a0))
    assert len(rx.findall(S)) == 1
    ia = S.index(a0)
    ib = S.index("    PROF_TILE;\n", ia)
    ib = S.rindex("\n", 0, ib) + 1   # (keeps the blank-only line before PROF_TILE outside)
    seg = S[ia:ib]
    f = _rep(seg, a0, """    wgmma_wait<1>();
#pragma unroll
    for (int e = 0; e < 16; ++e) reg_fence(s[e]);
""")
    f = _rep(f, """    if constexpr (WS) {   // K (wgmma) and V (conversion) of the stage are consumed
      mbar_arrive(&S.empty[st]);
    }
""", "")
    f = _rep(f, "      if constexpr (PK2) {\n", "      if constexpr (false) {   // (OVL: per-half path)\n")
    f = _rep(f, """      for (int kk = 0; kk < (PK2 ? 0 : 2); ++kk) {
        const float* sv = s + 16 * kk;
""", """      for (int kk = 0; kk < 2; ++kk) {
        if (kk == 1) {   // (OVL) half 1's scores (the groups left: QK half 1, PV half 0); the stage is consumed
          wgmma_wait<1>();
#pragma unroll
          for (int e = 16; e < 32; ++e) reg_fence(s[e]);
          if constexpr (WS) mbar_arrive(&S.empty[st]);
        }
        const float* sv = s + 16 * kk;
""")
    a = """          if constexpr (LSYNCN) {   // every warp's V^T for this tile written (and visible to the async proxy)
            if (kk == 0) {"""
    f = _rep(f, a, a.replace("if constexpr (LSYNCN)", "if constexpr (LSYNCN || LSYNC)"))
    S = S[:ia] + ("    // (OVL) common tiles: interior, no new rows, past the segment's first tile (offset check path)\n"
                  "    const bool ovl_f = %s && WS && !first && (((uint32_t)i - r_lo <= r_span) || (DYNC && i >= ntiles));\n" % ("true" if fast else "false") +
                  "    if (ovl_f) {\n" + f + "    } else {\n" + seg + "    }\n") + S[ib:]
    if test_hook:   # offset check at P > 1 instead of P > 256 (exercises the per-half offset move on most tiles)
        S = S.replace("constexpr uint32_t PB = 0x07070707u;", "constexpr uint32_t PB = 0x48484848u;")
    return S


QK_ISSUE = """      const uint32_t kl = klo0 + (uint32_t)(st_c * (STAGE_BYTES >> 4));
      const uint64_t kd = ((uint64_t)((1024 >> 4) | (1u << 30)) << 32) | kl;
      const uint64_t kd2 = ((uint64_t)((1024 >> 4) | (1u << 30)) << 32) | (kl + 2u);
      const uint64_t qd2 = ((uint64_t)((512 >> 4) | (2u << 30)) << 32) | (qlo0 + 2u);
      wgmma_fence();
      wgmma_m64n64k32_e4m3_ss(s_ovl, qdesc0, kd, 0);
      wgmma_m64n64k32_e4m3_ss(s_ovl, qd2, kd2, 1);
      wgmma_m64n64k32_e4m3_ss(s_ovl, qdesc0 + 512u, kd, 1);   // (QLO lo plane)
      wgmma_m64n64k32_e4m3_ss(s_ovl, qd2 + 512u, kd2, 1);
      wgmma_commit();
"""


def patch_rot(S):
    """ROT: the flat kernel issues tile i+1's QK^T (hi + lo pairs) at the end of tile i, right after tile i's PV, into
    the (dead by then) score registers: no wgmma wait at the tile's end; tile i+1's V transpose runs under PV(i) +
    QK(i+1), whose one wait<0> follows it. No extra registers for the scores (P / o stay live over the transpose)."""
    # the QK^T issue (wait for the stage, new-row patch, hi + lo wgmma pairs) as a lambda: before the loop for tile 0,
    # at each tile's end for the next one
    a = "#pragma unroll (UNR8)\n  for (int jj = 0; jj < nit; ++jj) {\n"
    b = ("  float s_ovl[32];   // (ROT, flat kernel) the scores of the tile whose QK^T is in flight\n"
         "  bool ovl_have = false;\n"
         "  auto ovl_qk = [&](int i) -> bool {   // (ROT) tile i's stage landed -> its QK^T in flight\n"
         "    if constexpr (WS) {\n"
         "      mbar_wait_full(&S.full[st_c], ph_c);\n"
         "      if (DYNC && i >= ntiles && *reinterpret_cast<volatile int*>(&S.tflag[st_c]) == 0) return false;\n"
         "      const bool patched = (!DYNC || i < ntiles) && ((uint32_t)i - r_lo > r_span) && (i >= i_new) &&\n"
         "                           ws_patch_ni<L, NT, false>(&S, i, st_c, start, seq, end, tid);\n"
         "      if (patched || i == 0) csync<WS, NT>();\n"
         + QK_ISSUE +
         "      return true;\n"
         "    }\n"
         "    return false;\n"
         "  };\n"
         "  if constexpr (WS) ovl_have = nit > 0 && ovl_qk(0);\n" + a)
    S = _rep(S, a, b)
    rx = re.compile(r"      if constexpr \(WS\) \{\n      // tile i landed \(\+ new rows\);.*?\n    \} else \{\n      cp_wait<STAGES - 2>\(\);",
                    re.S)
    assert len(rx.findall(S)) == 1
    S = rx.sub(lambda _: "      if constexpr (WS) {\n        if (!ovl_have) break;   // (ROT: tile i's QK^T was issued at the previous tile's end)\n"
                         "    } else {\n      cp_wait<STAGES - 2>();", S)
    a = "    float s[32];\n    wgmma_fence();\n"
    S = _rep(S, a, "    float s_in[32];\n    float (&s)[32] = WS ? s_ovl : s_in;\n    if constexpr (!WS) {\n    wgmma_fence();\n")
    a = """      wgmma_m64n64k32_e4m3(s, ql8[1][0], ql8[1][1], ql8[1][2], ql8[1][3], kd2, 1);
      }
     
    wgmma_commit();
"""
    S = _rep(S, a, a + "    }\n")
    S = _rep(S, "    if constexpr (FLAT) {\n      // the previous tile's PV (committed before this QK^T) is done",
             "    if constexpr (FLAT && !WS) {\n      // the previous tile's PV (committed before this QK^T) is done")
    # no wait at the tile's end; the next tile's QK^T goes in behind this tile's PV
    ia = S.index("    PROF_TILE;\n", S.index("float (&s)[32] = WS ? s_ovl : s_in;"))
    a = """           // done before the next tile (no wgmma in flight across the loop edge)
          wgmma_wait<0>();
"""
    ib = S.rindex(a, 0, ia)
    assert ia - ib < 600, ia - ib
    S = S[:ib] + "          if constexpr (!WS) {\n" + S[ib:ia] + "          }\n" + S[ia:]
    a = "      vdesc_c ^= vdesc_x;\n    \n  }\n  PROF_DUMP;\n"
    S = _rep(S, a, "      vdesc_c ^= vdesc_x;\n    \n    if constexpr (WS) ovl_have = jj + 1 < nit && ovl_qk(jj + 1);\n  }\n  PROF_DUMP;\n")
    return S


def patch_noend(S, nofence=False):
    """NOEND: the flat kernel drops the wgmma wait at each tile's end: tile i's PV runs on while tile i+1's stage wait,
    new-row patch and QK^T issue go ahead; the top-of-tile wait<1> (already there) then retires the PV before the
    transpose needs P's registers."""
    ia = S.index("    PROF_TILE;\n", S.index("wgmma_m64n64k32_e4m3_ss(s, qdesc0 + 512u, kd, 1);"))
    a = """           // done before the next tile (no wgmma in flight across the loop edge)
          wgmma_wait<0>();
"""
    ib = S.rindex(a, 0, ia)
    assert ia - ib < 600, ia - ib
    S = S[:ib] + "          if constexpr (!WS) {\n" + S[ib:ia] + "          }\n" + S[ia:]
    # ws_patch_ni is a real call (ABI register save around it): retire the PV first on that (rare) path, else ptxas
    # serializes every wgmma of the kernel (C7514)
    a = "patched = (!DYNC || i < ntiles) && ((uint32_t)i - r_lo > r_span) && (i >= i_new) && ws_patch_ni<L, NT, false>(&S, i, st_c, start, seq, end, tid);"
    S = _rep(S, a, "patched = (!DYNC || i < ntiles) && ((uint32_t)i - r_lo > r_span) && (i >= i_new);\n"
                   "        if (patched) {\n          wgmma_wait<0>();\n"
                   "          patched = ws_patch_ni<L, NT, false>(&S, i, st_c, start, seq, end, tid);\n        }")
    if nofence:   # (the top-of-tile wait<1>: no register fences on o / lacc after it)
        a = """      float* od0 = &o[0][0][0];
#pragma unroll
      for (int e = 0; e < 32; ++e) reg_fence(od0[e]);
#pragma unroll
      for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
"""
        S = _rep(S, a, "")
        a = """      wgmma_wait<1>();
#pragma unroll
      for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
"""
        S = _rep(S, a, "      wgmma_wait<1>();\n")
    return S


def patch_lato(S):
    """LATO (timing oracle, WRONG results): the lo QK^T pair leaves the QK -> softmax chain and is issued with the
    PV instead (same tensor work, same commit group as the PV). ~ss3 time => QLO's cost is throughput, not latency."""
    a = """        wgmma_m64n64k32_e4m3_ss(s, qdesc0 + 512u, kd, 1);
        wgmma_m64n64k32_e4m3_ss(s, qd2 + 512u, kd2, 1);
"""
    S = _rep(S, a, "")
    a = "          wgmma_m64n72k32_e4m3(odk, lacc, pa[0], vdesc_c);\n"
    S = _rep(S, a, "          wgmma_m64n64k32_e4m3_ss(s, qdesc0 + 512u, kd, 1);\n"
                   "          wgmma_m64n64k32_e4m3_ss(s, qd2 + 512u, kd2, 1);\n" + a)
    return S


SSZ = r"""// the same with scale-d = 0 and write-only accumulators (LNEXT: the masked scores of the tile before do not
// feed it, so ptxas sees no non-wgmma definition of its accumulators inside the pipeline)
__device__ __forceinline__ void wgmma_m64n64k32_e4m3_ssz(float* d, uint64_t da, uint64_t db) {
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
      "%32, %33, 0, 1, 1;\n}\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]),
        "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]),
        "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]), "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]),
        "=f"(d[24]), "=f"(d[25]), "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]), "=f"(d[30]), "=f"(d[31])
      : "l"(da), "l"(db));
}
"""


def patch_lnext(S, unr=2, full=False, nowait=False, test=False):
    """LNEXT (flat kernel, v16/v32 full): tile i+1's q_lo K^T pair is issued at the end of tile i, in the PV's commit
    group, into the score registers (dead after P's pack); tile i+1 then adds only the q_hi pair (scale-d 1) on its
    QK -> softmax chain. Tile i+1's stage wait + new-row patch move before tile i's PV issue (no wgmma in flight
    there, nor across the loop edge: the tile-end wait<0> stays). Same math (hi + lo, FP32 accumulate)."""
    S = _rep(S, "// K-major operand with 64-byte rows, 64B swizzle (8-row atoms of 512 bytes)\n",
             SSZ + "// K-major operand with 64-byte rows, 64B swizzle (8-row atoms of 512 bytes)\n")
    if unr != 2:
        S = _rep(S, "  constexpr int UNR8 = WS ? 2 : 1;", "  constexpr int UNR8 = WS ? %d : 1;" % unr)
    a = "#pragma unroll (UNR8)\n  for (int jj = 0; jj < nit; ++jj) {\n"
    S = _rep(S, a, "  float s_ovl[32];   // (LNEXT, flat kernel) scores: tile i+1's lo part lands during tile i's PV\n"
                   "  int ovl_st = 0;   // (LNEXT) next tile: 0 = full top path, 1 = its lo pair is in (hi pair only), 2 = pool sentinel\n" + a)
    a = "        mbar_wait_full(&S.full[st_c], ph_c);\n        if (DYNC && i >= ntiles"
    S = _rep(S, a, "        if (ovl_st == 2) break;   // (LNEXT: the pool's sentinel, seen at the previous tile's end)\n"
                   "        if (ovl_st == 0) mbar_wait_full(&S.full[st_c], ph_c);\n        if (ovl_st == 0 && DYNC && i >= ntiles")
    S = _rep(S, "        patched = (!DYNC || i < ntiles) && ((uint32_t)i - r_lo > r_span)",
             "        patched = ovl_st == 0 && (!DYNC || i < ntiles) && ((uint32_t)i - r_lo > r_span)")
    S = _rep(S, "    float s[32];\n    wgmma_fence();\n",
             "    float s_in[32];\n    float (&s)[32] = WS ? s_ovl : s_in;\n    wgmma_fence();\n")
    a = """       wgmma_m64n64k32_e4m3_ss(s, qdesc0, kd, 0);
      wgmma_m64n64k32_e4m3_ss(s, qd2, kd2, 1);
      if constexpr (true) {"""
    S = _rep(S, a, ("""      if (WS && ovl_st == 1) {   // (LNEXT) + q_hi K^T onto the lo part already in s
%s
      } else {
""" + a.replace("       wgmma_m64n64k32_e4m3_ss(s, qdesc0, kd, 0);\n",
                "       if constexpr (WS) wgmma_m64n64k32_e4m3_ssz(s, qdesc0, kd);\n"
                "       else wgmma_m64n64k32_e4m3_ss(s, qdesc0, kd, 0);\n")) % (
        "        // (FNEXT: the whole QK^T came in with the previous tile's PV)" if full else
        "        wgmma_m64n64k32_e4m3_ss(s, qdesc0, kd, 1);\n        wgmma_m64n64k32_e4m3_ss(s, qd2, kd2, 1);"))
    a = """        wgmma_m64n64k32_e4m3(s, ql8[1][0], ql8[1][1], ql8[1][2], ql8[1][3], kd2, 1);
      }
     
    wgmma_commit();
"""
    S = _rep(S, a, a.replace("    wgmma_commit();\n", "      }\n    wgmma_commit();\n"))
    a = """        if constexpr (LSYNC) {   // every warp's V^T for this tile written (and visible to the async proxy)
          fence_proxy_async();
          wgsync(wgi);
        }
        wgmma_fence();
"""
    b = """        uint32_t kl_n = 0;
        if constexpr (WS) {   // (LNEXT) the next tile's stage (+ new-row patch) before this tile's PV goes in
          int st_n = st_c + 1;
          uint32_t ph_n = ph_c;
          if (st_n == STAGES) { st_n = 0; ph_n ^= 1u; }
          kl_n = klo0 + (uint32_t)(st_n * (STAGE_BYTES >> 4));
          // tiles that need the new-row patch (a real call) keep the full top path: no call while P is live
          const bool needp = (!DYNC || i + 1 < ntiles) && ((uint32_t)(i + 1) - r_lo > r_span) && (i + 1 >= i_new);
          ovl_st = 0;
          if (jj + 1 < nit && !needp) {
            mbar_wait_full(&S.full[st_n], ph_n);
            ovl_st = (DYNC && i + 1 >= ntiles && *reinterpret_cast<volatile int*>(&S.tflag[st_n]) == 0) ? 2 : 1;
          }
        }
""" + a
    if test:   # take the early path only if the next stage has already landed (one thread tests, the pre-PV
               # warpgroup barrier broadcasts it: the wgmma issue must be warpgroup-uniform)
        S = _rep(S, "  int nxt[4][2];", "  int ovlf[2][2];     // (LNEXT) per warpgroup and tile parity: the next stage landed at its test\n  int nxt[4][2];")
        S = _rep(S, "__device__ __forceinline__ void mbar_wait_full(uint64_t* bar, int parity) {\n",
                 "__device__ __forceinline__ int mbar_test_full(uint64_t* bar, int parity) {\n"
                 "  uint32_t r;\n"
                 "  asm volatile(\"{\\n.reg .pred q;\\nmbarrier.test_wait.parity.shared::cta.b64 q, [%1], %2;\\nselp.u32 %0, 1, 0, q;\\n}\\n\"\n"
                 "               : \"=r\"(r) : \"r\"((uint32_t)__cvta_generic_to_shared(bar)), \"r\"(parity) : \"memory\");\n"
                 "  return (int)r;\n}\n"
                 "__device__ __forceinline__ void mbar_wait_full(uint64_t* bar, int parity) {\n")
        b = b.replace("""          if (jj + 1 < nit && !needp) {
            mbar_wait_full(&S.full[st_n], ph_n);
            ovl_st = (DYNC && i + 1 >= ntiles && *reinterpret_cast<volatile int*>(&S.tflag[st_n]) == 0) ? 2 : 1;
          }
        }
""", """          if ((tid & 127) == 0) S.ovlf[hk][jj & 1] = (jj + 1 < nit && !needp) ? mbar_test_full(&S.full[st_n], ph_n) : 0;
        }
""")
        b = b.replace("""        if constexpr (LSYNC) {   // every warp's V^T for this tile written (and visible to the async proxy)
          fence_proxy_async();
          wgsync(wgi);
        }
""", """        if constexpr (LSYNC) {   // every warp's V^T for this tile written (and visible to the async proxy)
          fence_proxy_async();
          wgsync(wgi);
        }
        if constexpr (WS) {
          if (*reinterpret_cast<volatile int*>(&S.ovlf[hk][jj & 1])) {
            int st_n = st_c + 1;
            uint32_t ph_n = ph_c;
            if (st_n == STAGES) { st_n = 0; ph_n ^= 1u; }
            mbar_wait_full(&S.full[st_n], ph_n);   // (landed: every thread's own acquire)
            ovl_st = (DYNC && i + 1 >= ntiles && *reinterpret_cast<volatile int*>(&S.tflag[st_n]) == 0) ? 2 : 1;
          }
        }
""")
    if nowait:   # (timing oracle, UNSAFE: the next stage is not waited for)
        b = b.replace("            mbar_wait_full(&S.full[st_n], ph_n);\n", "")
    S = _rep(S, a, b)
    a = """          wgmma_m64n72k32_e4m3(odk, lacc, pa[1], vdesc_c + 2);
         
        wgmma_commit();
"""
    S = _rep(S, a, a.replace("        wgmma_commit();\n", """        if (WS && ovl_st == 1) {   // (LNEXT) the next tile's q_lo K^T (lo plane: +512 in the q descriptor)
          const uint64_t kdn = ((uint64_t)((1024 >> 4) | (1u << 30)) << 32) | kl_n;
          const uint64_t kdn2 = ((uint64_t)((1024 >> 4) | (1u << 30)) << 32) | (kl_n + 2u);
          const uint64_t qdl2 = ((uint64_t)((512 >> 4) | (2u << 30)) << 32) | (qlo0 + 2u);
%s
        }
        wgmma_commit();
""" % (("          wgmma_m64n64k32_e4m3_ssz(s, qdesc0, kdn);\n"
         "          wgmma_m64n64k32_e4m3_ss(s, qdl2, kdn2, 1);\n" if full else
         "          wgmma_m64n64k32_e4m3_ssz(s, qdesc0 + 512u, kdn);\n") +
        "          wgmma_m64n64k32_e4m3_ss(s, %s + 512u, kdn%s, 1);" % (("qdesc0", "") if full else ("qdl2", "2")) +
        ("\n          wgmma_m64n64k32_e4m3_ss(s, qdl2 + 512u, kdn2, 1);" if full else ""))))
    return S
