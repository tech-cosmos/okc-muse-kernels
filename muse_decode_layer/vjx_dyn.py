"""n66: per-request dynamic tile pool for vjx's flat q_len-8 kernel (fixes the end-of-grid tail of n65).
Each long request's first Q = DYNF/1000 of its cached tiles (whole 64-key tiles, all interior: no mask, no new
rows) leave the static equal-share partition and form a pool; CTAs whose LAST segment is that request, plus the
CTA holding the request's end, drain it after their static tiles (atomic tile counter per (request, pair)), in the
same softmax state (no extra segment, no extra merge). Counters reset in flat_finalize after the grid barrier.
patch(src, dynf) -> patched CUDA source."""


SKIP = set()
_N = [0]


def _rep(s, a, b, n=1):
    assert s.count(a) == n, (s.count(a), a[:80])
    _N[0] += 1
    if _N[0] in SKIP:
        return s
    return s.replace(a, b)


def patch(S, dynf, dynch=4, mode=0, pf=1):
    _N[0] = 0
    S = _rep(S, "struct Params {\n",
             f"#ifndef DYNF\n#define DYNF {dynf}\n#endif\n#define DYNCH {dynch}\n#define DYNMODE {mode}\n#define DYNPF {pf}\n"
             "// pool keys of a request (q_len-8 flat kernel): whole tiles from key 0, all before the new rows\n"
             "__device__ __forceinline__ int dyn_q(int seq) {\n"
             "  const int q = ((seq >> 6) * DYNF / 1000) << 6;\n"
             "  return min(q, max(0, ((seq >> 6) - 1) << 6));\n}\n"
             "struct Params {\n")
    S = _rep(S, "  int pnseg, pchunk;\n", "  int pnseg, pchunk;\n  int dyn_t[2];\n  int tflag[8];\n  int dshare;\n")
    # pool only for requests whose static part is longer than one CTA share (+ a tile): then some CTA's range
    # ends inside the request, and every CTA whose last segment is the request drains its pool
    S = _rep(S, "  if (threadIdx.x < 32) {\n    int run = 0;\n    int incl_l = 0, n_l = 0, seq_l = 0;\n",
             "  if (threadIdx.x < 32) {\n    int run = 0;\n    int incl_l = 0, n_l = 0, seq_l = 0;\n"
             "    int dshare = 0;\n"
             "    if (L == 8 && DYNF > 0) {\n"
             "      int tot = 0;\n"
             "      for (int b = threadIdx.x; b < p.B; b += 32) {\n"
             "        const int sq = __ldg(p.seq_lens + b);\n"
             "        tot += sq + L - (p.window ? max(0, sq + 1 - p.window) : 0);\n"
             "      }\n"
             "#pragma unroll\n"
             "      for (int o = 16; o > 0; o >>= 1) tot += __shfl_xor_sync(0xffffffffu, tot, o);\n"
             "      const int G_ = gridDim.x;\n"
             "      dshare = (tot + (p.B - 1) * 600 + G_ - 1) / G_ + 64;   // (32-bit: a 64-bit division is a call)\n"
             "    }\n")
    # virtual (static) request lengths
    S = _rep(S, "        n = seq + L - lo;\n        seq_l = seq;\n",
             "        n = seq + L - lo;\n"
             "        if (L == 8 && DYNF > 0) {\n"
             "          const int dq = (p.window == 0 && n - dyn_q(seq) > dshare) ? dyn_q(seq) : 0;\n"
             "          if (b0 == 0 && threadIdx.x == 0) S.dshare = dshare;\n"
             "          n -= dq;\n"
             "        }\n        seq_l = seq;\n")
    a = ("      const int s0 = S.seg[k][1] + lo, s1 = S.seg[k][2] + lo;\n      \n      if (s0 < s1) {\n"
         "        segment<L, FLAT, R>(p, S, b, pair, s0 == lo, s0, s1, seq, n_keys, vc - fdiv_i(vc, p.nc) * p.nc, 0,\n")
    b = ("      const int dq_ = (L == 8 && DYNF > 0 && p.window == 0 && n_keys - dyn_q(seq) > S.dshare) ? dyn_q(seq) : 0;\n"
         "      const int s0 = S.seg[k][1] + lo + dq_, s1 = S.seg[k][2] + lo + dq_;\n"
         "      const int dyn_ = (DYNMODE != 1 && dq_ > 0 && p.coop && k == nseg - 1) ? 1 : 0;\n"
         "      if (s0 < s1) {\n"
         "        segment<L, FLAT, R>(p, S, b, pair, S.seg[k][1] == 0, s0, s1, seq, n_keys, vc - fdiv_i(vc, p.nc) * p.nc, dyn_,\n")
    S = _rep(S, a, b)
    S = _rep(S, "          mbar_init_n(&S.empty[st], NT);   // F2: one warpgroup per head takes each tile\n",
             "          mbar_init_n(&S.empty[st], NT);   // F2: one warpgroup per head takes each tile\n"
             "          if constexpr (FLAT) S.tflag[st] = 1;\n")
    # producer: drain the pool after the static tiles
    a = "      \n      \n      PROF_DUMP;\n"
    drain = r"""
      if constexpr (L == 8 && FLAT && DYNF > 0) if (chunk) {
        // pool tiles in tickets of DYNCH tiles (one atomic per ticket, the next ticket already in flight);
        // page indices one tile ahead inside a ticket
        const int nq = dyn_q(seq) >> 6;
        int* ctr = p.ws_cnt + 31000 + bp;
        int pend = 0;
        if (q == 0) { S.dyn_t[0] = atomicAdd(ctr, 1); pend = atomicAdd(ctr, 1); }
        asm volatile("bar.sync 9, %0;\n" ::"n"(NP) : "memory");
        int cur = *reinterpret_cast<volatile int*>(&S.dyn_t[0]);
        int x = ntiles;
        int pgd[PK];
        while (cur * DYNCH < nq) {
          const int t0 = cur * DYNCH, t1 = min(nq, t0 + DYNCH);
          for (int t = t0; t < t1; ++t, ++x) {
#pragma unroll
            for (int k = 0; k < PK; ++k) pgd[k] = __ldg(ptr + ((t * TILE + r0 + PR * k) >> p.lp));
            if (DYNPF && q < 2 && t + 1 < t1)   // the next tile's page indices toward L2 (no registers held)
              asm volatile("prefetch.global.L2 [%0];" ::"l"(ptr + (((t + 1) * TILE + 32 * q) >> p.lp)));
            const int st = (tb + x) % STAGES;
            if (tb + x >= STAGES) mbar_wait_full(&S.empty[st], (((tb + x) / STAGES) - 1) & 1);
            const uint32_t kb = smem_u32(S.k + st * STAGE_BYTES), vb = smem_u32(S.v + st * STAGE_BYTES);
#pragma unroll
            for (int k = 0; k < PK; ++k) {
              const int r = r0 + PR * k;
              const int pos = t * TILE + r;
              const uint32_t so = r * ROWB + ((ch ^ (r & 7)) << 4);
              if (p.off32) {
                const uint32_t off = ((((uint32_t)pgd[k] << p.lp) + (uint32_t)(pos & (p.P - 1))) << p.lrb) + (uint32_t)(pair * 128 + ch * 16);
                cp_async16(kb + so, p.k_cache + off, 16);
                cp_async16(vb + so, p.v_cache + off, 16);
              } else {
                const int64_t off = (((int64_t)pgd[k] << p.lp) + (pos & (p.P - 1))) * p.row_bytes + pair * 128 + ch * 16;
                cp_async16(kb + so, p.k_cache + off, 16);
                cp_async16(vb + so, p.v_cache + off, 16);
              }
            }
            cp_async_mbar_arrive(&S.full[st]);
          }
          asm volatile("bar.sync 9, %0;\n" ::"n"(NP) : "memory");
          if (q == 0) { S.dyn_t[0] = pend; pend = atomicAdd(ctr, 1); }
          asm volatile("bar.sync 9, %0;\n" ::"n"(NP) : "memory");
          cur = *reinterpret_cast<volatile int*>(&S.dyn_t[0]);
        }
        {   // sentinel stage: no copies, flag 0
          const int st = (tb + x) % STAGES;
          if (tb + x >= STAGES) mbar_wait_full(&S.empty[st], (((tb + x) / STAGES) - 1) & 1);
          if (q == 0) *reinterpret_cast<volatile int*>(&S.tflag[st]) = 0;
          mbar_arrive(&S.full[st]);
        }
      }
"""
    S = _rep(S, a, drain + a)
    # consumers: dynamic trip count, sentinel stops, pool tiles are interior
    S = _rep(S, "  const int nit = ntiles;\n",
             "  constexpr bool DYNC = L == 8 && FLAT && WS && DYNF > 0;\n"
             "  const int nit = (DYNC && chunk) ? 0x7fffffff : ntiles;\n")
    S = _rep(S, "        mbar_wait_full(&S.full[st_c], ph_c);\n        patched = ((uint32_t)i - r_lo > r_span)",
             "        mbar_wait_full(&S.full[st_c], ph_c);\n"
             "        if (DYNC && i >= ntiles && *reinterpret_cast<volatile int*>(&S.tflag[st_c]) == 0) break;\n"
             "        patched = (!DYNC || i < ntiles) && ((uint32_t)i - r_lo > r_span)")
    S = _rep(S, "    const bool common_t = WS && ((uint32_t)i - r_lo <= r_span);\n",
             "    const bool common_t = WS && (((uint32_t)i - r_lo <= r_span) || (DYNC && i >= ntiles));\n")
    S = _rep(S, "  csync<WS, NT>();\n  const int pairs = p.Hkv >> 1;\n  const int rows = p.B * pairs * L * 16;\n",
             "  csync<WS, NT>();\n  const int pairs = p.Hkv >> 1;\n  const int rows = p.B * pairs * L * 16;\n"
             "  if (L == 8 && DYNF > 0 && c == 0)\n"
             "    for (int i = tid; i < p.B * pairs; i += NT) p.ws_cnt[31000 + i] = 0;\n")
    return S
