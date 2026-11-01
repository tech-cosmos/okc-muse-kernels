"""n66 v2 'donor' pool for vjx's flat q_len-8 kernel: the static equal-share partition is unchanged; every CTA
whose range ENDS inside a request (below its new rows, with >= X + 64 keys of that request) donates the last
X = XT*64 keys of its range to the request's pool, and every CTA whose last segment is that request drains the
pool after its static tiles (atomic ticket counter per (request, pair), same softmax state, no extra segment).
Pool per drainer is ~X for every request, so groups of 7 and 9 CTAs end together (n66 v1's residual spread).
patch(src, dynf=per-mille of a share donated, dynch=tiles per ticket) -> CUDA source."""


def _rep(s, a, b, n=1):
    assert s.count(a) == n, (s.count(a), a[:80])
    return s.replace(a, b)


def patch(S, dynf=100, dynch=2, pn=1):
    S = _rep(S, "struct Params {\n", f"#define DYNF {dynf}\n#define DYNCH {dynch}\n#define DYNPN {pn}\nstruct Params {{\n")
    S = _rep(S, "  int pnseg, pchunk;\n",
             "  int pnseg, pchunk;\n  int dyn_t[2];\n  int tflag[8];\n  int dce[64];\n  int ndon, dxt, mydon;\n")
    # donor list of this CTA's last request (warp 0, after the segment list)
    a = "  if (!fpart) __syncthreads();   // (warp 0 wrote the list before the barrier above)\n  const int nseg = S.nseg;\n"
    b = r"""  if (!fpart) __syncthreads();   // (warp 0 wrote the list before the barrier above)
  if constexpr (L == 8 && FLAT && DYNF > 0) {
    if (threadIdx.x < 32) {
      const int lane = threadIdx.x;
      const int bl = S.seg[S.nseg - 1][0];
      const int seqb = __ldg(p.seq_lens + bl);
      const int64_t vs = (int64_t)S.cum[bl] + (int64_t)bl * SEGK, len = S.cum[bl + 1] - S.cum[bl];
      const int share = (int)(T / G);
      const int xt = max(1, (int)((int64_t)share * DYNF / 64000));
      const int X = xt * 64;
      // the group (CTAs ending in the request) spans < 31 CTAs: it fits the window c-32 .. c+31 of every member
      const bool ok = p.coop && p.window == 0 && len * G < 30 * T;
      int rank = 0;
      int mine = 0;
#pragma unroll 1
      for (int h = 0; h < 2; ++h) {
        const int cc = c - 32 + 32 * h + lane;
        bool don = false;
        int rce = 0;
        if (ok && cc >= 0 && cc < G) {
          const int64_t vce = (int64_t)(cc + 1) * T / G, vcs = (int64_t)cc * T / G;
          rce = (int)(vce - vs);
          const int rcs = (int)max(vcs - vs, (int64_t)0);
          don = rce > 0 && rce <= seqb && rce - rcs >= X + 64;
        }
        const unsigned m = __ballot_sync(0xffffffffu, don);
        if (don) S.dce[rank + __popc(m & ((1u << lane) - 1u))] = rce;
        if (don && cc == c) mine = 1;
        rank += __popc(m);
      }
      mine = __any_sync(0xffffffffu, mine);
      if (lane == 0) { S.ndon = rank; S.dxt = xt; S.mydon = mine; }
    }
    __syncthreads();
  }
  const int nseg = S.nseg;
"""
    S = _rep(S, a, b)
    a = ("      const int s0 = S.seg[k][1] + lo, s1 = S.seg[k][2] + lo;\n      \n      if (s0 < s1) {\n"
         "        segment<L, FLAT, R>(p, S, b, pair, s0 == lo, s0, s1, seq, n_keys, vc - fdiv_i(vc, p.nc) * p.nc, 0,\n")
    b = ("      const bool lastk = k == nseg - 1;\n"
         "      const int s0 = S.seg[k][1] + lo;\n"
         "      const int s1 = S.seg[k][2] + lo - ((L == 8 && DYNF > 0 && lastk && S.mydon) ? S.dxt * 64 : 0);\n"
         "      const int dyn_ = (L == 8 && DYNF > 0 && lastk && S.ndon > 0) ? 1 : 0;\n"
         "      if (s0 < s1) {\n"
         "        segment<L, FLAT, R>(p, S, b, pair, s0 == lo, s0, s1, seq, n_keys, vc - fdiv_i(vc, p.nc) * p.nc, dyn_,\n")
    S = _rep(S, a, b)
    S = _rep(S, "          mbar_init_n(&S.empty[st], NT);   // F2: one warpgroup per head takes each tile\n",
             "          mbar_init_n(&S.empty[st], NT);   // F2: one warpgroup per head takes each tile\n"
             "          if constexpr (FLAT) S.tflag[st] = 1;\n")
    a = "      \n      \n      PROF_DUMP;\n"
    drain = r"""
      if constexpr (L == 8 && FLAT && DYNF > 0) if (chunk) {
        // pool tile t = donor t / XT's piece [ce - X, ce), tile t % XT; tickets of DYNCH tiles, next one in flight
        const int xt = S.dxt;
        const int nq = S.ndon * xt;
        int* ctr = p.ws_cnt + 31000 + bp;
        int pend = 0;
        if (q == 0) { S.dyn_t[0] = atomicAdd(ctr, 1); pend = atomicAdd(ctr, 1); }
        asm volatile("bar.sync 9, %0;\n" ::"n"(NP) : "memory");
        int cur = *reinterpret_cast<volatile int*>(&S.dyn_t[0]);
        int x = ntiles;
        int pgd[PK], pgn[PK];
        auto tpos = [&](int t) { const int j = t / xt; return S.dce[j] - (xt - (t - j * xt)) * TILE; };
        while (cur * DYNCH < nq) {
          const int t0 = cur * DYNCH, t1 = min(nq, t0 + DYNCH);
          {
            const int b0_ = tpos(t0);
#pragma unroll
            for (int k = 0; k < PK; ++k) pgd[k] = __ldg(ptr + ((b0_ + r0 + PR * k) >> p.lp));
          }
          for (int t = t0; t < t1; ++t, ++x) {
            const int base = tpos(t);
            if (DYNPN) {   // the next tile's page indices in flight while this tile issues
              const int bn = tpos(min(t + 1, t1 - 1));
#pragma unroll
              for (int k = 0; k < PK; ++k) pgn[k] = __ldg(ptr + ((bn + r0 + PR * k) >> p.lp));
            }
            const int st = (tb + x) % STAGES;
            if (tb + x >= STAGES) mbar_wait_full(&S.empty[st], (((tb + x) / STAGES) - 1) & 1);
            const uint32_t kb = smem_u32(S.k + st * STAGE_BYTES), vb = smem_u32(S.v + st * STAGE_BYTES);
#pragma unroll
            for (int k = 0; k < PK; ++k) {
              const int r = r0 + PR * k;
              const int pos = base + r;
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
            if (DYNPN) {
#pragma unroll
              for (int k = 0; k < PK; ++k) pgd[k] = pgn[k];
            } else if (t + 1 < t1) {
              const int bn = tpos(t + 1);
#pragma unroll
              for (int k = 0; k < PK; ++k) pgd[k] = __ldg(ptr + ((bn + r0 + PR * k) >> p.lp));
            }
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
