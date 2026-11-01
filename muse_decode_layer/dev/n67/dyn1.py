"""n67: n66's per-request dynamic tile pool (dev/n66/dyn.py) extended to vjx's flat q_len-1 kernel (d16/d32 full).
q_len 1 flat is not cooperative, so: drainers add their pool keys to the per-request merge counter (mine), and the
request's last-arriving CTA (the merger: every drainer has drawn its last ticket before it arrives) resets the
ticket counter. patch(src, dynf, dynch, l8) -> source; l8=0 keeps q_len 8 unpooled (q_len-1-only build)."""
import sys
sys.path.insert(0, "dev/n66")
import dyn


def _r(s, a, b, n=1):
    assert s.count(a) == n, (s.count(a), a[:90])
    return s.replace(a, b)


def patch(S, dynf, dynch=2, l8=1):
    S = dyn.patch(S, dynf, dynch)
    LC = "(L == 8 ? DYNL8 : 1)"
    S = _r(S, "#define DYNMODE", f"#define DYNL8 {l8}\n#define DYNMODE")
    S = _r(S, "    if (L == 8 && DYNF > 0) {\n      int tot", f"    if ({LC} && DYNF > 0) {{\n      int tot")
    S = _r(S, "(p.B - 1) * 600 + G_", "(p.B - 1) * ((L == 8) ? 600 : 400) + G_")
    S = _r(S, "        if (L == 8 && DYNF > 0) {\n          const int dq", f"        if ({LC} && DYNF > 0) {{\n          const int dq")
    S = _r(S, "const int dq_ = (L == 8 && DYNF > 0 &&", f"const int dq_ = ({LC} && DYNF > 0 &&")
    S = _r(S, "p.coop && k == nseg - 1", "(p.coop || L == 1) && k == nseg - 1")
    S = _r(S, "if constexpr (L == 8 && FLAT && DYNF > 0) if (chunk) {", "if constexpr (FLAT && DYNF > 0) if (chunk) {")
    S = _r(S, "        {   // sentinel stage: no copies, flag 0\n",
           "        if (q == 0) S.dyn_t[1] = x - ntiles;   // pool tiles taken (merge counter)\n"
           "        {   // sentinel stage: no copies, flag 0\n")
    # tflag for the first segment (L == 1 inits its barriers at kernel start)
    S = _r(S, "        mbar_init_n(&S.empty[st], Cfg<L, FLAT>::NT);\n      }\n      mbar_fence_init();\n",
           "        mbar_init_n(&S.empty[st], Cfg<L, FLAT>::NT);\n        S.tflag[st] = 1;\n      }\n      mbar_fence_init();\n")
    # q_len 1 consumer loop
    S = _r(S, "  for (int i = 0; i < ntiles; ++i) {\n    if constexpr (WS) {\n      if (ws_acquire(i, (tb + i) % STAGES)) csync<WS, NT>();\n",
           "  constexpr bool DYNC1 = FLAT && WS && DYNF > 0;\n"
           "  const int nit1 = (DYNC1 && chunk) ? 0x7fffffff : ntiles;\n"
           "  for (int i = 0; i < nit1; ++i) {\n    if constexpr (WS) {\n      if (ws_acquire(i, (tb + i) % STAGES)) csync<WS, NT>();\n"
           "      if (DYNC1 && i >= ntiles && *reinterpret_cast<volatile int*>(&S.tflag[(tb + i) % STAGES]) == 0) break;\n")
    S = _r(S, "    const bool interior = (t0 >= lo_all) && (t0 + TILE - 1 <= hi_all);\n",
           "    const bool interior = ((t0 >= lo_all) && (t0 + TILE - 1 <= hi_all)) || (DYNC1 && i >= ntiles);\n")
    # merge counter: pool keys count; merger resets the ticket counter
    S = _r(S, "  if (tid == 0) {\n    const int mine = end - start;\n",
           "  if (tid == 0) {\n    const int mine = end - start + ((L == 1 && FLAT && DYNF > 0 && chunk) ? S.dyn_t[1] * TILE : 0);\n")
    S = _r(S, "  if (tid == 0) p.ws_cnt[bp] = 0;\n  if (p.dbg",
           "  if (tid == 0) p.ws_cnt[bp] = 0;\n  if (L == 1 && FLAT && DYNF > 0 && tid == 0) p.ws_cnt[31000 + bp] = 0;\n  if (p.dbg")
    return S
