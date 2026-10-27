# Porting TK2F_SQ (single-copy tk2f stages) to r-layer's moe_src.py  (r-moe 01326b4; H200 d32 -1.5, d16 -0.55, d1/v1 0 us)
1. prepare(): after w13s2/w2s2 exist:
     st["w13sq"] = _sq_pack(st["w13q"], st["w13s2"]); st["w2sq"] = _sq_pack(st["w2q"], st["w2s2"])
   with _sq_pack from r-moe submission.py (5 lines, torch reshape + cat; lossless).
2. tk2f_produce: add template `bool SQ = false`; per row
     const char* qsrc = reinterpret_cast<const char*>(Wq) + ((size_t)e * RPE + lr) * NQ * SB;
   and per stage, instead of the 3 copies: `bulk_g2s[_h](sbase, qsrc + q * SB, SB, &full[st][, pol]);`
3. moe_tk2f producer call site: `constexpr bool SQ = (PF & 512) != 0;` and pass SQ to both tk2f_produce calls.
   Dispatch: e.g. cfg + 10000000 -> PF | 512 for the configs you use (PF 4 / 36 / 52 ...): one extra instantiation each.
4. The 3 layer.py moe_tk2f_launch calls: pass st["w13sq"], st["w2sq"] for w13_/w2_ when SQ (s13_/s2_ unused then).
Bitwise identical to off (consumer and smem image unchanged). Keep the race-fix arrive and the PDL wait as they are.
