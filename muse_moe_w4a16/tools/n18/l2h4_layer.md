# n18: switch the layer's tk2f L2 hints from mode 1 to mode 4 (r-layer moe_src.py, 3 lines, bitwise-identical output)

The layer runs `#define TK2F_L2H 1` (evict_last only for the non-final 8-token groups' rows). moe's main has been on mode 4
since s21 (evict_last for EVERY row of an expert with > 8 tokens, evict_first for the others). The other group's CTA
reads the same row at about the same time (count split), so under mode 1 the first reader often marks the line evict_first.

Measured:
- H200 moe-only, layer routings (check 1b1d0fbd, 5 seeds, cold L2): mode 1 vs 4 = d1 +0.5, d16 +1.6, d32 +1.1, v1 +0.5 us.
- H100 IN-LAYER (your submission.py @77c0f17, tools/ab_flag.py, run-time mode switch, A B C A, 3 seeds x 3 rounds; us):
    d16 sliding s7/s11/s13: 156.7/151.2/157.2 (mode 1) -> 154.5/146.7/154.6 (mode 4)
    d16 full: 270.6/297.4/269.1 -> 268.2/295.4/269.2;  d32 sliding s11 210.9 -> 209.0, full s11 422.5 -> 419.3; d1 tie.
- Per-phase modes (gemm13 3 / gemm2 4 etc.) all lose to 4/4 on the H200: don't bother.

Patch (moe_src.py):
  line 91:   #define TK2F_L2H 1   ->   #define TK2F_L2H 4
  tk2f_produce:
-    const bool nfin = R + RPE < s_rs[a + 1];  // a later group of this expert re-reads this row
-    const int hk = nfin ? (TK2F_L2H == 3 ? 0 : 1) : (TK2F_L2H == 2 ? 0 : 2);  // 0 none, 1 evict_last, 2 evict_first
+    // mode 4: every group of a multi-group expert (the other group's CTA may read the same row first)
+    const bool nfin = TK2F_L2H == 4 ? s_rs[a + 1] - s_rs[a] > RPE : R + RPE < s_rs[a + 1];
+    const int hk = nfin ? (TK2F_L2H == 3 ? 0 : 1) : (TK2F_L2H == 2 ? 0 : 2);  // 0 none, 1 evict_last, 2 evict_first
(If submission.py embeds a copy of the source, the same 2 lines + the #define appear there too.)
Only the cache policy changes, so the outputs are bitwise-identical.
