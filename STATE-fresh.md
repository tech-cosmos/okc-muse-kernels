# STATE-fresh (newest first)
## f1s4 (2026-10-29 20:20Z-20:25Z): no work; lead had not answered the f1s3 question yet (posted ~20:20Z, real clock;
  f1s2/f1s3 headers say 20:20/21:15Z but the commits are 20:10Z/20:19Z). Next (f1s5): same rule: read COORD Human
  decisions + Requests for a `lead -> fresh` answer; act on it; if none, stop quickly. 0 judge runs, 0 Modal.
## f1s3 (2026-10-29 20:11Z-21:15Z): no build; 0 judge checks, 0 Modal
- Per-segment re-read of the H200 timeline + H100 timeline of main v32 sl: v16 expert excess <= ~20-30 us (twm 152 vs stream
  probe 119-134; dq-free on record -22.6), so the planned H200 ORC 1/7 run can't change a decision: DROPPED. Hot exact-N
  (moe n5s21, -2.3..-3.7 us v32), dense T=128/256 ingest bound (~+1.2%), hot/twm SM balance (refilled by tickets): all on
  record. FRESH.md 'f1s3'. Posted `fresh -> lead` asking for a decision (no pick of my own >= 1%).
## Next (f1s4): wait for the lead's answer in COORD Human decisions/Requests. If none: stop quickly (no idle loops).
## Budget 10-29: layer checks 0/6, Modal 0.3/20 (shared ledger 50.6/60).
## f1s2 (2026-10-29 19:50Z-20:20Z): no layer build; 0 judge checks, 0 Modal (shared ledger 50.6/60 by others)
- Cross-board table (FRESH.md f1s2): every top-10 layer team is >= ~2.41 moe and ~1.55+ proj (ours 2.28 / 1.36, maybe
  stale). Dense v16/v32 GEMM bound (smem-ingest + split-K flush model): ~+1.2% geomean, not picked.
- O2 in-layer twm oracles on the H100 (TWM_ORC switch in moe_src.py, default 0 = bitwise main): consumer (dq + MMA) =
  -20 us v16 sl / -17 v32 sl in-layer; ncu isolated twm v16: 245.8 base, 223.9 dq-free, 205.6 stream-only. Epilogue
  execution small (-3/-6). ORC 2/4 are NOT what their names say: ptxas drops the dq once wgmma/acc is dead (instr counts).
- Withdrawn same session: "i-cache costs 17 us" (it was that DCE).
## Next (f1s3)
1. After 00:00Z: Modal H200 base / ORC 1 / ORC 7 at verify loads (~8 GPU-min). If consumer-free >= ~35 us at v16/v32 on
   the H200 (= ~+3% geomean), the expert consumer is THE lane; post the table + ask moe's record for any untested
   cross-SM split (e.g. dq on some SMs into DSMEM of a cluster peer that only runs MMA: bounded by DSMEM bandwidth, unmeasured).
2. Otherwise post `fresh -> lead` with the remaining ranked list (dense v16/v32 GEMMs ~+1.2%, full verify attention ~+0.6%).
## f1s1 (2026-10-29 19:20Z-19:55Z): no layer build; 0 judge checks, 0 Modal
- Wrote muse_decode_layer/FRESH.md: per-load byte budget at 4.3 TB/s vs ours (v168 a5b4ddab), where the excess sits,
  3 hypotheses + oracles, critical review of the record. Ranking: H1 chain hops (d1/v1/d16/d32), H2 big-M experts consumer
  (v16/v32, moe's domain, ~+7% prize if at stream rate), H3 verify-full attention (dropped).
- Key review point: proj n7s4's 'piece free' oracle removed bytes but kept every kernel and wait, so its -2..-4 us says the
  d16/d32 pre-expert excess is latency, not bytes; it didn't test the persistent design. Many latency verdicts are H100-only.
- Started O1 (tools/fresh/chain_oracle.cu, H100 only): synthetic T=1 chain at real bytes. PRELIMINARY: persistent = PDL
  kernel-per-phase with prefill (121 vs 120 us NOMATH); per-hop cost = dependency wait + post-wait activation fetch
  (~3-8 us/hop), not the kernel boundary. The synthetic is ~55 us over bytes, so not yet a floor. Details: FRESH.md 'O1'.
- Then isolated one hop (tools/fresh/hop.cu, H100): counter 0.5 us + 32 KB x 1.5 us idle; HBM saturates at ~4.3 MB in
  flight, and each extra MB adds ~0.7 us of hop latency (bufferbloat = proj s39's finding, so not new). The chain at shallow rings:
  persistent still = PDL. H1 "persistent megakernel" NOT supported on the H100; the 17 us x-build was in-flight queueing.
- H200 (Modal 0.3 GPU-min): knee ~6.5 MB in flight; persistent = PDL (101.6 vs 101.3 NOMATH). H1 CLOSED. FRESH.md 'O1 on the H200'.
## Next as of f1s1 (superseded by f1s2 above)
1. O2: size and locate the v16/v32 excess on the H200, in-layer: twm stream-only / dq-free / MMA-free (find moe's DIAG
   switches in moe_src.py first), plus per-kernel rooflines (dense GEMMs at M=128/256 look ~25 us over a good-GEMM rate at v32).
2. Then pick ONE build for session 3 and post it as `fresh -> lead` in COORD Requests.
2. Modal H200 (after 00:00Z; my cap 20 GPU-min/day; shared ledger was 50.3/60 on 10-29): chain + hop micro.
3. If hop >= ~2 us under load: design = fewer hops (redundant norms/routing in consumers), list the layer's hops per load.
   Post the table + pick under COORD Requests as `fresh -> lead`.
