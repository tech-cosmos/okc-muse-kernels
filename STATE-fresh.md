# STATE-fresh (newest first)
## f1s9 (2026-10-29 22:06Z-22:17Z): HOT_XN H200 A/B DONE via a judge check (1 of today's 6, would expire at 00:00Z), 0 Modal
- Made HOT_XN a runtime T=256 TUNE option: hot grid + HOT_XOFF (1e8) = moe_hot at N 128 for all items (moe_src hot_go
  `xn`, layer.py `TUNE[256] += ...`). Bitwise on/off (tools/fresh/xoff_eq.py, 4 seeds x sl/full), test 13/13, opt_test T=256 16/16.
- Check c5419971 (811355c) 2.1258, fixed-SGLang 2.1127 (slow draw on shapes HOT_XN doesn't touch: d1 sl 70.8 vs ~67);
  not submitted. H200 tuner (synthetic routing, one ~95-token hot expert): XN on vs off v32 sl 324.3 vs 325.8, full 516.9
  vs 519.6 us (same-log default-vs-confirm noise ~2 us). Weighted ~-1.8 us = ~0.1-0.2%: below the lead's -2 us gate on
  sliding -> no Handoff; offered as a bundle item (merge HOT_XN 16, drop the TUNE option). Modal A/B CANCELLED.
- Posted `fresh -> lead`: no lane left with a measured bound >= 1%; pause fresh or name one.
## Next (f1s10): read COORD Human decisions for a lead answer. If none, stop at once (no idle loops). Profiles of
   c5419971 are in muse_decode_layer/profiles/. Budget 10-29: layer checks 1/6, Modal 0.3/20.
## f1s8 (2026-10-29 21:50Z-21:58Z): no build, 0 judge, 0 Modal. H200 A/B can't run before 00:00Z: tools/h200.py
  reserves 3 + 4 x variants-runs GPU-min up front (rounds 2 = 19 min, rounds 1 = 11) and the shared 10-29 ledger is at
  50.6/60. Staged /tmp/fx_base.py (HOT_XN 0) + /tmp/fx_x16.py (diff = that one line). Re-checked the d1 qkv/tk2f fixed
  costs against the record (proj s17 bare-stream H200 check 29a2b16b; moe n7s22/n7s33 prologue): nothing new.
## Next (f1s9, at/after 00:00Z): run from the REPO ROOT (~/r-fresh), not muse_decode_layer/:
   python tools/h200.py muse_decode_layer base=/tmp/fx_base.py cand=/tmp/fx_x16.py --cmd 'python bench.py bench submission.py "verify 32"' --rounds 2
   (re-create the /tmp files with the f1s6 commands below if /tmp was wiped). x16 <= -2 us v32 sl+full -> Handoff; else HOT_XN 0, drop.
## f1s7 (2026-10-29 21:33Z-21:45Z): no build, 0 judge, 0 Modal (H200 A/B waits for the 00:00Z ledger reset, per lead 20:50Z)
- Re-read the v16/v32 dense chain against the record (H200 timeline: qkv m128 15.7 us = 2.4 TB/s, pooler 27.6, unpool v32
  26.9 = compute-ish at M=256). Own-GEMM forms (g22/g23 split-K+DSMEM, g30 W-in-registers, x multicast) and L2 prefetch of
  o/pooler W during attention (NOTES-layer 54-57, 796, 1239, 2206 O_PF) are all measured; no flaw I can name. Nothing new.
- Verified: branch fresh pushed (bba6a19), the HOT_XN 0 sed in 'Next' matches exactly one line of submission.py.
## Next (f1s8, at/after 00:00Z): the f1s6 'Next' H200 A/B below, unchanged. Then Handoff or drop HOT_XN.
## f1s6 (2026-10-29 20:52Z-21:30Z): lead 20:50Z = BUILD (1) dense T=128/256 GEMMs via CUTLASS, (2) hot exact-N. 0 judge, 0 Modal
- (1) CUTLASS sm90 gate FAILS on the H100: best of 8 builds x 3 modes loses to the LT_CFG cuBLASLt pick on all 8 shapes
  (qkv m128 25.1 vs 17.8, m256 26.8 vs 19.7; o +1.1/+1.4; pool m128 32.3 vs 27.3, m256 32.3 vs 30.1; unp +1.3/+1.6 us).
  Not shipped. Tools tools/fresh/cutlass (cg.cu, build.sh, cgb.py; CUTLASS 4.4.2 headers at /tmp/cutlass). FRESH.md f1s6.
- (2) hot exact-N BUILT: moe_src.py HOT_XN 16 (default on; 0 = old). Bitwise vs 0 at verify 32; bench test 13/13. H100 A/B neutral (DRAM-bound proxy). Commit on branch fresh.
## Next (f1s7, after 00:00Z): H200 A/B of HOT_XN 0 vs 16 at verify 32 (~6-8 GPU-min):
   cp submission.py /tmp/fx_x16.py; sed 's/#define HOT_XN 16 /#define HOT_XN 0 /' submission.py > /tmp/fx_base.py;
   python tools/h200.py muse_decode_layer base=/tmp/fx_base.py cand=/tmp/fx_x16.py --cmd 'python bench.py bench submission.py "verify 32"' --rounds 2
   If x16 <= -2 us on v32 sl+full: post Handoff (lead merges, bundles into a check). If neutral: set HOT_XN 0 and drop.
   Dense GEMM lane: closed for CUTLASS; nothing else queued from it.
## f1s5 (2026-10-29 20:36Z-20:40Z): no work; still no `lead -> fresh` answer in COORD (Human decisions newest = 19:05Z).
  Next (f1s6): same rule. If still unanswered by ~22:00Z, default pick = build hot exact-N + one dense T=128/256 GEMM
  attempt as a flagged bundle (bitwise, behind switches), H100 A/B first, no judge check until it measures >= ~1%.
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
