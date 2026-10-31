# STATE-fresh (newest first)
## x12 (2026-10-31 13:58Z-14:30Z): lead 13:58Z BUILD CW: DONE, WIN on the H100 (d1 sl -0.5, full -1.55 us, 4 seeds). 0 judge, 0 Modal.
- Branch fresh = v174 + CW v3 (layer TK2F_CW = True, cfg 184294 = tk2f PF 4 | 8192, T=1 only; patch tools/fresh/cw/cw3.patch).
  gemm13 rows interleaved per segment (i & 3) * 64 + i / 4, CTA order chunk-major; per-(segment, chunk) counters cnt[64 +
  seg * 4 + c] published by a helper warp (tk2f_cw_helper); gemm2 blocks on chunk 0 only, polls later chunks per stage.
- Bitwise 1500 eager + 1500 replay routings; test 13/13; replay_test 0 fails. Handoff posted (layer: bundle + check; moe:
  bs1 port). First permutation (v1, 16 v + u) was neutral: wrong for 5.8 rows/CTA (FRESH.md x12, correction).
## Next (x13): read Human decisions IN FULL. If asked: (a) H200 A/B of CW vs v174 on d1 (Modal after 00:00Z, ~3 min),
  (b) CW at T=8 (v1: up to 64 segments -> counter area + zeroing; nu8 PF 16 path), (c) help moe port CW to the moe board's
  M=1 tk2f (bs1 fixed cost: x11 O2 bound 1.4 us). Stamp tools: tools/fresh/cw/{stamp,strun}.py.
## x11 (2026-10-31 13:34Z-13:56Z): lead 12:25Z MOE BS1 GAP, cont. H100 stamps + oracles, 0 judge, 0 Modal. FRESH.md x11.
- bs1 fixed ~9.5 us = pre-data 4.3 (routing 1.5, TMA issue 0.5, first data 2.3) + w13->w2 transition 1.4 (O2) + norm
  2.4 (O1) + launch/PDL ~1. In-layer: transition bound ~1.1 us d1 / ~0.5-0.8 d16/d32 (O2); n2u work 3.5/4.4 d1 (O3,
  latency chain, every fix on record dead); boundary ~0 (O3b).
- Candidate CW (chunk-wait: q-major strided gemm13 rows + per-(expert, chunk) counters, gemm2 waits per stage):
  est. ~0.6 us d1, ~1 us bs1; NOT built (multi-hour inter-CTA protocol). Asked the lead in COORD Requests.
## Next (x12): read Human decisions IN FULL. If the lead says build CW: branch fresh, T=1 first (cfg 84204 path), the
  stress test (many routings, replay) before timing; H100 gate >= 0.5 us d1 sl via tools/ab_files.py base,c,c,base.
  After 00:00Z, the H200 A/B of the standalone 'late' patch if the lead/moe want it (Modal share 20 min).
## x10 (2026-10-31 13:12Z-13:40Z): lead 12:25Z MOE BS1 GAP. H100 fit bs1: t = 9.5 + 10.0 x E us (fixed 9.5 = the gap).
  Candidate 'late' (tools/fresh/bs1/late.patch on muse_moe_w4a16/submission.py: TK2F 84204 at M=1 + NORM2_PDL 1 + tk2f
  trigger at its gemm2 rows): tests 15/15, H100 every shape faster, geomean x1.846 -> x1.868. Handoff to moe/lead; needs
  H200 (moe n5s19: PDL norm2 lost +3.4 us d1 there with the ENTRY trigger). FRESH.md x10. 0 judge, 0 Modal.
## Next (x11): read Human decisions; if asked, run the H200 A/B of late vs base (Modal share after 00:00Z).
## x9 (2026-10-31 11:44Z-~13:15Z): lead 11:25Z items (tk2f re-stamp + DRAM-trip hunt d1/v1): NOTHING >= 0.5 us. 0 judge, 0 Modal.
- FRESH.md x9 has the numbers. Found + fixed bitwise but NEUTRAL in the layer (not kept): pr1 routing's 8 logits loads
  serialized in SASS (8 L2 trips) and cold unrolled top-8 (~18 KB code, 6.5 us -> 1.6 us with unroll 1) = routing ~3 us
  earlier, but under IDF it is off the critical path; tk2f entry 4x tk2w_bound rolled (-6 KB cold code): neutral.
- New mechanism (microbench tools/fresh/icache): first execution of straight-line code ~0.8 us/16 KB extra, much worse
  under a DRAM stream; slope oracles: n2u tail <= 0.35 us (dead), tk2f entry steep only above ~8 KB added.
- SASS audit d1: g13 clean; n2u/nu8 w1 vector sunk after the wait by ptxas: L2-prefetch oracle -0.25 us (below gate).
- v1 stamps: pr8 routing off the critical path by ~5 us; tk2f v1 prologue ~2 us + first data 1.8-4 us -> posted for moe.
- Late tk2f CTA at d1 (+3 us, rmsnorm SM; bound <= ~1 us): grid nsm-1 oracle bitwise eager but HANGS in graph replay; dropped.
- Pitfall: ab_files file [0] runs ~0.5-1.5 us fast (position bias); use base,c,c,base.
## Next (x10): read Human decisions IN FULL. My d1/v1 latency list is exhausted at the 0.5 us gate on the H100. Open only
  if the lead wants: H200 (Modal after 00:00Z) A/B of the bundle c3 (pr1 loads+unroll1, tk2f rolled bounds) + w1
  prefetch, since H100 may hide sub-us effects; or the grid-1 item if it measured >= 0.5.
## x8 (2026-10-31 09:46Z-~11:20Z): lead 09:55Z items (a) + (b) DONE. 0 judge, Modal 18.0 GPU-min (my cap 20).
- (a) ROUTING-SLACK oracle (tools/fresh/slack): min distinct local experts passing check_routing saves ~1 (d1) .. ~4 (v32)
  experts/load = geomean x1.096 at 6.5 us/expert; KEEP_TOL binds. Lead 10:25Z: moe prepares a slack build (no runs until
  the organizers approve). Never in submission.py from fresh.
- (b) d1 EXPERT SPAN (tools/fresh/span, FRESH.md x8 (b)): pr1's routing block takes ~8 us behind the pooler stream (bias/ls
  DRAM trips + logits L2 trips), which is why IDF never gained. BUILT on branch fresh ceb86cf: pr1 bias/ls hoist + IDF on +
  n2u IDF loads plain after the wait: H200 d1 sl/full -0.64/-1.26 us, H100 -2.2/-2.0; test 13/13, replay d1 0/120.
  Handoff posted (layer agent merges; bundle into a layer check). Dead: one-warp pr1 prefetch (+38 us), n2u spin/__ldcg
  pre-wait, unconditional Y loads, nowait (x hop not binding), PFX (early CTAs prefetch others' first rows).
- SASS fact: the n2u base-path routing loads are BEFORE ACQBULK in this build (layer comment says otherwise): racy if
  anything launches n2u before pr1's routing is written (IDF does: IDF uses the post-wait path).
## Next (x9): read Human decisions IN FULL first (lead may want the Handoff ceb86cf bundled/checked, or slack follow-ups).
  Own list: (1) v1 IDF build (pr8 st.release routing flag + tk2f flag acquire at T=8 + nu8 routing after its wait; the
  flagless oracle HANGS); bound ~1 us v1 (~0.1%): build only if nothing better. d16/d32 routing: closed (not critical).
  (2) tk2f per-CTA routing at entry (~1.5 us d1/v1): precomputed schedule saves only the compute (~0.5): skip.
  Pitfalls: background Bash jobs default to a 10-min limit (use run_in_background + timeout); `pkill -f` matches itself.
## x7 (2026-10-30 11:44Z-11:47Z): idle stop again; still no lead line naming fresh after 09:10Z (layer n12s1 DEFNORM gate also not met). 0 judge, 0 Modal.
## x6 (2026-10-30 11:28Z-11:31Z): idle stop; no lead line naming fresh after 09:10Z (x5 answered the sweep: all dead). 0 judge, 0 Modal.
   Next (x7): same as x5 'Next' (act only on a new lead line naming fresh; else stop at once).
## x5 (2026-10-30 09:04Z-11:30Z): lead 09:10Z ORACLE SWEEP (6 items): ALL DEAD, none >= 1 us on a d-load. 0 judge, 0 Modal.
- (1) conc5.py: every vector pipe slows next to HGMMA (INT 2.1-2.7x, HMUL2 4.5x; FFMA serializes the HGMMA instead). (2) tlb.py: 0.
  (3) ptorc.patch + pt_order.py: chain <= ~1 us, KV order <= ~1 us (permute cache WITH the table, else routing changes = fake gains).
  (4) uniq.py + ncu: DRAM bytes = unique bytes. (5) layout.py: 0. (6) router real FP32. Tools in muse_decode_layer/tools/fresh/sw.
- Pitfalls: `pgrep -f X` in a wait loop matches the loop's own command line (cost 2 h idle); background jobs default to a
  10-min limit; the layer agent's ab_flag held the GPU lock ~20 min (flock is not FIFO).
## Next (x6): paused. Act only on a new lead line naming fresh after 09:10Z; else stop at once.
## x4 (2026-10-30 08:12Z-08:14Z): idle stop; no lead line after 08:10Z (x3 answered it: not confirmed, paused). 0 judge, 0 Modal.
   Next (x5): same as x3 'Next' (act only on a new lead line naming fresh; else stop at once).
## x3 (2026-10-30 07:54Z-07:58Z real clock): lead "08:10Z" (ahead of clock) lane rfport (RF read-port hypothesis), Step 1 DONE: NOT CONFIRMED. 0 judge, 0 Modal.
- tools/fresh/xd/conc4.py (H100, SASS-verified): 112 LOP3/stage next to HGMMA N64: 3 regs 505, 2 regs+imm 500, 1 reg (UR+RZ)
  497, 3 regs with .reuse 547, 2 regs with .reuse 560 (all 235 alone; N8/N32 same pattern). Slowdown is per instruction,
  not per register read; reuse slightly worse. Step 2 (dq rewrite for fewer reads) not started. FRESH.md 'x3'.
- Branch: lead named fresh/rfport, but refs/heads/fresh blocks a `fresh/...` branch name; committed on branch fresh.
## Next (x4): paused (lane closed by Step 1). Act only on a new lead line naming fresh; else stop at once.
## x2 (2026-10-30 07:29Z-07:31Z): idle stop; no lead line after 07:20Z (x1 answered it: gate fails, paused). 0 judge, 0 Modal.
   Next (x3): same as x1 'Next' (act only on a new lead line naming fresh; else stop at once).
## x1 (2026-10-30 07:12Z-07:30Z): lead 07:20Z lane "EXACT new expert-kernel design", session 1. 0 judge, 0 Modal.
- Merged main v173 (8841e7d) into branch fresh. H100 microbenches tools/fresh/xd/{pipes,conc,conc2,conc3}.py; write-up
  FRESH.md 'x1'. Pipe rates: LOP3/SHF/PRMT 2 cyc (ALU), IMAD 2 (FMA, co-issues with LOP3), HMUL2 ~1, IMAD.WIDE 4.
- NEW mechanism: an HGMMA in flight slows ALU work in another WG ~1.3-2.2x (grows with N, same for SS and RS); with scale-d 0
  (no accumulator read) the slowdown almost vanishes => dq/MMA serialization = FP32 accumulator reads through the RF.
- Gate (<= 0.80x twm consumer at bs128+bs256) FAILS for all 3 candidates: pipe-balance (1 rotate via IMAD.WIDE) ~0.93x;
  overlap blocked by acc reads (fixed per MAC; FP16 acc banned); lead's (a) cluster dq / (c) tile shape fail by arithmetic.
  Posted Results + `fresh -> lead` in COORD. Per the lead's rule: gate fails -> fresh pauses itself.
## Next (x2): paused. Act only on a new lead line naming fresh. If the lead wants more on this lane, the cheapest open test is
   ncu RF-bank/dispatch metrics on conc.py roles (1,3) vs (8,3) to confirm the acc-read reading on hardware counters.
## f1s19 (2026-10-30 01:59Z-02:01Z): idle stop; no lead line after 01:55Z, proj's table not posted yet. 0 judge, 0 Modal.
   Next (f1s20): same as f1s18 'Next'.
## f1s18 (2026-10-30 01:52Z-02:05Z): lead 01:55Z: merged a412f50 into main as v172 (e80d4c3; build.py reproduces
   submission.py bit-for-bit) and SUBMITTED once: ae6c4bd3 2.1453 raw (CI 2.113-2.177), fixed-SGLang 2.1175 (same code's
   check 8b4e6cf7 = 2.1608): slow draw on our side; board keeps 2.158. No resubmit of the same code (lead rule).
   Fixed-score: copy profiles/submission.<job>.* into muse_decode_layer/profiles, run tools/fixed_sglang_score.py with the
   path pointed at r-fresh and the job added (see /tmp/fx.py pattern). Budget 10-30: layer checks 1/6, submits 1, Modal 7.8/20.
## Next (f1s19): fresh PAUSED per lead 01:55Z unless proj's table hands fresh a build. Read Human decisions IN FULL; act
   only on a new lead line naming fresh; else stop at once.
## f1s17 (2026-10-30 01:36Z-01:40Z): idle stop; no lead line after 01:00Z. Bundle a412f50 fails the lead's own submit rule
   (raw 2.1539 < 2.158), so no submit; re-checking the same code for a lucky draw is not allowed. 0 judge, 0 Modal.
   Next (f1s18): act only on a new lead line naming fresh (merge/submit/new lane); else stop at once.
## f1s16 (2026-10-30 01:21Z-01:24Z): idle stop; no lead answer yet to the f1s15 Handoff (Human decisions newest = 01:00Z).
   0 judge, 0 Modal. Next (f1s17): same as f1s15 'Next' (submit a412f50 only on a lead OK; else stop quickly).
## f1s15 (2026-10-30 01:03Z-01:30Z): lead 01:00Z FRESH = BUNDLE: built, checked, Handoff posted; 1 judge check, 0 Modal
- Bundle = branch fresh a412f50 = main v171 + HOT_XN 16 (T=256 HOT_XOFF TUNE option dropped). Searched the record: every
  proj/moe Handoff is already on main; moe n7s53 closed its leftovers; so HOT_XN is the only exact item left.
- v16 sliding last-CTA merge (LCM: the last nvs CTA per (request, KV head) merges the S=4 splits, no _combine_hb): built,
  correct (1 bf16 ulp vs combine, scored-shape checks pass), DEAD after 3 forms: runtime-S loops nvs +18 us; compile-time
  S, 128 thr +6.6; 256 thr one L2 trip +3.5 us nvs / in-layer attn (tools/n8/attn_orc.py) 14.5 -> 15.5 us. The chain
  store-ack fence -> atomic -> L2 loads on one SM is as long as the combine's. Patch tools/fresh/lcm_dead.patch.
- bench test 13/13; H100 ab_files verify neutral. Check 8b4e6cf7 2.1539 (CI 2.122-2.186), fixed-SGLang 2.1607 (best on
  record; main v171 2.1337) but shapes HOT_XN can't touch drew fast too (v1 full 120.2, v16 sl 254.9) = mostly draw.
  Raw < 2.158 -> not submitted; Handoff in COORD asks the lead to merge / decide.
- Budget 10-30: layer checks 1/6, Modal 7.8/20 (incl. f1s12).
## Next (f1s16): read Human decisions IN FULL for the lead's answer to the f1s15 Handoff. If the lead OKs a submit of
   a412f50, submit it (okc submit muse_decode_layer submission.py in muse_decode_layer/) and log it. Otherwise no lane is
   open for fresh: stop quickly unless a new lead line names one.
## f1s14 (2026-10-30 00:26Z-00:28Z): idle stop, still PAUSED per lead 00:00Z (no new lead line). 0 judge, 0 Modal.
## f1s13 (2026-10-30 00:10Z-00:12Z): idle stop, still PAUSED (no new lead line after 00:00Z; moe n9s2 closed the consumer
   rebuild by mechanism). 0 judge, 0 Modal. Next: act only on a new lead line naming fresh; else stop at once.
## f1s1..f1s12 + f1s20 (2026-10-29/30): trimmed; see git history of this file and FRESH.md (O1, O2, f1s*).
