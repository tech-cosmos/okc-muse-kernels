# STATE-layer (muse_decode_layer): short current picture; NOTES-layer.md is the long log

n64 (11-01 18:35-18:50Z, session 4): SUBMITTED v187 (lead yes) -> 122374d8 2.2028 < board 2.2048 (board keeps v185, #237). Full-attention TB/s + bare random-gather ceiling (dev/n64/gather_oracle.py, H100: B32 234 us = 2.83 TB/s, B16 119 us; contiguous 213/108): best attention d16/d32 full is only 7.5/7.7 us above the ceiling (lead's 15 us d32 gate unreachable); v16/v32 full 15/18 us above (q_len-8 compute/latency, ~0.2% realistic). Table in NOTES n64. Budget 11-01: 19 checks / 9 scored. NEXT: lead decides whether the v16/v32 full q_len-8 gap is worth a build; else hold for new-code handoffs.
n63 (11-01 18:20-18:25Z, session 3): status only, 0 judge, 0 GPU. Still no lead yes/no on submitting v187 under the 09:28Z rule (n61 Request); the 17:00Z bar (fixed > 2.2048) was missed, so no submit. Board: layer #1 pjmullin3 2.484; ours v185 2.2048, overall #236. 11-01 budget: 19 checks / 10 scored left. No new-code candidate (n61/n62 closures). NEXT: lead yes -> submit v187 (main 81e7675); else wait for a new-code handoff.
n62 (11-01 18:03-18:20Z, session 2): status + paper audit, 0 judge, 0 GPU. No lead reply yet to n61's v187 submit question (09:28Z rule; layer won't submit without a yes). Candidate scan: vjx on SLIDING (vjx_src supports windows) = H200 standalone (fresh x24) win only d32 sl -0.9 us (~0.05%), loses d1/v1/d16 sl -> below the 0.3% judge bar, not built; lead_ideas fable/codex items all closed on record (DEFNORM n12s1, fresh x5 oracles, megakernel forms). NEXT: lead yes -> submit v187 (main 81e7675); else hold for a new-code handoff.
n61 (11-01 17:55-18:05Z (sandbox clock), session 1): **MAIN = v187** (81e7675, sha1 d0ffbdd1; vjeux full attention, human-approved 17:00Z); bench test ALL PASS. CHECK 7b620515 2.2039 raw / fixed 2.1933 (= v185's) < bar 2.2048 -> NOT submitted. Full shapes best on record except d32 fu (303, noise range); shortfall = untouched sliding draws. H200 tuner: vjx wins T=1/8/128/256, ties T=32 (already per-shape dispatch). vjx plan re-tune (n_splits/coop, dev/n61/vjx_sweep.py) DEAD: default plan optimal on all 6 full shapes. NEXT: bundle v187 with the next new-code item; check; submit if raw > 2.2048. Asked lead whether v187 may be submitted under the 09:28Z 'within 1% + per-shape evidence' rule.
n60 (11-01 01:39-01:45Z, session 54): status only. No lead line after 01:20Z (n59 answered it: v187 rule check PASS); no STOP-layer yet but lead said layer pauses. Board #189 at 2.2048 (#1 pjmullin3 2.469). Hold main v186; v187 (layer/vjeux 3b2a200) waits on the human. 0 judge runs, 0 GPU.
n59 (11-01 01:23-01:35Z, session 53): lead 01:20Z rule check on v187/vjeux = PASS (SASS: HMMA.16816.F32 + QGMMA.*.F32.E4M3.E4M3 only, no FP16 math, FP8 KV upcast in registers only; details NOTES n59, COORD Results). 0 judge runs, 0 GPU. Layer pauses (lead STOP). NEXT: on human approval, one check of v187, submit if > 2.2048 raw.
n58 (11-01 01:07-01:12Z, session 52): status only. No lead line after 00:35Z (human's vjeux/QLO decision pending); moe/proj/fresh all STOPped, no Handoff. Board #1 pjmullin3 2.469; ours v185 2.2048. Main v186; v187 ready on layer/vjeux 3b2a200. 0 judge runs, 0 GPU.
n57 (11-01 00:44-01:10Z, session 51): lead 00:35Z -> VJX PREPARED on branch layer/vjeux = **v187** (submission.py there; main stays v186): vjeux's Apache-2.0 attention (vjx_src.py verbatim + attribution header + full LICENSE) for FULL layers only; default on T=1/8/32, tuner option T=16/128/256. H100 pinned-routing in-layer: d1 -0.1..-1.2, v1 -2.8..-3.4, d32 -0.6..-2.7 us; d16/v16/v32 +1..3.5 (tuner decides on H200). Decode-full error vs reference 8x lower (FP16 QK at q_len 1). validate --quick ALL OK, prep 71.3 s, file 998,087 B (1.9 KB headroom). ZERO judge runs; check only after a lead line says the human approved.
n47 (10-31 22:35Z, session 41): MAIN = v185 = v183 + fresh x35 CSL_PW at T=1 w/o carveout (sha1 d837bea5 = f064596; H100 d1 sl -1.2..-1.4 us, bitwise; validate --quick ALL OK, prep 63.8 s). Same code already checked by fresh 4a3434a9: fixed 2.1629 < bar 2.1913 (v32 sl slow draw, untouched shape) -> NOT submitted, not re-checked. NEXT: bundle v185 with the next new-code candidate after 00:00Z; one check; submit if fixed > 2.1913. Fresh stamps: no other hop >= 1 us (v1 early start <= 0.5).
n48 (10-31 22:55Z, session 42): SUBMITTED v185 (main d837bea5) per lead 09:28Z rule (its check 4a3434a9 raw 2.1959 > board, d1 sl 64.4 best on record = its target, only v32 sl slower = untouched shape within per-shape noise, validate ALL OK) -> fcb67ee9 **2.2048x** (CI 2.172-2.238) = NEW BOARD BEST, #180. Per-shape speedups: d1 2.71/2.44, d16 2.11/2.16, d32 1.75/2.12, v1 2.41/2.28, v16 2.06/2.04, v32 2.24/2.18 (sl/full). Do NOT resubmit v185. NEXT: new code only; bar = beat 2.2048 board.
n49 (10-31 22:38-22:55Z real clock, session 43): status + one audit. v1 sl g14 qkv 2.9 us CTA-end spread (fresh x35) = the bare HBM reader's own spread (n10: g14 at the reader floor in first-op position; n13s2 SM-weighted split dead) -> no build. Pooler L2 prefetch in the csl/o window = dead on record (pr1 npf, G10_PFL2, oracle 2 n7). No candidate; layer 10-31 left 4 checks / 5 scored, unused (no new code). NEXT: hold v185 (board 2.2048); new code only.
n50 (10-31 22:56-23:00Z real, session 44): status only. Lead 22:55Z: LAYER = hold v185 (board 2.2048, #180 of board; #1 pjmullin3 2.469), integrate handoffs; fresh stamps d16/d32 sl hops, moe stamps v1 sl expert hops. No handoff since x35 (in v185). 0 judge runs. NEXT: integrate fresh/moe hop builds when posted; one check per bundle with H100 sum >= 1 us; submit if raw > 2.2048 / best fixed.
n52 (10-31 23:27-23:45Z, session 46): d16 sl kernel timeline on v185 (H100): all boundaries PDL-overlapped; hops _routing tail 2.6 (n14s3), unpool->bnd 2.1 (dead forms), rmsnorm->tk2f 0.5; nothing new >= 1 us at kernel level. Hold v185; 0 runs. NEXT: integrate fresh/moe hop builds; 11-01 budget fresh at 00:00Z.
n51 (10-31 23:11-23:15Z, session 45): status only. No lead line after 22:55Z (hold v185, integrate handoffs); no Handoff since x35 (in v185); moe o42 claims v1 sl expert hop stamps, fresh d16/d32 sl stamps pending. Board #180 2.2048 (#1 pjmullin3 2.469). 0 judge runs. NEXT unchanged (n50).
n53 (10-31 23:43-23:58Z, session 47): MAIN = v186 (4c405e9, sha1 d41255da) = v185 + moe o42 nu8 mmq (H100 v1 sl -0.67 / full -0.73 us; validate --quick ALL OK). Check 1d07cbf1 2.1875 raw < board 2.2048 (v1 sl/full 2.375/2.286 vs v185 2.41/2.28 = noise; d1 sl untouched slower = node) -> NOT submitted. NEXT: bundle v186 + next new-code item, one check, submit if > 2.2048.
n54 (10-31 23:56Z-11-01 00:02Z, session 48): status only. Lead 23:55Z (bundle v185 + nu8 mmq as v186, check, submit if > 2.2048) is already done by n53 (check 1d07cbf1 2.1875 -> not submitted); identical code can't be re-drawn (judge dedup) or resubmitted. No new Handoff (fresh x39: no hop >= 1 us; moe o42 rider = v186). Waiting on fresh's 00:00Z CSL_PW H200 A/B (confirmation only). HOLD v186 main / board v185 2.2048. 0 judge runs; 11-01 layer budget untouched (20 checks / 10 scored).
n55 (11-01 00:12-00:20Z, session 49): status only. No new lead line/Handoff; fresh x39b H200 confirms CSL_PW T=1. Waiting on the human for vjeux full-attention copy (fresh x39c: ~+0.6% layer if 1:1) / QLO full half. Hold main v186 / board v185 2.2048 (#185). 0 judge runs.
n56 (11-01 00:28-00:35Z, session 50): status only. No lead line after 23:55Z, no Handoff/Claim; moe+proj have STOP files (no handoffs coming). Board #1 pjmullin3 2.469; ours v185 2.2048. Hold main v186. 0 judge runs, 0 GPU. NEXT unchanged (n55): vjeux/QLO wait on the human.
BOARD BEST = submit fcb67ee9 = v185, **2.2048x**. Main = v187 (checked 7b620515 2.2039; submitted 122374d8 2.2028, not best).
Previous BOARD BEST = submit bc16338a = v183 (e0dd26f, main), **2.1874x**, #211 (n44, 10-31 21:32Z; d1 65.7/73.5, d16 124.4/204.8, d32 151.6/297.0, v1 106.4/125.5, v16 252.0/343.1, v32 323.7/497.6). Robust to non-SA weights.
Previous BOARD BEST = submit 50534b24 = v174 (c62cdb0), 2.1717x (crashes on non-SA weights; superseded).
n46 (10-31 22:15Z, session 40): lead 21:50Z attn-history audit: layer csl/vtk = attn main kernels + layer PDL/QSS extras; every attn-board sliding T=1/T=8 standalone winner is already in the layer; rest tie/lose (list in NOTES n46). Only unbuilt item: v1 KV_PF (<= 0.3 us, under gate). Waiting on fresh's per-CTA stamps to pick a hop. 0 runs.
n45 (10-31 21:40Z, session 39): status only. Board #211 at 2.1874 (#1 pjmullin3 2.459); no new lead line after 21:25Z, no candidate (every lever on record closed; QLO full half + vjeux attention wait on the human). HOLD. 0 runs.
n44 (10-31 21:35Z, session 38): lead 21:25Z -> SUBMITTED v183 = 2.1874 (best). v184 00:00Z submit CANCELLED (same fix/speed). NEXT: hold; new code only, bar = beat 2.1874 board (and fixed-SGLang > 2.1913 on its check).
Previous BOARD BEST = submit 3e3b9fd3 = v153 (3ef3268), **2.1580x**, #50 on 10-29 (#35 when set) (n5s10; d1 66.3/75.0, d16 129.9/205.9, d32 155.7/308.2, v1 110.6/122.6, v16 266.6/356.1, v32 335.7/506.4). Previous: 35dd53d0 v125 2.1339x.
Same code checked 2.0731x (19163546): whole-run noise ~+-1.5%. Older: 19ebde23 v108 2.0860x. Lead 09:28Z rule: <= 2 'within 1% + per-shape evidence' submits/day (1 used 10-27).
n43 (10-31 21:20Z, session 37): status only. Board: we are #224 at 2.172x, #1 pjmullin3 2.459x. v184 on origin/layer/v181nsa still sha1 926ce2ec; layer 10-31 left: 5 checks / 7 scored (lead cap used). NEXT unchanged: submit v184 once after 00:00Z 11-01. 0 runs.
n42 (10-31 21:15Z, session 36): v184 file pre-submit stress, new seeds 31,37: replay 0/720, route_judge 0/720 (tools/n42). 0 runs.
  NEXT unchanged: submit v184 (70591ea, sha1 926ce2ec) once after 00:00Z 11-01.
n41 (10-31 20:50Z, session 35): status only. No new lead line; v184 file on origin/layer/v181nsa 70591ea verified (sha1 926ce2ec).
  Layer budget 10-31: 5 checks / 7 scored left but no candidate (and lead 2/day submit cap used). NEXT unchanged: submit v184 after 00:00Z. 0 runs.
n40 (10-31 ~20:55Z, session 34): H100 ab_files 4 rounds v181 vs v184 (branch layer/v181nsa 70591ea, = v181 + non-SA fix only;
  tools/n40/ab.log) = TIE on all 12 shapes (each within +-1.5 us, same as copy-vs-copy). By construction too: for SA weights
  the fix sits under an undefined #ifdef MOE_NSA, so the compiled kernels are v181's. **NEXT after 00:00Z 11-01: SUBMIT v184
  (`git show 70591ea:muse_decode_layer/submission.py`, sha1 926ce2ec) once**: v181's speed (its check 84eb0719 2.1896 raw /
  ~2.21 fixed is the speed evidence) + no crash on non-SA weights; v184's own check 612cdd17 2.1554 is within 1% of the board
  (09:28Z rule). Fall back to v181 (38d0c23) only if a lead/human line says so. Asked under Requests. 0 judge runs.
n39 (10-31 20:45Z): v181 recheck -> judge `status: duplicate` of 84eb0719 (content-hash dedup, 0 checks spent): identical code can't be re-drawn. Plan unchanged (n37 NEXT). If the 00:00Z submit returns `duplicate`, log it and ask the lead.
n38 (10-31 20:10Z): v183 cold prepare non-SA 112 s / SA 67.6 s (H100; tools/n38): OK. 0 runs. NEXT unchanged (n37).
n37 (10-31 ~19:55Z real): **main = v183** (e0dd26f) = v182 + NON-SA WEIGHTS FIX (moe o25 mode 2: zero/subnormal block scales
  crashed v181/v182 'MOE_PRUNE: tk2f cfg 84204 not built'; prepare now rebuilds the moe lib with MOE_NSA for such weights).
  non-SA 37/37, validate ALL OK. CHECK 662fa349 2.1414 / fixed 2.1551. Branch layer/v181nsa 70591ea = **v184** = v181 + fix only:
  CHECK 612cdd17 2.1554 / fixed 2.1252 (equally slow d1/v1 -> v182's escape rule is NOT slower; node noise). Generator weights
  are always SA (scales clamped >= 2^-6), so the crash needs real-checkpoint weights. **NEXT after 00:00Z 11-01: submit v181
  per the rule unless the lead picks v183 (main: same speed, robust to non-SA and held-out-fallback weights; recommended)
  (asked in Requests).**
n36 (10-31 19:25Z): status; energy lever (spin->nanosleep) DEAD by inspection (1 thread/CTA spins). 0 runs.
n35 (10-31 19:05Z): status only, no lead answer, nothing to build; plan unchanged (submit v181 after 00:00Z). 0 runs.
n34 (10-31 19:25Z): **main = v182** (856cc18) = v181 + proj n26 escape-base rule (_compress base emax-15 or -16, fewest escapes;
  board weights: qkv/pooler now emax-16, pooler max escapes/row 7 -> 5; held-out fallback 6/1000 -> 0/1000 per proj n26). validate ALL
  OK. H100 ab_files 6 rounds all 12 shapes = TIE (each within +-1.5 us; d1 full -1.0, v1 -0.8..-1.1). CHECK f8de6feb 2.1570 raw /
  fixed-SGLang 2.1508 < bar 2.1913 (node draw: d1/v1 sl slow, v32 sl fast). **NEXT after 00:00Z 11-01: submit v181
  (`git show 38d0c23:muse_decode_layer/submission.py`) per the rule, UNLESS the lead says v182 (same speed, robust to held-out
  weights; asked under Requests).**
n33 (10-31 18:45Z): T=16 {g30n:unpool} TUNE_BASE pin = H100 tie on 3 real-routing seeds (not pinned); d1 sl v181 timeline = record. 0 runs.
n32 (10-31 18:50Z): DIAG check 2657b536 (v181 + tuner SM-clock probe, tools/n32/mk_diag.py; never submit): judge H200 THROTTLES
  under sustained T=256 (1500 replays: sl 1926->1870 MHz +1.5%, full 1908->1840->1617 MHz +11%); T=16 flat; H100 flat. Mostly
  cancels in the paired SGLang ratio; ceiling ~0.2-0.4% via energy cuts (NOTES n32). Raw 2.1790. Plan: submit v181 after 00:00Z.
n31 (10-31 18:25Z): T=128/256 qkv TileLang split-K (m128 23.0 w/o reduce vs cuBLAS 25) + CUTLASS sm90 stream-K/multicast (m128 36
  vs cuBLAS 29.8, m256 43 vs 34.7, event harness) oracles DEAD; tools/n31 (system python3.14 has tilelang+CUTLASS hdrs). 0 runs.
n30 (10-31 17:52Z): T=128/256 qkv Triton split-K (plain + TMA/warp-spec) oracle DEAD: H100 m128 best 27.1 vs cuBLAS 24.9, m256 36.6 vs 29.5 us; tools/n30. 0 runs.
n29 (10-31 17:50Z): T=256/128 bnd2 L2-warm oracle DEAD (H100 warm -1.0/-0.6 us, ~0.04%); tools/n29/bnd_warm.py. 0 runs.
n28 (10-31 17:45Z): v181 HOT_IL stress tools/n28/il_stress.py: replay vs eager bitwise 0/360 + check 0/360 (routers 5/12/17, v16/v32 sl/full). 0 runs.
n27 (10-31 17:10Z real): **main = v181** (38d0c23) = v180 + proj n22 HOT_IL (IL=2) built with the new build.py comment strip
  (submission.py 909824 B: ~90 KB headroom; srccmp: compiled sources identical modulo comments). validate ALL OK. CHECK 84eb0719
  **2.1896 raw** / fixed-SGLang ~2.21 (> bar 2.1913, best on record). **NEXT (first session after 00:00Z 11-01): SUBMIT v181
  (main) once** (supersedes the v180 plan below), unless a lead line after 17:10Z 10-31 says otherwise.
n26s1 (10-31 17:10Z): **main = v180** (30d3612) = v179 + moe o17 HOT_ET race fix (rg zeroes meta[67] before its trigger)
  + fresh x18 nvs QLO. validate ALL OK (66.7 s prep). CHECK 55b9da18 **2.1977 raw** (best raw; fixed-SGLang 2.1629, node mix:
  v16 sl 267 / d32 sl 158 slow, v1 sl 104 / d32 full 294 fast). 999018 B.
  **NEXT (first session after 00:00Z 11-01): SUBMIT v180 (main)** once, NOT v178 (v178 has the demonstrated HOT_ET race), unless
  a lead line after 17:10Z 10-31 says otherwise. Its check beats the board raw; fixed bar 2.1913 missed by node noise.
n25s1 (10-31 17:05Z): **main = v178** (161c986) = v177 + proj n20 HOT_SL (>128-token experts as moe_hot 128-token slices).
  validate ALL OK (65 s prep); H100 v32 sl/full -6.0/-8.7 us mean (dominant routers -8..-16, control tie), bitwise.
  CHECK 9117bd11 **2.1886 raw** (> board 2.1717; fixed-SGLang 2.1874 vs bar 2.1913); v32 profile hot 133 + twm 193 vs e0f2159f 2.8 + 231.
  v179 (faf3590, main) = v178 + proj n21 balanced slices: H100 dominant v32 sl -1.5 / full -2.7 us vs v178, controls tie; validate OK;
  CHECK 36474b5e 2.1634 = slow node on untouched shapes (v1/d32 sl), v32 ~ v178. submission.py 999109 B (~0.9 KB under 1 MB cap!).
  (superseded by n26s1: submit v180) SUBMIT v178 = `git show 161c986:muse_decode_layer/submission.py`** once (not 10-31:
  lead cap). Then check v179 (main); submit it if that check beats the board. Then H200 A/B v176/v177 per shape.
n24s1 (10-31 16:05Z): v1 (T=8) seed spread = distinct local experts only (~9.7 us/expert H100, only tk2f moves; layer's own
  count +-1 vs ref from legal near-ties). No slow mode at T=8. tools/n24/seedspread.py, layer_distinct.py; tlmed SEED=. 0 runs.
n23s1 (10-31 16:05Z): v177's H100 v1 full +1.1 us = tuner noise (GD_PW off at batch 1, NVS_PW not at v1, CB_HB_TRIG
  ab_flag tie over 3 seeds). Nothing to build; v1 full spreads 128.7 vs 155.6 us by routing seed (H100). 0 runs.
n22s1 (10-31 16:20Z): **main = v177** (51eaeff) = v176 + fresh x13 tk2h MRG 2 + x14 attn PDL-secondary (NDS/NVS/GD_PW,
  CB_HB_TRIG). validate ALL OK (63 s cold prep). H100 d32 sl -4.6 / full -3.1, d16 full -2.3, sl -1.2, v1 full +1.1 us.
  CHECK a4582885 2.1776 raw / fixed-SGLang 2.1673 < bar 2.1913 (H200 node noise; only d32 sl visibly better): NOT submittable.
  Next: H200 Modal A/B v176 vs v177 per shape after 00:00Z 11-01 (fresh), then decide GD_PW/CB_HB_TRIG per shape.
n21s1 (10-31 15:35Z): v16/v32 sl+full H100 timelines of v176 (NOTES n21s1): no hop >= 2 us outside the record (T=256 unpool
  ingest-bound, bnd2 at its bytes, qkv T=128 g30 dead s101, gw_verify closed in ~/attn). 0 runs.
n20s1 (10-31 15:10Z): lead 15:00Z: SUBMIT e0f2159f v176 = 2.1497x (fixed-SGLang 2.1533; v32 slow node draw, SGLang slow too) <
  board 2.1717. v176 is now SUBMITTED: do NOT submit it again (the n18s1 'submit after 00:00Z' item is DONE). Bar for new code:
  fixed-SGLang > 2.1913. Waiting on fresh MRG 2 (d32 tk2h) Modal A/B after 00:00Z 11-01.
n19s1 (10-31 14:58Z): v176 d1/v1 H100 timelines (NOTES n19s1): no new hop; still waiting for fresh CW T>1; 0 runs.
n18s1 (10-31 15:05Z): **main = v176** (b051588) = v175 + fresh a269066 TK2F_CW (T=1 tk2f chunk-wait). H100 d1 sl -0.3, full
  -1.2..-2.2 us; test 13/13, replay 0/480. CHECK 0ac4a2f0 2.1584 raw / **fixed-SGLang 2.1913 (best on record)**.
  NOT submitted: 10-31 cap reached. (submit DONE n20s1: e0f2159f 2.1497, slow draw). Then integrate fresh's CW T=8/16/32 when handed off.
n17s1 (10-31 14:40Z): **main = v175** (da9b87c) = v174 + RF_EARLY_ALL (batched _red_fin triggers o at entry): H100 d16/d32 full
  -2.0/-2.2 us, rest tie; check d8fc46c5 2.1705 (fixed-SGLang 2.1846 best on record) -> submit 68c3ba66 2.1641 < board 2.1717
  (noise; don't resubmit unchanged). d1 full oracles: gd math 0.6, atomics 0.6, G13_O PF2 dead, red_fin trigger moves dead.
  Submit cap 2/day reached for 10-31. NOTES n17s1.
n16s1 (10-31 14:20Z): lead 13:58Z 'late' port: nothing maps (T=1/T=8 already have slot+norm3, PDL norm, late trigger).
  smem-base audit of all 147 v174 kernels CLEAN. n2u pre-wait routing (fresh x8 oracle ~-1.2 us H200) DEAD: H100 tie +0.3;
  n7s7's 'stale routing' was ptxas hoisting the .nc Y loads (tk2f output) above griddepcontrol.wait, also inside fresh's
  oracle gain. Branch layer/n2upre (correct, 0/80). Waiting for fresh's CW handoff. 0 runs (NOTES n16s1).
n15s1 (10-31 13:50Z): lead 12:25Z v1 IDF8 (pr8 routing flag) BUILT + DEAD: H100 on vs off v1 sl +0.8..1.1, full +0.8/-0.4 us
  (tk2f starts 2 us earlier but its 83 us stream ends the same). Branch layer/idf8 703418e, not merged. FINDING: pr8 edits
  cost +4.5 us v1 when pr8's dynamic smem base drops from 128-B (0x2880) to 16-B alignment; __align__(128) fixes it (the
  'ptxas' cost of n14s1 / proj n13s1c). Check the base in SASS after big-smem kernel edits. 0 runs (NOTES n15s1).
n14s3 (10-31 12:30Z): d16/d32 _routing tail is OFF the critical path (d16 routing ends 63.1 < rmsnorm 63.8; d32 before the
  pooler): free routing = 0; routing-fold + XN together ceiling ~-2.2 us d16 sl (~0.1% real), not built. Free first boundary
  ceiling d16 -1.5 / d32 -1.9 us, forms dead on record. 0 runs (NOTES n14s3).
n14s2 (10-31 12:15Z): lead 11:25Z SUB-GATE BUNDLE: layer items DEAD (H100): d32 qkv g30 reduce-scatter +0.1..0.7 us standalone
  (cluster reduce = latency, not bytes); uneven K split (rank 0 more stages) best -0.2 standalone, in-layer d32 tie; stealing/BAL
  not built. tk2f K-split: moe o7b DEAD. Whole bundle closed (sum 0). d16 lever left = _routing tail 2.9 us (moe o7) = proj 11:25Z item. main = v174; 0 runs. Patches dev/n14s2_*.patch, tools/n14/rs_ab.py.
n14s1 (10-31 ~12:15Z): moe/slack T<=8 (lead 10:55Z): T=1 slack fused into pr1 (branch layer/slack1 1fc464c on moe/slack, NOT
  on main, NOT for submission until approved): H100 d1 sl/full -4.6/-2.9 us mean over 8 seeds (~-2.7 expected), 0/600 route
  fails, 13/13. T=8 OFF: anything compiled into pr8 costs +1.5..5 us (ptxas), gain ~1.75. moe's separate kernel +9..11 us. 0 runs.
n13s5 (10-31 10:43Z): main = v174 = v173 + fresh x8 ceb86cf (pr1 bias/ls hoist + IDF on at T=1); H100 d1 -1.6/-2.1 us; 13/13,
  replay 0/480; check 2d1f9256 2.1514 (fixed-SGLang 2.1650 > v172 2.1597) -> submit 50534b24 2.1717 = board best.
n13s4 (10-31 10:45Z): no gate item; board analysis (COORD Results): slack x1.096 ~ #6-#9, moe-board gap 1.18x = faster legal experts exist.
  Offered moe the layer-owned selection sites for moe/slack (T=1 pr1/_router_fma, T=8 r3/pr8, T>8 _routing/rg). 0 runs.
n13s3 (10-31 10:40Z): lead 09:55Z item 2 MEASURED (tools/n13/orc_nobnd.py + tlmed.py, H100): free first boundary saves d1 sl
  1.0 / v1 sl 3.5 us (o end -> pr1/pr8 end); v1 forms all dead (OB8 barrier, pr8 prologue, deferred-norm pr8 = no smem/regs).
  Baseline unchanged (value math): the board shift is real speed. 0 runs. NOTES n13s3.
n13s2 (10-31 10:55Z): own-list item SM-WEIGHTED static split (byte share per %smid from measured rate) DEAD: H100 bare reader
  28/100 MB span equal 13.0/38.2 vs weighted 12.9-13.1/38.4-38.6; slow SMs reshuffle when bytes move (corr 0.13-0.19). Every
  layer-lane lever on record is closed; 0 runs (NOTES n13s2).
n13s1 (10-31 09:50Z): lead 10-31 09:55Z d32 qkv oracle GATE NOT MET (tools/n13/qkv_stamp.py, per-CTA globaltimer stamps, H100):
  g30n S3 qkv standalone 16.5 vs ~12.1 reader = math 1.1 | DSMEM reduce 0.85 | start ~1 | per-SM imbalance ~0.85 (GPC-level,
  repeatable; unpool bimodal 14.3 vs 16.9 us per CTA). Buildable ~0.6 < 1.5. v1 hops: all closed on record. BOARD SHIFT: 180 teams
  passed us 10-30/31; we are #212, top-50 cut 2.2465, #1 2.418 (NOTES n13s1). okc check of unchanged code = "duplicate" (dedupe, no run).
night12 s1 (10-30 09:04Z): lead 09:10Z DEFNORM (deferred pre-expert rmsnorm) GATE NOT MET: H100 hop pr1->norm->tk2f d1 sl 1.1 /
  v1 1.7 us; d16 routing-bound (0); d32 3 us but g30k stream-K slabs (needs XN, dead). Oracle hangs (as n7s4); s12 XN already
  measured no-norm-kernel at -0.4..-0.8. No build, 0 runs. NOTES n12s1.
night11 s2 .. night7 s1 (10-29/30): idle sessions, v164-v172 merges, attention/T=128 oracles: summaries in NOTES-layer.md.
Older session summaries (night7 s1-s10, night8, night10-11, night5 s1-s30, night7 s9/s11, night8 s3-s4, s85-s107): NOTES-layer.md.

## Per-load H200 us vs the s12 floor (board run 19ebde23 = v108; lead 05:07Z request; s74)
Score %/us = gain in total score per us saved on that layer type (geomean of 6 loads x (46 sl + 16 full)).
| load | sliding us / floor / gap | %/us sl | full us / floor / gap | %/us full | gap x weight (sl + full) |
| d1  |  67.7 /  40 /  27.7 | 0.177 |  75.8 /  44 /  31.8 | 0.062 | 4.9 + 2.0 % |
| d16 | 127.9 /  77 /  50.9 | 0.083 | 208.9 / 137 /  71.9 | 0.029 | 4.2 + 2.1 % |
| d32 | 173.3 /  85 /  88.3 | 0.058 | 326.8 / 208 / 118.8 | 0.020 | 5.1 + 2.4 % |
| v1  | 106.8 /  62 /  44.8 | 0.112 | 121.8 /  66 /  55.8 | 0.039 | 5.0 + 2.2 % |
| v16 | 274.6 / 127 / 147.6 | 0.041 | 368.5 / 189 / 179.5 | 0.014 | 6.1 + 2.6 % |
| v32 | 335.7 / 144 / 191.7 | 0.033 | 500.0 / 267 / 233.0 | 0.011 | 6.3 + 2.7 % |
Biggest H200 items (check c755dcc3 profiles): experts = 29-53% of every load (tk2f d1 20-28, d16 62-79, d32 70-103,
v1 54-64; twq/twm+hot v16 169-186, v32 217+114 us); full attention _gd_full d16 87.5 / d32 167.6 (at its ~4.0 TB/s gather
floor per ~/attn STATE); g24 pooler+unpool d16 35, d32 48; pr1 17.3 (d1). Layer glue (bnd, routing, norms) <= ~10%.
Judge noise: one shape moves up to 10% between runs of the same code (v108: d32 full 291 vs 327 us).

## NEXT (n5s8)
0. RULE (n5s8): routing-sensitive expert knobs (hot, twf_split, twq, bkt, bal, tk2h) are decided on real-input H100 A/Bs over
   >= 3 seeds (tools/ab_flag.py, SEEDS=99,7,13, variants with TUNE = {}) and pinned in TUNE_BASE; the H200 tuner's synthetic
   routing (SYNTH_P) does not represent the judge's inputs for them. Candidate: same sweep for T=8 / T=1 expert knobs (tk2f*).
Plateau: every kernel-level lever in NOTES s45-n5s7 is measured or bounded (dead-ends below + NOTES). H200 judge noise
(+-1.5% total; the tuner's own T=16 sl default moved 162 -> 180 us between nodes) hides < 2% changes: decide on H100
interleaved A/Bs (tools/ab_flag.py), confirm on a check. Submit only > 2.1339x, or per the 09:28Z rule (>= 2.1126x +
per-shape evidence). Open items, biggest first (none started):
1. o+bnd+pr1 megakernel at T=1 (bound ~-3 us d1 = ~+0.5%; NOTES s45/s61/s74; moe n5s9 audit: post-attention chain <= 3-4.5 us).
1b. DONE n5s16/n5s17 (v160, v161): no serial PDL hand-off left (timeline_all audit, NOTES n5s17).
2. Any new Handoff from moe (attention lane) / proj: port as a TUNE option, validate, bundle into one check.
3. (proj is prototyping it since 07:50Z 10-28; integrate when handed off) (n5s21, ~+0.19%, 2-3 sessions, below the 1% bar) T=128 'pr128': g30 FP32-slab pooler S2 (128 CTAs) that also carries the
   router's 256 hi rows (12-bit) + 256 lo rows (raw BF16, 2nd decode path) within <= 132 CTAs; _routing sums router slabs,
   moe_routegathern sums the 2 pooler slabs (gnorm). H200 Modal: pooler slab 22.3 vs cuBLAS 26.9 us; FR = pooler + ~3 us.
4. DONE n5s22: g30:unpool + fr stay on at T=128 (H200 real-input, 6/6 inputs).
T=16: every g30 option ties on 3 H200 checks (f824b925, 133068b8, 4937830f): don't add more g30 work at T=16.

## Board
- Best scored = see top (19ebde23 v108 2.0860x). Older bests: 703d3a7e v70 1.9954x, 371a17ee v42 1.9558x (NOTES s32/s27).
- Judge noise: SGLang's own per-shape times move 2-6% between runs; identical code's per-shape ratios spread ~+-1.5%,
  total ~+-1%: one check cannot resolve <2% changes. Decide on H100 interleaved A/Bs (tools/ab_flag.py), check to confirm.
- SGLang per shape (H200 us sliding/full): d1 171/183, d16 262-271/431-444, d32 261-268/614-634, v1 259-261/271-280,
  v16 523-537/700, v32 699-745/1049-1116.
- Scoring: geomean over 6 loads, each = 46 sliding + 16 full layers: 1 us on d1 sliding = +0.16% total, v1 +0.10%,
  d16 +0.07%, d32 +0.05%, v16 +0.04%, v32 +0.03%.
- Floors (us): d1 40/44, d16 77/137, d32 85/208, v1 62/66, v16 127/189, v32 144/267.
- Routing robustness: tools/route_judge.py 0/900 (s9). Output robustness: tools/replay_test.py (s10, 0/90 + 0/60).

## Code (muse_decode_layer/)
- `layer.py` = the layer. Inlined by **`python build.py` -> submission.py** (never edit submission.py by hand):
  `attn_src.py` = ~/attn submission.py @82866c6 (kernel_fn -> attn_kernel_fn, constexpr np2 fix, csl::decode patched
  to trigger PDL dependents); `moe_src.py` = r-moe 499bde6 (TK2F 204 at T=1, TK2F_DYN 84204 + moe_norm3 at T>1; tk2f trigger text-patched by TK2F_LATE_TRIG; T<=32 = moe_tk2f one launch; names prefixed `_moe_`; T>32 = its kernel_fn (moe_twf: gemm13+gemm2 in one persistent launch); layer.py patches moe_norm2 to
  trigger dependents); `g10_src.cu` = proj gemv10 (r-proj d894f4e, m=1), `g12_src.cu` = proj gemv12 (r-proj e71d015,
  m=8), `g13_src.cu`/`g14_src.cu` = proj gemv13 (m=1)/gemv14 (m=8) over lossless 12-bit BF16 (r-proj 3f173eb, bit-exact escape headers; packs `_pack13/_pack14` in `_g12_prepare`, used for qkv+pooler (G13_CFG/G14_CFG) when the weight compresses, built in `_g10_lib`), `lt_src.cu` = proj cuBLASLt explicit-algo GEMM (LT_CFG fixed H200 picks, `_lt_prepare`); layer.py `_g10_lib` builds g10/g12 in one load_inline and text-patches them (PDL trigger, PDL-secondary launch).
  Re-copy: `git -C ~/r-moe show <c>:muse_moe_w4a16/submission.py > moe_src.py` (check the 68-int route meta
  matches `_moe_bucket`, and that the patch anchors in layer.py/build.py still match: asserts fire at import).
- Router: T=1 `_router_fma` (Triton FP32), T=8 `r3_router` (CUDA FP32, in `_g10_lib`), T=128 proj `pj_router3` (`pr3_src.cu`,
  PR3_CFG; hi/lo one kernel, atomics), other T>8 `_router_split` (SGLang hi/lo BF16).
- KV_PF (T=1 sliding): `_linear(..., kv_pf=)` passes page_table[0, -2049:], k/v cache to the text-patched g13 (`_g10_lib`).
- gemv19 (s10): `g19_src.cu` (proj dev/gemv19.cu @6996d0c, built in `_g10_lib` with G19_ONLY2), `_pack19`,
  `_g19_prepare` (packs + per-pick F.linear check), `_g19(x, key, state)` -> FP32 [T, n] or None. G19_CFG (key, T),
  G19_ON, G19_GRID (pooler 108 blocks at T=16: leaves SMs for the side-stream router), BOUNDARY_TRIG. EPOCH COUNTER:
  one per (pick, grid size) (`e[3]` dict): a counter shared across grid sizes deadlocks.
- Pipeline: qkv `_linear` (gemv10 T=1, gemv12 T=8, cuBLASLt pick at T=128/256, else cuBLAS) -> attn -> o `_linear` -> `_boundary` -> side stream:
  router + `_routing` || pooler `_linear` -> `_rmsnorm_gain` -> experts (`_moe_kernel_fn`: tk2 at T<=32, route+gather+
  moe_tw at T>32) -> unpool `_linear` (1024-row slice at T<=8; gemv17 `_G17_SRC` at T=1/8) -> `_boundary`.
  T=1: experts = moe_tk2f launched by the layer (`_experts_n2u`) + `n2u` (moe_norm2 fused into the unpool GEMV, `_N2U_SRC`).
  T=16/32: pooler = 2-slab split-K bmm FP32 out (POOLER_BMM), slabs summed in `_rmsnorm_gain` (NS/SS).
  PDL: GEMV_PDL (gemv triggers; glue after it launches with PDL at T in PDL_TOKENS) and GEMV_PDL_SECONDARY
  {(tokens, sliding): keys} = GEMVs launched as PDL secondaries.
- bench.py `test` / `bench [file] [filter]` / `prof`; tools/timeline.py <file> "<load> <layer>" (per-kernel
  start/end of one replay); tools/ckparse.py <check.json> (per-shape table). variants/ (gitignored) builds for A/B
  (sub_v15/v16 ...): regenerate from git. Flags kept OFF (dead, s5): TW_TOKENS, TK2_PDL13/2, ROUTER_AFTER_POOLER,
  POOLER_HI_PRIORITY, ROUTER_L2_PF_TOKENS.

Older ranked-next list (s3-s64 items, all closed): NOTES-layer.md and the dead ends below.

## Dead ends (details in NOTES)
- s75: moe n21 twq K-split at T=128 (moe-only H200 -10.6 us) loses in-layer on H200 (+6 us sl real, +11..16 tuner).
- s74: PR1_BND (boundary fused into pr1 at T=1) H100 +1.3..1.8, H200 tuner +0.6..1.4 us; BND_SPLIT 1/2/4/8 per T: noise on both.
- s72: QSS (2-slab split-K qkv bmm, slabs summed at load) at verify sliding T=128/256 (nvs FP32 loads +1.6 us; H200 tie);
  moe_norm2 -> unpool prologue fusion (oracle: norm2 is free, it hosts the unpool's PDL window).
- s62: csl PW at d1 (+1.3..1.7 us: csl CTAs crowd g13 / lose KV_PF's warm rows); g19 full smem carveout so csl co-resides
  with the d16 qkv (csl starts at 2.8 us but its gathers slow g19 17.3 -> 20.6: net +0.3..0.6; kept as TUNE cslcarve).
- s61: nds at d16 sliding (+4..4.6 us); pooler W L2 prefetch at T=16/32 (g19 consumer-bound, warm L2 no faster); qkv T=16/32 at
  H100 cold-stream floor; T=1 pr1 SMEM preload bounded ~-3 us (needs o+bnd+pr1 megakernel; not started).
- s60: pr3 router at T=32 beside g19 pooler (+1.5..2.7 us in layer); pooler grid 108/120/132 all equal standalone (g19-bound).
- s15: n2u8 (T=8 norm3+unpool on clusters), bndc (cluster boundary), CUDA 1-CTA boundary, BND_ROWS, UPF3 (unpool
  W L2 prefetch at tk2f end / in norm; H200 diag b1ee542c d32 -4%), G10_PFL2 (o W prefetch during csl), pr1 npf 6-12,
  qkv/o g19 at T=32 (8 rounds H100 +2.6..3.4), qkv g19 at T=8 (m=8 unsupported). Patches: dev/n2u8_s15.patch, dev/bndc_s15.patch.
- s14: UPF2 (tk2f finishers L2-prefetch the unpool slice), N2U_BOUNDARY (still +2.8), BND_SPIN (flag-spin boundary:
  +2 us, race), g17 8/16 rows per block. RULE: no __ldg/.nc on producer data after griddepcontrol.wait (hoisted
  above it); register loads "before the wait" get sunk below it: use cp.async.bulk.prefetch.L2 / LDGSTS; check SASS.
- s12: XN (pre-expert rmsnorm inside tk2f: +0.2..0.7 us as its own instantiation; in the shared one it slowed
  d16/d32 by 1-2 us), UPF (tk2f L2-prefetches the unpool slice: tie). A/B kernel edits against the original binary.
- s10: g19 at T=32 (H200: proj diag + check 5cf5247d); g19 pooler on all 132 SMs (router starved, +4 us H100); router3
  at T=16/32 beside g19 (tie); o g19 at T=16 (H100 tie); d1 g13 pooler on fewer blocks (+0.7..1.1 us).
- s9: router3 at T=16/32 (+1..3 H100, crowds the pooler bmm); ROUTER_L2_PF (1,)/(8,) (noise); T=8 pooler g14 on 64
  fat blocks (+2..6); KV_PF at v1 (vtk barely gains from warm KV).
- s8: boundary as n2u's last-block epilogue (+2.7); n2u PDL after tk2f (+2..3); moe_norm2 PDL after tk2f (g_pdl 6:
  +1.1..1.4); bmm split-K for o/unpool at T=16/32 (+0.3..0.6); pooler bmm at T=128/256 (noise). RACE: gemv17 PDL
  secondary of norm2 at T=8 fails test 8 on repeats (unexplained; off).
- s7: fused pooler+router GEMM with cuBLAS heuristics (tie, kept as FUSED_ROUTER_TOKENS=()).
- s6: moe twf at decode T=16/32 (+22/+25 us H100); L2 prefetch of the unpool W from twf CTAs at their work-counter
  exhaustion (PF 16-40 MB: +1..5 us); gemv14 for o (verify full) and unpool slice at T=8 (+0.6..1.5 us).
- s5: moe_tw at decode T=16/32 (+14/+29 us); Triton split-K skinny GEMM (ties cuBLAS); tk2 gemm13 as PDL secondary
  of rmsnorm (tie); pooler-first / priority-stream pooler at T=8 (router starves: +2-3); router W L2 prefetch (T=8
  router is issue-bound); tl.dot FP32 router; cuBLASLt pooler m256 in the layer (+11.5 beside the router); pooler
  m16/m32 lt picks (judge noise). Race lesson: allocate side-stream-written tensors before freeing main temporaries.
- s4: router on few fat CTAs + pooler on the remaining SMs (HBM-bound together: tie); _red_fin PDL trigger (+2.5 us
  d1 full); gemv12 PDL secondary for unpool at T=8 (tie) and for o on verify-full layers (+1.3).
- s3: proj gemv7 as layer GEMVs (ties cuBLAS m1, loses m2-8); pooler before router fork; router grid sweeps (noise);
  L2 bulk prefetch inside boundary CTAs (-0.7 us) or concurrent on a side stream (+3 us); moe_big config sweep.
- Multi-CTA boundary (atomics + spin barrier); one-pass boundary (4 sums): both slower.
- PDL on glue after cuBLAS: no gain (cuBLAS never triggers; after our gemv10/12 it does gain, v12). Side-stream moe bucketing at T>=128: +29..67 us on the judge.
- L2 prefetch of the next weight on a side stream (load kernel or cp.async.bulk.prefetch.L2): GEMM speeds up but
  attention is delayed/slowed by the same amount; a side-stream fork at graph START costs ~9 us.
- GEMV with PDL preloading W before griddepcontrol.wait AFTER A BIG PRODUCER (s2: Triton o GEMV after attention): zero-sum. After small producers (norm2, csl at d1) it wins (v12-v14).
- torch.mm(out_dtype=float32) instead of F.linear: same speed.
