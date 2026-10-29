# STATE-layer (muse_decode_layer): short current picture; NOTES-layer.md is the long log

BOARD BEST = submit 3e3b9fd3 = v153 (3ef3268), **2.1580x**, #50 on 10-29 (#35 when set) (n5s10; d1 66.3/75.0, d16 129.9/205.9, d32 155.7/308.2, v1 110.6/122.6, v16 266.6/356.1, v32 335.7/506.4). Previous: 35dd53d0 v125 2.1339x.
Same code checked 2.0731x (19163546): whole-run noise ~+-1.5%. Older: 19ebde23 v108 2.0860x. Lead 09:28Z rule: <= 2 'within 1% + per-shape evidence' submits/day (1 used 10-27).
night8 s3-s4 (10-29 21:31Z, 21:46Z): idle, no lead decision after 21:05Z; 0 runs. Lane closed (see s2); waiting for a lead assignment.
night8 s2 (10-29): v16 4-CTA cluster nvs merge BUILT + DEAD: max 30 clusters of 4 resident (120 CTAs < 128) -> 2 waves,
  H100 v16 sl attn 13.5 -> 22.3 us; extra 13.8 KB smem also cost v32 +1.6 (nvscl crosses the 196 KB carveout; L1-sensitive).
  Reverted (tools/n8/cl4_patch.py kept). Verify-sliding lane CLOSED: last-CTA global merge ~-1.3 us v16 (0.05%) not built.
night8 s1 (10-29): lead 21:05Z verify-sliding attention lane: ORACLES only (H100, NOTES n8s1), 0 runs, no build. Bound
  (stripped nvs): v16 sl -3.9 us (combine 3.3 of it), v32 sl -1.2. nvs body = serial seq->pt (2.3)->KV gather (2.1)->tiles
  latency chain; L2-warm all of it -0.4/-0.6 only; math/stores/transposes small. Only item: v16 4-CTA cluster merge + O_PF
  (~-2 us, ~0.08%). Lane CLOSED pending lead. main = v168/v171 unchanged.
night7 s11 (10-29): no build, 0 runs. g30kgx 'drop' on the judge = records don't fit (weight-dependent), not a build
  flake (~0.01%, skip). Layer lane idle: asked lead to pause the loop until a moe/proj Handoff (COORD Requests n7s11).
night7 s10 (10-29): main v168 checked a5b4ddab 2.1540x (fixed-SGLang 2.1343) < board 2.158: not submitted. No build (no lever
  >= 0.3% found). Judge sometimes drops g30kgx (build flake, 3/12 logs; graceful fallback). 10-29 left: 14 checks / 7 submits.
night7 s9 (10-29): no build, 0 runs. v1 T=8 o post-attention latency (H200 5.9 sl / 7.6 full us) re-sized = closed
  (pr8 end is set by its own pre-wait W stream, s88 / n5s5). Layer lane: nothing >= 0.3% left; d1/v1 full attention native
  kernel (~0.5%, multi-session, s73 lost) needs a lead assignment. main = v168 unchanged.
night7 s8 (10-29): IDF lane CLOSED: pooler-progress throttle (tk2f waits for 0/50/85/97 of 100 pr1 pooler blocks) = no change
  (all +4 us); the loss is n2u's post-wait routing reads (+4.3 us); with base n2u reads IDF ties base (H100 d1, NOTES n7s8).
  Early tk2f at T=1 is dead by mechanism (x not ready before pr1+rmsnorm). 0 runs. main = v168 (= v167 timing; check 59013cf5
  2.1553 < board 2.158). NEXT: bundle the next moe/proj handoff into one check; layer-lane builds left are all <= ~0.3%.
night7 s7 (10-29): main = v168 = v167 + IDF option (off): pr1 routing release flag + early moe_tk2f at T=1 (flag spin,
  rmsnorm TRIG at start, n2u reads routing after its wait, tail boundary resets flag). Correct (replay 0/80 on, 0/72 off), but
  H100 d1 +3 us (sl 89.2 vs 86.0, full 109.3 vs 106.3): early tk2f CTAs slow pr1's pooler stream ~3 us. DEAD unless moe's DYN
  throttles early CTAs. PITFALL: n2u's pre-wait routing loads are safe only because ptxas sinks them (NOTES n7s7). tools/graph_eq.py.
night7 s6 (10-29): v167 check 59013cf5 2.1553x (fixed-SGLang 2.1482, 2nd best; v16 sl 258.0 / v32 sl 316.7 best ever) < board 2.158:
  not submitted. routing->rmsnorm fusion at T=16/32 sized <= 1 us (already overlap). NEXT: integrate moe TK2F DYN/pr1-flag and proj
  compacted-unpool when handed off; check bundles; submit when a check > 2.158.
night7 s5 (10-29): **main = v167** = v165 + proj LTFR (v166, check 93027e57 2.1341x, fixed-SGLang 2.1354 = tie, not submitted)
  + _routing packed int64 max (bitwise; H100 -0.9 us/call T=16/32, -0.2 T=128/256; validate ALL OK, unchecked: wait for moe's unpool
  skip bundle). Judge runs draw new input seeds per case (routing differs per run): v32 sl spread 320-378 us on same code.
  hbm_fold v1/d16 (NOTES n7s5): v1 tail ~10, pr8-wait ~4-5, pr8->tk2f ~4 us; d16 router/pooler drain -> routing -> tk2f ~9 us (H100;
  proj n7s4's H200 oracle says the d16 piece is worth <= 2-3 us there). ab_files placement noise +-1 us/shape (dup-file A/B).
  NEXT: lead's 08:00Z big-idea pick; else T<=8 early tk2f + pr1 ids flag + DYN (Requests n7s5, ~0.4-0.5%, 2 sessions).
night7 s4 (10-29): main unchanged (v165), 0 judge runs, 0 Modal. Lead 06:00Z big-bet step 1 MEASURED: tools/hbm_run.py +
  hbm_fold.py (nsys GPU-metrics DRAM BW folded per replay, NOTES n7s4). H100 d1 sliding: HBM < 50% busy ~26 of 85 us in 5 op-boundary
  bubbles (qkv->o 4.5, o->pr1 2, pr1->tk2f 6, in-tk2f 3, tail 8). Early-tk2f oracle (launch tk2f during pr1): live ids HANG (needs
  pr1 ids flag); stale-ids timing form = TIE (static T=1 tk2f split). NEXT candidate: TK2F DYN at T=1 + pr1 ids release flag
  (ceiling ~2-3 us H200 d1/v1, ~0.4%); first run hbm_fold on v1/d16 to see if the bubble pattern holds there.
night7 s3 (10-29): **main = v165** (d2bb0c2) = v164 + proj BNOS (T=256 bnd thinning, v32 bitwise): check 4787d92e 2.1632x, submit b87bd6aa 2.1219x
  (low draw, board stays 2.158; don't resubmit unchanged). 10-29 left: 17 checks / 7 submits, Modal layer 6.2/15. Batch-1 full attention sized + 3 oracles DEAD (NOTES n7s3):
  H200 gather floor 20.5 MB random 6.16 / dense 5.5 us vs in-layer _gd_full 10.7 + _red_fin 3.2; _red_fin->o fusion <= 1 us;
  pooler W L2 prefetch during attention zero-sum (gd +4, pr1 -1.5); d1 plan variants on H200 all lose. Only a new native
  batch-1 kernel at the random floor is left (~0.25% each d1/v1; ~/attn s73 native cfl lost).
night7 s2 (10-29): main unchanged (v164). n7s1 NEXT (a) RGR (routing fused into rg + grid barrier) BUILT + DEAD: bitwise,
  13/13, but H100 v16 sl +1.7, v32 sl +2.7 us (barrier ~1 us; rg was already PDL-overlapped). dev/n7s2_rgr.diff. (b) bounded
  0.04% (n5s28). Biggest layer-lane pool left = full attention vs a 4.4 TB/s gather floor ~2.1% (v16/v32 _gw_verify ~0.9%);
  table NOTES n7s2. Requested a fresh attention-full redesign lane in COORD. 0 runs.
night7 s1 (10-29): **main = v164** (e5d3272) = moe merge 9d2b3a8 (v163 + twm EPI/KSC2/REL + TK2F_REL 2; submit 725fc518 2.1350x, board
  stays 2.158 = #50, #1 2.403) + RG_XPRE (rg row loads before its PDL wait: H100 -0.6..-0.9 us v16/v32, bitwise). layer.py synced so
  build.py reproduces submission.py (moe had edited submission.py only). H200 Modal timeline of all 12 shapes + top-5 non-expert
  table: NOTES n7s1, dev/n7s1_tl_h200.txt. T=128 g30:o + bnd1 PDL: dead (tie). 10-29 left: 18 checks / 8 submits, Modal layer 2.5/15.
  NEXT: (a) T=128/256 _routing + rg one launch (grid barrier) ~-2 us; (b) T=256 boundaries ~-2..3 us each. Bundle with v164, then check.
night5 s20: **main = v163** (d02f7a6) = v162 + rg in T=128 base: H200 Modal (tools/h200.py) real-routing ab_flag 14/14 inputs
  (v16 sl -0.9..-3.5, full -2.3..-3.6 us); T=256/32/16/8/1 pins re-confirmed on the H200 (every escape loses; T=8 bal option is
  unbuilt now). check c9f263d6 2.1476x (d32 full 283.2 best; v16 sl node-slow), prepare_s 88.6; not submitted. 10-28: 6 checks / 7 submits left after v163.
  Use h200.py (3-4 GPU-min per 3-shape x 3-seed sweep, shared 60/day) to re-decide H100 'ties' before spending checks.
night5 s30: tk3/PR CLOSED on its zero-overhead ceiling (s29 H100 A/Bs minus the 3.2-3.8 us fixed part: d32 sl ~-1.6..-3 us,
  d16 none since tk2h/BAL already merge 2-group experts) = <= ~0.16% score; tk2f bs1 fallback is on record dead (moe n5s37).
  No build, 0 runs. Lane request in COORD. 10-28 left: 4 checks / 7 submits.
night5 s29: tk3 lane: BUILT "PR" (pair-role tk2f: k CTAs run 16-token pairs with a compact rolled TWO body, rest tk2f's
  1-group rows) on BRANCH layer/tk3 a2ed836 (option tk2pr, TK2PR build flag; main untouched). Correct 37/37. H100 d16 ~= non-BAL,
  +4 vs BAL; d32 tie..+5. DIAG check df997d8f 2.158x (= v163 redraw, PR not picked, not submitted), prepare_s 91.9: H200 tuner
  T=16 +3 (fixed overhead, no pairs), T=32 tie. NEXT: cut PR's ~3 us fixed overhead (L1 carveout from 200 KB dyn smem ~1 us,
  gemm2 long_sb) -> T=32 ~-3 us; then Modal real-routing A/B. Rule found: hot loops at kernel code offset >= 64 KB stall
  (no_inst ~50%), not only straddling ones. Micro: wgmma N16 not cheaper than mma.sync pair (NOTES n5s29). 10-28 left: 4 checks / 7 submits.
night5 s28: no build, 0 runs: pr1/qkv at d1 sit on the H200 bulk-stream fit (3.8 us + B/4.26 TB/s); co-resident pr1
  bounded <= 1 us; T=256 b1/b2 byte cut 0.04%. Lane-change request posted (lead 09:20Z). 10-28 left: 5 checks / 7 submits.
night5 s27: no build, 0 runs: H100 v32 sl timeline (experts 235 of 385 us); remaining gaps are expert-kernel
  (proj TWS drain, moe closed); lane request n5s26 still open. 10-28 left: 5 checks / 7 submits.
Older session summaries (night5 s1-s19, s85-s107): NOTES-layer.md.

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
