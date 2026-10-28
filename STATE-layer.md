# STATE-layer (muse_decode_layer): short current picture; NOTES-layer.md is the long log

BOARD BEST = submit 3e3b9fd3 = v153 (3ef3268), **2.1580x**, #35 (n5s10; d1 66.3/75.0, d16 129.9/205.9, d32 155.7/308.2, v1 110.6/122.6, v16 266.6/356.1, v32 335.7/506.4). Previous: 35dd53d0 v125 2.1339x.
Same code checked 2.0731x (19163546): whole-run noise ~+-1.5%. Older: 19ebde23 v108 2.0860x. Lead 09:28Z rule: <= 2 'within 1% + per-shape evidence' submits/day (1 used 10-27).
night5 s11: **main = v156** (50355f2) = v155 + nvcc --split-compile=0 on moe/g10 libs + build pool 16 + 4 dead g30 builds dropped
  (local cold 142 -> 131.5 s; H100 12-shape A/B tie). check 0e4ba3c2 2.1421x, **judge prepare_s 182.0** (no gain on the judge's
  build host). 10-28 budget after it: 12 checks / 9 submits. No new layer lever found (NOTES n5s11).
night5 s10: **main = v155** = v153 + T=8 sl base g30n:o, T=8 full g14alt 3, T=32 base g30kgx:pooler, TUNE lists cut to the H200
  picks of the last 3 runs (lead 03:25Z: judge prepare_s <= 220 s): cold 139.4 s local, **judge prepare_s 176.7** (v153 189.0).
  check cd6b3ba4 v155 2.1074x (d16/d32 sl best ever 123.1/147.2; d1/v32 sl worse = run noise per tuner logs; not submitted).
  Next: prune builds that serve no TUNE option any more (g19 qkv/o T=32, g24sk2, lt alts...: check each against TUNE_BASE).
  v153 = v152 + proj G30_GX (branch proj/g30gx afc0c8b: TUNE options; H200 tuner: g30kgx:pooler T=32 sl -0.3..0.6 us picked,
  rest tie) - pr32 builds (cold 220 -> 161.5 s). check 60f02748 2.1373x -> submit 3e3b9fd3 2.1580x (board best). 10-28 budget
  after cd6b3ba4: 13 checks / 9 submits.
night5 s9: main = v152 = v150 + T=8 TUNE_BASE {cslpw, pr8_npf 220} (H100 real routings + H200 tuner agree); TUNE[1]/[8] tk2fx-off
  options dropped (real routings +0.5..1.7 us). check bb8ea548 (v151 = same + nu8d/nu8x off in the base) **2.1168x** (within 1%; v1
  2.358/2.237 not a clear win: not submitted). H200 tuner: nu8d/nu8x ON wins -1.1/-1.6 us (H100 said off): reverted to options in v152.
  v152 validate test/scored/opt/fuzz OK (v151 full ALL OK, cold 162.5 s). 10-28 budget: 15 checks / 10 submits. v152 not checked (tie).
  H100 d32 sl timeline (195 us): qkv 15, attn 19, o 7, pooler g30k 28 || router 20.5 + routing (ends 71), tk2f 97 (72-170), norm2,
  unpool ends 23 us after tk2f, bnd -> glue is back to back; experts are the gap (moe LANE 6).
night5 s8: **main = v150** = v148 + T=256 base {rg, twf_split 66} + T=16 full base bal (H100 REAL-routing A/Bs, 3 seeds: v32 sl
  -1/-9/-10, full -13/-14/-5; d16 full -2.5/-4.0/-1.5 us) + TUNE[256] 36->9, TUNE[128] 22->10 (routing-sensitive options that lose or
  flip sign on real inputs dropped: f824b925's tuner picked hot 24 on synthetic routing = v32 sl 378 us). validate ALL OK, cold 146.7 s.
  check 73c50eea (v149) 2.1048x (v32 sl/full 2.095/2.089 = good draw / best of 4; not submitted). 10-28 budget: 16 checks / 10 submits.
night5 s7: main = v148 = v147 + T=32 sl/full TUNE_BASE {g30k:pooler 104, g30n:unpool, g30n:qkv_gate} (escapes in TUNE[32]; H200 tuner:
  every escape loses +0.6..3.5 us). validate ALL OK, cold 154.4 s. check f824b925 2.0944x (noise: d32 full +12, v32 sl +45 vs 133068b8); not submitted.
  10-28 budget after it: 17 checks / 10 submits.
night5 s6: **main = v147** = v146 (v145 + proj G30_PL) + build.py: Python comments stripped + g22/ob8 sources not embedded:
  **the judge REJECTS files > 1,000,000 B** (v141-v144 were 1.05 MB); now 933.6 KB, build.py asserts < 1 MB. validate ALL OK, cold 154.1 s.
  10-28 checks: 4937830f v145 2.0907x, 133068b8 v146 **2.1097x** (d32 sl 152.2 / full 295.5 vs board 164.3/308.0: g30k104 + g30n unp+qkv
  picked at T=32 sl+full in both; v32 noise +11..38). Not submitted (1.13% < board; rule needs >= 2.1126x). Budget 10-28: 18 checks / 10 submits.
  Dead this session: T=8 pooler on g30k MTOK 8 + side router (no pr8): H100 v1 +5 us (router starves beside it).
Older session summaries (night5 s1-s5, s85-s107): NOTES-layer.md.

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
2. Any new Handoff from moe (attention lane) / proj: port as a TUNE option, validate, bundle into one check.
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
