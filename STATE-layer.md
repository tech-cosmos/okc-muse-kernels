# STATE-layer (muse_decode_layer): short current picture; NOTES-layer.md is the long log

BOARD BEST = submit 35dd53d0 = v125 (2663c2c), **2.1339x**, #24 (s82; d1 67.4/75.0, d16 124.9/204.5, d32 164.3/308.0, v1 112.8/127.3, v16 273.0/352.7, v32 322.0/494.6).
Same code checked 2.0731x (19163546): whole-run noise ~+-1.5%. Older: 19ebde23 v108 2.0860x. Lead 09:28Z rule: <= 2 'within 1% + per-shape evidence' submits/day (1 used 10-27).
night5 s4: **main = v143** = v142 + moe PR_SA pr32 (r-moe 11d49b7 pr32_src.cu; option only). v142 = v141 + **g30n o** in TUNE_BASE at T=32 sl/full
  and T=16 full (12-bit o, K 1024: each CTA's W share ~ its ring, filled under PDL before the attention ends; o_pf -> g30 pack);
  H100 ab_flag d32 sl -1.5..-1.9, full -1.8..-2.2, d16 full -2.4 us (d16 sl +0.8 vs g27: option only). Escapes {g30n:o False}(+lt:o).
  Dead: qkv g30k stream-K on 132 CTAs (standalone tie); _combine_hb early trigger (CB_HB_TRIG, -0.3 noise, inert False).
  **00:00Z: check 1 = v143**; submit if > 2.1339x or per the 09:28Z rule; read 'layer tune T=16/32' lines (g30n:o False must lose;
  paste g30k/g30n/pr32 lines to proj/moe).
night5 s3: main = v141 = v140 + proj/g30k139 deb4a62 (g30k stream-K pooler, g30n qkv_gate actually built (uneven S3), moe pr32 one-launch
  pooler+router+routing; all TUNE options at T=16/32, off by default). validate --quick ALL OK (opt 0/344, fuzz 0/508), cold prepare 153.6 s.
  H100 tuner T=32 sl: default 220.3, g30k104+g30n unp+qkv 215.0 (picked); full 439.8 vs 435.1 (g30k104+unp). T=16: ties (+-1).
  **00:00Z: check 1 = v141** (covers v140 stream fix + g30k/g30n/pr32): submit if > 2.1339x or per the 09:28Z rule; paste the
  'layer tune T=16/32' g30k/g30n/pr32 lines into NOTES and to moe (pr32) / proj (g30k).
night5 s2: **main = v140** = v139 + OWN STREAMS: side_stream/hi_stream were torch pool streams; the harness's capture stream (pool,
  round-robin 32) could BE our side stream, so the side branch serialized (H100 first capture after prepare d16 sl 201.6 vs 192.2 us;
  also every 32nd tuner candidate). Now cudaStreamCreateWithPriority via ctypes -> ExternalStream (NOTES n5s2). validate ALL OK, cold 151.4 s.
  **00:00Z: check 1 = v140**; submit if > 2.1339x or per the 09:28Z rule. moe asks: paste the 'layer tune T=16/32' pr32 lines to moe once pr32 is in a bundle.
  (v139 = v138 + ZR (g30n unpool -> the pre-expert rmsnorm zeroes the G24X qkv buffer: no trailing zero_
  kernel; H100 d32 sl -1.3 us both seeds) + v138's g30n:qkv_gate TUNE options removed (no-ops: S=3 needs K % 384). validate --quick ALL OK, cold 166.5 s.
  )
night5 s1 (2026-10-27 19:4xZ): **main = v138** = v137 + TUNE[16] {g30n:unpool}/{g30n:qkv_gate}/{both}, TUNE[32] {g30n:qkv_gate}/{unp+qkv}
  (no new builds). validate --quick ALL OK on night5 (v137 cold 137.7 s, v138 cold 144.3 s). H100 tuner: T=16 ties, T=32 sl picks unp+qkv -2.0 us.
  **00:00Z: check 1 = v138**; read 'layer tune T=16/T=32' g30n entries; submit if > 2.1339x or per the 09:28Z rule.
  Judge profiles are NOT in git: recover any job's with `okc status <full 16-hex id> --json` (writes profiles/ into cwd); board run 35dd53d0d33246a1 restored.
s107: **main = v137** = v136 + proj/g30n 453790c (TUNE[32] option {g30n:unpool: True}, off by default; proj H100 d32 unpool -1.2..-2.1 us, tuner picks it at sl;
  proj validate ALL OK cold 122.3 s; my re-validate ALL OK, cold 125.3 s). **00:00Z: check 1 = v137** (one check covers v136 + g30n: read 'layer tune T=32' g30n lines;
  if the H200 tuner picks g30n:unpool at full too, consider moving it into TUNE_BASE[(32,*)]). Submit if > 2.1339x or per the 09:28Z rule.
s106: no build (sandbox moves ~18:00Z; checks 20/20; proj claimed the g30 T=16/32 port, branch proj/g30n: qkv/pool/unp/o, H100 per call d32 -1.1/-2.6/-2.9/-1.3, d16 -0.5/-0.4/-1.5 us vs g24).
  **00:00Z plan:** check 1 = main v136 (submit if > 2.1339x, or per the 09:28Z rule). check 2 = v136 + proj/g30n once proj's validate is ALL OK
  (read 'layer tune T=16/T=32' g30 lines; d32 is the target, ~-8 us/layer H100 = ~+0.4%). Before merging: cold prepare <= 200 s (2nd g30 lib).
s105: one-launch pooler+router g30 DEAD (H100: S2 + any router rows = 136 CTAs = 2 waves 43.7 us; S3 33.8, S1 37.0 vs cuBLAS FR 29.3; NOTES s105). Queue empty. main = v136. 0 runs.
s104: **main = v136** = v135 + proj g30_sort + G30_PRE (proj/g30pre 88ba088); validate --quick ALL OK, cold 117.8 s; H100 v16 tie (bitwise equal). Lead 16:45Z (2) qkv/FR-pooler g30 = s101/s102 DEAD; only one-launch pooler+router g30 left (<= -4 us H100; router lo rows do not 12-bit compress: 18 esc/row, needs raw BF16 stages; queued). **First check after 00:00Z = v136.** 0 runs.
s103: no build (SMEM-resident T=1 pooler preload dead on paper: csl latency contention, NOTES s103). Board #30. 0 runs.
s102: T=128 pooler on split-K g30 slabs (summed in the rmsnorm) + separate router: H100 v16 +7..17 us, DEAD (NOTES s102; dev/s102_g30sp.patch). Lead item (2) closed. main = v135 unchanged. 0 runs.
s101: qkv on g30 at T=128 (uneven split-K S3 + ticket reduce, dev/s101_g30split.cu, bit-exact) standalone 21.4 vs cuBLAS 20.0 us, in-layer v16 +0.1..4.1 us: DEAD (reduce chain ~5 us; NOTES s101). main unchanged = v135. 0 runs.
s100: **main = v135** = v134 + g30 (g30_src.cu: T=128 unpool on the 12-bit wgmma GEMM, proj s91 in-fragment escape patch, PDL + G30_TRIG) in TUNE_BASE[128]; TUNE[128] escape hatches {g30:unpool False}, + {g30:o True} options. H100 standalone 22.4 vs cuBLAS 25.6 us, bit-exact (dev/s100_g30exact.py); in-layer ab_flag v16 sl -4.1..-5.2, full -4.5..-5.5 us, G30_TRIG -0.6..-1.5 more (~+0.3% total if it holds on the H200). T=256 form DEAD (per-SM x ingest). validate --quick ALL OK, cold 119.7 s. **First check after 00:00Z = v135**: read 'layer tune T=128' lines (g30:unpool False must lose).
s99: lead 13:30Z item (2) BUILT standalone: dev/s99_g30.cu = 12-bit-W wgmma GEMM at T=128 (W DEC3-decoded into register A-fragments, x TMA, split W/x rings). H100: unp 21.6 vs cuBLAS 25.7 us, pool S2 25.4 vs 26.5; x-latency/ingress-bound, not HBM (multicast, K-rotation, 128-row CTAs all measured no better: NOTES s99). Escapes + integration not done; worth ~0.3% only if it holds on the H200. 0 runs.
s98: T=128 split-K bound for moe's xr (NOTES s98): FP32 red.add ~2.1 TB/s (FR T128 ~17 us of reds; DSMEM cluster reduce no faster); xr = 44 vs cuBLAS 29 us, red-free 27.5: dead unless red volume /4. 0 runs.
s97: plateau, no build, 0 runs (e4m3-cvt dequant re-derived = moe dead end). s96: s95's g24 m32 M-split idea measured DEAD (H100 consumer oracle: 2x warps at m16 only -2.4%, split would double the decode; NOTES s96). s95: no build (plateau). s94: **main = v134** (566828e) = v133 + proj/qf35 (moe s35 d1-full qf + proj s68 TUNE[128] ltalt:pooler; proj full validate ALL OK, cold 116.5 s). First check after 00:00Z = v134.
s93: no build (plateau re-check, NOTES s93: d32 g24 pooler+unpool ~2.0 TB/s = biggest non-expert gap, proj-closed). s92: no build; verify-gap audit (NOTES s92): twm dq-bound ~57/80 us above stream rate at v16/v32 (~6.6% weighted) = biggest gap to #1, closed lane; all else <= ~1%.
s91: **main = v133** = v132 + TUNE[8] {cslpw, pr8_npf 206/220/112} (H200 picks pr8 bulk-PF depth; no new compile; validate ALL OK incl. fuzz, cold 118.5 s). First check after 00:00Z: main v133; read 'layer tune T=8' pr8_npf lines.
s90: main v132 + proj stream_fuzz tool (validate step fuzz): full fuzz 0/1462 bad. 0 judge runs.
s89: **main = v132** = v131 + moe s33 verify-1 full attention spec=True (H200 standalone -0.65 us, H100 in-layer ~-0.4; validate
--quick ALL OK, cold 118.9 s). proj/attn31 stack = H100 tie, left out (NOTES s89). First check after 00:00Z: main v132.
s88: main was v131 = v130 + pr8 template generalized (light 512-thr / x-ring configs, none compiled). Light pr8 and
OB8+light pr8 measured DEAD on H100 (NOTES s88): pr8 ends at the same time whatever precedes it; boundary already hidden.
pr8's stream rate (H200 ~17 us after bnd vs 13.7 floor) was already worked by proj s36-s38/s80 (PF, bulk/2nd-wave prefetch,
RGI, stealing bound ~2 us): no lever > ~1 us. First check after 00:00Z: main v131 + any handoff bundle. validate --quick ALL OK, cold 116.8 s. 0 judge runs.
s87: main = v130 = v129 minus all {pr1: False} TUNE[8] options (lead 11:50Z: never picked on 22 H200 logs incl. board run
35dd53d0, +1.7..5.8 us; board run clean, no withdraw). validate --quick ALL OK, nu8_race 0/180, cold prepare 147.8 s. 0 judge runs.
First check after 00:00Z: main v130 + any proj bundle (light pr1/pr8 asked, COORD Requests s87).
s86: main = v129 = v128 + fix of a latent race in the non-default T=8 {pr1: False} path (nu8d rmsnorm read side-stream
local_ids before the join; old main failed 7/30 seeds there; tuner never picked it) + inert OB8 code. Lead 10:50Z lane (a) built
and MEASURED DEAD on H100: OB8 (T=8 o GEMM + first boundary in its epilogue, grid barrier, ob8_src.cu) v1 sl +2.2..2.5, full
+0.2..0.6 us: pr8/pr1 fill a whole SM (1024 thr x 64 regs) so they cannot ramp beside o blocks; the separate 16-program boundary
hides pr8's W ramp, a barrier exposes it. T=1 analogue closed by the same mechanism (+ PR1_BND s74). NOTES s86. 0 judge runs.
Only open form: a lighter pr1/pr8 (<= 512 thr) that co-resides with o (proj redesign; ask under Requests).
s85: main = v128 (536adfd) = v127 + moe s31 verify-32 sliding 2-CTA cluster nvs (attn_src _NVSCL, TUNE nvscl default on;
H100 v32 sl -1.1, H200 tuner -0.9 us). Check 940a4fca 2.0943x (noise; not submitted, < 1% rule). Budget 10-27: 0 checks / 4 submits.
Lane question to the lead still open (Requests s84). tools/attn_probe.py (proj) now on main: run --only=<shape> --passes=0 after attention changes.

CLOSED lanes (measured): tk2f (s81 H200 diag, <= 0.9% unreachable), full attention in-layer (s84, ~0.15%), T<=8 glue/tail
(s78), o+bnd+pr fusion T=1/T=8 (s86), light pr8 (s88). Remaining open: moe/proj handoffs (attention csl/vtk/nvs ports), lighter pr1/pr8.

Prepare-time rule: the judge fails prepare > 300 s. Measure cold with tools/prep_time.py (fresh TORCH_EXTENSIONS_DIR +
TRITON_CACHE_DIR) after any build/tuner change; keep <= 200 s. New load_inline builds must go on _BUILD_POOL (see _PREBUILD).
Judge noise dominates: identical-ish code 2.04-2.12x across checks; per-shape T=32 moves 5-10%.


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

## NEXT
00. (s79: still no answer, no handoffs; see NOTES s79) (s78) Layer-own structural items re-bounded (NOTES s78): T<=8 prologue fusion ~0, persistent T<=8 tail <= ~0.5%,
    T=1 o+bnd+pr1 megakernel <= ~0.25% (o/csl window has no spare HBM). Waiting for the lead's lane decision (COORD Requests s78).
0. (s77) Lead 06:41Z: up to 4 of the 10-27 checks go to moe as in-layer DIAG bundles (watch Handoffs for "moe DIAG for layer"; run on main v124, copy the tuner lines to moe). Don't spend checks on tuner-option ties.
   Only unbuilt layer item: T=1 o+bnd+pr1 megakernel (bound ~-3 us d1 = ~+0.5%, multi-session, NOTES s45/s61/s74).
1. Integrate new moe/proj Handoffs on top of main (v111) (stack several), H100 ab_flag, tests (bench test, replay_test, route_judge,
   cold compile < ~5.5 min), then check; submit only if the check beats 2.0860x.
2. Weakest ratios on the board run: d32 sl 1.589, v16 sl 1.923, d16 sl 1.976: experts (tk2f/twm) are moe's lane; ask moe.
3. okc check can exceed a 600 s tool timeout: run it in the background, poll `okc jobs`, read with `okc status <job> --json`.

Mains v71-v88 (older, H100-validated): NOTES s32-s44.

s19-s23 (v48-v55): generic prepare() tuner `TUNE` (layer.py): per (T, sliding/full) it times whole-layer graph replays of
option sets on synthetic inputs (shapes only) and keeps the fastest in state["tune"]; `_opt()` reads it at trace time.
Options: g19:<key>, lt:<key>, ltalt:<key>, g10alt/g13alt/g14alt:<key> (1-based `_ALTS` index), pr1, pr1_npf, pr1_rb,
pr1_pf, kv_pf. Every option was forced and passed scored + replay tests. Defaults = v45 (checked 1.9634x).

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

## Next (ranked)
-4. (s64) d32 sliding attention: CLOSED. csl NC 2 (one wave at B32) loses to nds+combine (18.1 vs 17.4 us); real-input gather
   floor 12.3 (tools/gather_real.py) but nds is 14.0 us even fully L2-warm (pipeline-latency-bound): K/V L2 prefetch <= ~2 us,
   nds+combine fusion <= ~1-1.5 us at d32 sl (<= 0.1% each): not built. NOTES s64, dev/csl_nc2_s64.patch.
-3. (s15) Watch Handoffs: proj is building g19 mode 3 (BF16 out, per-tile tickets) for qkv T=16/32 (drops v44's 1.2 us
   cast at d16; m32 lost on H100 so far) and was asked for an M=8 g19 (T=8 qkv: g14 m8 is issue-bound, 13.0 us H200 vs
   g19 m16 10.6). moe works big-M (v16/v32 twm) on its calibrated H200 proxy (H100 with the expert grid capped at 88,
   x0.73): TRUST moe's proxy over full-grid H100 for expert kernels (it reversed the twm order call, v44 uses {}).
-2. s14 state of the tail: d1 n2u ends ~5 us after the last tk2f CTA (PDL release ~0.9 + y32 L2 trip + reduce) + the
   boundary ~2 us; v1 tail 11.6 us H100 (norm3 3.7, g17 4.6, boundary 3.3). s15: the cluster fusion n2u8 and the
   cluster boundary bndc are DEAD (NOTES s15): the tail needs a non-cluster idea or nothing.
-1. Layer-side gap at T=16/32 is GEMMs: H200 d32 qkv cuBLAS 12.8 + splitK reduce 1.9, o 7.1 (cuBLAS); pooler/unpool now
   g19 (v44). The experts (tk2f d16/d32, twm v16/v32) are 40-55% of every T>=16 shape: moe's lane.
0. Re-copy procedure (moe/proj): `git -C ~/r-moe show <c>:muse_moe_w4a16/submission.py > moe_src.py` (check the patch
   anchors in layer.py still match: asserts at import; layer overrides _moe_TWM_ORDER_M and _moe__lib (adds moe_set_upf));
   proj g19 patches: `git apply -p1` on r-layer. Then bench.py test, tools/scored_test.py, tools/replay_test.py,
   tools/tk2f_stress.py if tk2f changed, tools/ab_files.py old/new, check.
1. s55 DEAD: L2 prefetch of the routed experts' w13q beside the d32 g19 pooler (H100 tk2f -7 us/40 MB, but the prefetch
   blocks take the router's free SMs: routing +12..16 us; real ids only exist at the pooler's end; the real form XPF after the routing: tk2f only -1.1 us per 16 MB
   prefetched, its own L2 traffic evicts the rest: <= ~1 us even on the H200). NOTES s55, dev/xpf_s55.patch.
   s54 DEAD: d1 qkv/gate split with the gate GEMV beside csl (oracle +0.5..1.3 us, NOTES s54).
   Dead this session (don't retry without a new idea): every L2 prefetch of a next weight (unpool during tk2f/norm,
   o during csl, more pooler chunks in pr1) on H100 AND H200; boundary variants (cluster, CUDA 1-CTA, 2-4 rows per
   program); qkv/o g19 at T=32; csl overlapped with g13; T=8 norm3+unpool cluster fusion.

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
