# STATE-layer (muse_decode_layer): short current picture; NOTES-layer.md is the long log

Updated 2026-10-27 01:30Z (s67): **main = v103 = v102 + proj s56 g24z (T=16 zero-ahead chain qkv->o->pooler->unpool). BOARD BEST =
submit b9685e03 = v103, 2.0606x, 10,977 us/step, layer board #30** (#1 2.380x). Previous: 129ad449 = v102 2.0398x; 703d3a7e = v70 1.9954x.
Per shape (H200 us, b9685e03): d1 70.3/77.0, d16 133.4/216.1, d32 164.7/320.7, v1 114.0/122.9, v16 271.3/360.5, v32 352.9/520.7.
Checks: c973ab6f v102 2.0817x, 0eb3484a v103 2.065x. Check->submit spread ~1-2% (d1/v1 routing draw moves 5-13 us).
H200 tuner picks per run are in NOTES s67 (T=32 {tk2h 4, lt:o} flips sign between runs: tuner-only; g24z wins on H200).
Budget 2026-10-27: 18 checks / 8 submits left after s67 (resets 00:00Z).

s68: **main = v105 = v103 + v104 (all JIT builds concurrent: cold import+prepare 314 s -> 111 s on the H100, lead 00:48Z
done) + moe n14 TK2F_SQ at T>1** (bitwise; H100 tie at T>1, +0.7 at d1 so off there). Check 18504372 = v105 (see NOTES s68).
Prepare-time rule: the judge fails prepare > 300 s. Measure cold with tools/prep_time.py (fresh TORCH_EXTENSIONS_DIR +
TRITON_CACHE_DIR) after any build/tuner change; keep <= 200 s. New load_inline builds must go on _BUILD_POOL (see _PREBUILD).

## NEXT
1. Integrate new moe/proj Handoffs on top of v103 (stack several), H100 ab_flag, tests (bench test, replay_test, route_judge,
   cold compile < ~5.5 min), then check; submit only if the check beats 2.0606x.
2. Weakest ratios on the board run: d32 sl 1.589, v16 sl 1.923, d16 sl 1.976: experts (tk2f/twm) are moe's lane; ask moe.
3. okc check can exceed a 600 s tool timeout: run it in the background, poll `okc jobs`, read with `okc status <job> --json`.

Mains since the board best (v70), all H100-validated (bench test 13/13, replay_test + route_judge 0 fails), details in
NOTES s32-s44: v71 BND_SPLIT | v72 O_PF (d32/v16 sl) | v73 moe TK2F_L2H | v74 pr1_npf 0 at T=1 | v75 TUNE_BASE (H200-agreed
per-(T, sliding) defaults) | v76 moe TWM_L2H | v77 fused pooler+router GEMM at T=256 | v78-v80 g19 M=8 + T=8 candidates +
proj evict_last on all L2 prefetches | v81-v83 PDL edges in the routing glue | v84 proj g13o (T=1 o 12-bit, TUNE) | v85 proj
DEC4 | v86 proj DEC3 (12-bit decode 3 instr/pair) | v87 moe_hot at T=256 (TUNE_BASE hot 32; H100 v32 -14 sl / -11 full) |
v88 proj s36 pr8 bulk L2 prefetch after the wait (PR8_CFG npf 212; H100 ABBA 6 rounds v1 -1.9 sl / -1.6 full).


s19-s23 (v48-v55): generic prepare() tuner `TUNE` (layer.py): per (T, sliding/full) it times whole-layer graph replays of
option sets on synthetic inputs (shapes only) and keeps the fastest in state["tune"]; `_opt()` reads it at trace time.
Options: g19:<key>, lt:<key>, ltalt:<key>, g10alt/g13alt/g14alt:<key> (1-based `_ALTS` index), pr1, pr1_npf, pr1_rb,
pr1_pf, kv_pf. Every option was forced and passed scored + replay tests. Defaults = v45 (checked 1.9634x).

## Board
- **Best scored: 703d3a7e = v70 (909010f), 1.9954x (s32 14:00Z).** Previous: 371a17ee = v42 (cfd3bc5), 1.9558x, 11,566 us/step, overall #49** at 07:10Z 10-26 (#1 2.368x, #10 2.176x,
  #50 ~1.953x). Per shape (H200 us): d1 71.4/82.3, d16 145.1/223.9, d32 185.9/334.2, v1 120.5/132.4, v16 264.9/368.3,
  v32 368.0/544.3. Its check 9db0d4eb 1.9545x. (Old best e5512e00 1.9459 had moe's tk2f race; now superseded.)
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

## Where the time goes (judge check 58f0ef6f profile, v15, H200 us)
- decode 1 sliding (81): gemv10 qkv+pooler 30.6, tk2 13.9+8.5 (3 local experts: floor ~17), boundaries 8.4, o 6.4,
  unpool 4.4, router 4.3 (side; delays the pooler start), csl 4.2, norm2 3.8, routing 3.7, rmsnorm 1.5.
- decode 32 sliding (190): tk2 64.4+36.4 (9-11 experts: floor ~56; tk2 is ~83% of HBM peak on H100 but ~58% on
  H200: issue-bound, moe's lane), cuBLAS qkv 12.6 / o+unpool 26.5 / pooler 20.5 + 2 splitK reduces 4 (floor ~40;
  cuBLAS is ~at the practical limit, see dead ends), attention 12, boundaries 7.5, norm2 5.8, routing 4.1.
- verify 1 sliding (136): tk2 41.7+23.3, g12 GEMVs 46, router_fma 9.8 (issue-bound; delays pooler start ~8 us on
  H100), boundaries 8.8.
- verify 16/32 sliding (298/382): moe_tw 122+71 / 152+78, cuBLAS ~70-90, boundaries 9.5/16, router_split 11.
  Full layers add attention (_gd_full/_gw_verify 100-190 us, ~83% of KV floor; frozen lane).

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
