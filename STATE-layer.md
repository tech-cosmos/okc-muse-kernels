# STATE-layer (muse_decode_layer): short current picture; NOTES-layer.md is the long log

Updated 2026-10-26 ~03:55Z (end of session 9).

## Board
- **Best scored: e5512e00 = v33 (bb6fb77), 1.9459x, 11,624 us/step (2026-10-26 ~04:00Z), overall #52.** Per shape:
  d1 73.5/82.0, d16 149.9/238.6, d32 187.7/323.8, v1 118.8/131.5, v16 282.8/363.1, v32 361.3/533.9.
  Board #1 2.360x, #10 2.167x (10-26 04:00Z). Previous best be766193 (v26) 1.927x.
- main = v33 (bb6fb77) = v29 + moe e8d47eb (SA) + proj router3 at T=128 (v32) + KV_PF (qkv gemv13 L2-prefetches the
  d1 sliding KV window). Checks: 50cb4a5b (v32) 1.930x; **6aaff5a2 (v33) 1.9377x, 11,674 us**: d1 73.2/85.2,
  d16 148.1/229.6, d32 177.3/309.3, v1 120.2/131.7, v16 264.6/348.6, v32 357.0/537.2 (H200 us sliding/full).
- Budget 2026-10-26: submits 81e501e9, ca8f833a, dae762e4, be766193, 3fca3609 e5512e00 (6 of 9); checks 8 used (~04:00Z).
- SGLang per shape (H200 us sliding/full, moves 2-3% between runs): d1 171/182, d16 257/436, d32 260/618, v1 257/268,
  v16 502/680, v32 681/1046. Judge noise: identical code moves up to 4% per shape; total ~+-1%.
- Scoring: geomean over 6 loads, each = 46 sliding + 16 full layers: 1 us on d1 sliding = +0.16% total, v1 +0.10%,
  d16 +0.07%, d32 +0.05%, v16 +0.04%, v32 +0.03%.
- Floors (us): d1 40/44, d16 77/137, d32 85/208, v1 62/66, v16 127/189, v32 144/267.
- Routing robustness (lead 01:45Z): DONE s9: tools/route_judge.py (judge-like graph replay on fresh inputs) 0/900
  fails; our margins = the FP8-query reference's own (attention's FP8 queries dominate; glue adds nothing). Details
  NOTES s9. Rerun it after any change upstream of the router: `python tools/route_judge.py <file> 25 11,12,13 [filter]`.

## Code (muse_decode_layer/)
- `layer.py` = the layer. Inlined by **`python build.py` -> submission.py** (never edit submission.py by hand):
  `attn_src.py` = ~/attn submission.py @82866c6 (kernel_fn -> attn_kernel_fn, constexpr np2 fix, csl::decode patched
  to trigger PDL dependents); `moe_src.py` = r-moe d899a76 (TK2F 204; T<=32 = moe_tk2f one launch; names prefixed `_moe_`; T>32 = its kernel_fn (moe_twf: gemm13+gemm2 in one persistent launch); layer.py patches moe_norm2 to
  trigger dependents); `g10_src.cu` = proj gemv10 (r-proj d894f4e, m=1), `g12_src.cu` = proj gemv12 (r-proj e71d015,
  m=8), `g13_src.cu`/`g14_src.cu` = proj gemv13 (m=1)/gemv14 (m=8) over lossless 12-bit BF16 (r-proj 3f173eb, bit-exact escape headers; packs `_pack13/_pack14` in `_g12_prepare`, used for qkv+pooler (G13_CFG/G14_CFG) when the weight compresses, built in `_g10_lib`), `lt_src.cu` = proj cuBLASLt explicit-algo GEMM (LT_CFG fixed H200 picks, `_lt_prepare`); layer.py `_g10_lib` builds g10/g12 in one load_inline and text-patches them (PDL trigger, PDL-secondary launch).
  Re-copy: `git -C ~/r-moe show <c>:muse_moe_w4a16/submission.py > moe_src.py` (check the 68-int route meta
  matches `_moe_bucket`, and that the patch anchors in layer.py/build.py still match: asserts fire at import).
- Router: T=1 `_router_fma` (Triton FP32), T=8 `r3_router` (CUDA FP32, in `_g10_lib`), T=128 proj `pj_router3` (`pr3_src.cu`,
  PR3_CFG; hi/lo one kernel, atomics), other T>8 `_router_split` (SGLang hi/lo BF16).
- KV_PF (T=1 sliding): `_linear(..., kv_pf=)` passes page_table[0, -2049:], k/v cache to the text-patched g13 (`_g10_lib`).
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
0. Watch r-moe / r-proj Handoffs (proj gemv18 = compressed m16/m32 in progress; moe tk2f T=16/32); re-copy + rebuild +
   test (+ tools/tk2f_stress.py, tools/n2u_eq.py, tools/route_judge.py) + interleaved A/B (/tmp-style: two subs in one
   process, 5 rounds; separate-process bench drifts +-1 us) + check. NOTE: the layer calls moe_tk2f_launch directly at
   T=1 (`_experts_n2u`, cfg + 100000 for SA) and reimplements moe_norm2 in `_N2U_SRC`: update both when moe changes.
1. Pooler + router (+ top-8) in ONE kernel at T=1 and T=8 (layer-owned, the best lever left in the glue): today the
   router's CTAs fill every SM's register file, so the pooler starts 3.4 us (d1) / 5 us (v1) after the boundary and
   the two run at ~2.5 TB/s instead of ~3.2 (H100). Plan: one grid, blockIdx role split (~110 pooler g13/g14 blocks +
   ~22 router2-style FP32 blocks, 1 block/SM, one wave), the last router block (atomic ticket) does the top-8 +
   weights (_routing) while the pooler streams. Estimate d1 -2..3, v1 -3..5 us (+0.5..1%).
2. d1 tail: n2u/boundary ~-1.4 us if the 8 MB unpool slice were in L2 (needs evict_first on moe's bulk copies +
   evict_last prefetch in an HBM-idle window): ask moe first.
3. Experts are 50-60% of every T>=16 shape (moe), cuBLAS GEMMs ~87 us vs ~61 BF16 floor at d32 (proj).
4. v1: tail tk2f -> norm2 4.7 -> g17 7 -> boundary (~14 us). n2u doesn't scale to 8 tokens (1 MB of y32 per block).

## Dead ends (details in NOTES)
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
