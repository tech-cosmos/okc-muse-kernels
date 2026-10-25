# STATE-layer (muse_decode_layer): short current picture; NOTES-layer.md is the long log

Updated 2026-10-25 ~23:30Z (end of session 5: main v17 checked 1.8045x (v15 1.818x); submit main after 00:00Z).

## Board
- **Best scored: 5fc7a807 = v10 (64bfdcf), 1.728x, 13,094 us/step.** Board #1 2.285x (10-25 19:30Z).
- **main = v17 (5cf795f): check 842b5e2c 1.8045x. v15 (3508986) check 58f0ef6f 1.818x, v16 check f8e06892 1.811x:
  all noise-equal (+-1%); v17 = v15 + cuBLASLt qkv m256 etc. (v32 +1-2%). FIRST THING after 00:00Z 2026-10-26:
  `okc submit muse_decode_layer/submission.py --json` (main = v17, code identical to check 842b5e2c).**
- Budget 2026-10-25: 9 of 9 submits used, 17 of 20 checks used.
- v15 per shape, H200 check 58f0ef6f (SGLang / ours, us): d1 171.0/80.7, d1f 182.8/92.5, d16 269.5/162.9,
  d16f 431.9/234.1, d32 264.8/190.3, d32f 617.0/343.3, v1 267.3/135.7, v1f 273.8/144.6, v16 523.6/297.5,
  v16f 703.6/394.0, v32 718.8/382.3, v32f 1060.3/553.1. Judge per-shape noise: identical code moves up to 4%.
- Per shape speedup (v15): d1 2.12/1.98, d16 1.65/1.85, **d32 1.39/1.80**, v1 1.97/1.89, v16 1.76/1.79, v32 1.88/1.92.
- Floors (us): d1 40/44, d16 77/137, d32 85/208, v1 62/66, v16 127/189, v32 144/267.

## Code (muse_decode_layer/)
- `layer.py` = the layer. Inlined by **`python build.py` -> submission.py** (never edit submission.py by hand):
  `attn_src.py` = ~/attn submission.py @82866c6 (kernel_fn -> attn_kernel_fn, constexpr np2 fix, csl::decode patched
  to trigger PDL dependents); `moe_src.py` = r-moe c39a176 (names prefixed `_moe_`; T>32 = its kernel_fn (moe_tw); layer.py patches moe_norm2 to
  trigger dependents); `g10_src.cu` = proj gemv10 (r-proj d894f4e, m=1), `g12_src.cu` = proj gemv12 (r-proj e71d015,
  m=8), `lt_src.cu` = proj cuBLASLt explicit-algo GEMM (LT_CFG fixed H200 picks, `_lt_prepare`); layer.py `_g10_lib` builds g10/g12 in one load_inline and text-patches them (PDL trigger, PDL-secondary launch).
  Re-copy: `git -C ~/r-moe show <c>:muse_moe_w4a16/submission.py > moe_src.py` (check the 68-int route meta
  matches `_moe_bucket`, and that the patch anchors in layer.py/build.py still match: asserts fire at import).
- Pipeline: qkv `_linear` (gemv10 T=1, gemv12 T=8, cuBLASLt pick at T=128/256, else cuBLAS) -> attn -> o `_linear` -> `_boundary` -> side stream:
  router + `_routing` || pooler `_linear` -> `_rmsnorm_gain` -> experts (`_moe_kernel_fn`: tk2 at T<=32, route+gather+
  moe_tw at T>32) -> unpool `_linear` (1024-row slice at T<=8) -> `_boundary`.
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
0. Submit main after 00:00Z (see Board). Then watch r-moe / r-proj Handoffs: moe next = 256-thread moe_tw RT=4
   (big-M), gather fusion, PDL; proj = qkv m16/32 native kernel, router m<=8. Re-copy + rebuild + test +
   interleaved bench vs variants/sub_v17.py + check on every handoff. Judge noise: ~+-1% total, up to 4% per shape:
   only submit on a check that beats the board by more than that, and don't chase <1% local wins with checks.
1. Experts are 50-60% of every T>=16 shape: biggest lever, moe's lane (tk2 at d16/d32 issue-bound on H200).
2. proj's CUDA FP32 router at T=8 (pj_router, r-proj dev/router.cu): -2 us router at v1 (H100 9.47 vs 11.4), which
   is on v1's critical path (pooler waits for the router's SMs). ~1% of v1. Not done.
3. Glue fusions (each ~1-3 us at d1/v1): rmsnorm into moe tk2 x staging (patch moe CUDA), boundary into a GEMV
   epilogue (last-CTA), router last-CTA top-8 (drops _routing).
4. Routing risk: diag check f5fdc570 once failed a timed rep's routing (3.18e-2 > 0.03). Not seen since (13 runs).

## Dead ends (details in NOTES)
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
