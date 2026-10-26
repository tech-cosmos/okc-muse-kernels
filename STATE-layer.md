# STATE-layer (muse_decode_layer): short current picture; NOTES-layer.md is the long log

Updated 2026-10-26 ~01:20Z (session 7).

## Board
- **Best scored: 81e501e9 = v20 (164ea45), 1.8815x, 12,023 us/step (2026-10-26 00:11Z).** Board #1 2.285x (10-25 19:30Z).
- main = v22 code (3fab554) + FUSED_ROUTER_TOKENS option off (522da14: different submission.py bytes, same kernels).
  v21 (f57c3be, moe tk2f) check f72eff30 1.8834x (tie: v1 sliding 135.9 -> 123.9, d1 80.4 -> 77.8, rest noise).
  v22 (3fab554 = v21 + proj's bit-exact escape headers) check 511378a9: see NOTES (submitted if it beat 1.8815).
- v21 per shape, H200 check f72eff30 (ours, us sliding/full; SGLang in brackets): d1 77.8/85.1 (172/181), d16
  160.8/233.2 (270/433), d32 196.4/346.5 (261/610), v1 123.9/138.4 (260/273), v16 280.3/371.6 (531/722), v32
  367.2/542.9 (733/1089). Speedups: d1 2.21/2.13, d16 1.68/1.86, **d32 1.33/1.76**, v1 2.10/1.97, v16 1.89/1.94,
  v32 2.00/2.01. Judge noise: identical code moves up to 4% per shape; total ~+-1%.
- Floors (us): d1 40/44, d16 77/137, d32 85/208, v1 62/66, v16 127/189, v32 144/267.

## Code (muse_decode_layer/)
- `layer.py` = the layer. Inlined by **`python build.py` -> submission.py** (never edit submission.py by hand):
  `attn_src.py` = ~/attn submission.py @82866c6 (kernel_fn -> attn_kernel_fn, constexpr np2 fix, csl::decode patched
  to trigger PDL dependents); `moe_src.py` = r-moe 128b11a (= local 60d4a66; T<=32 = moe_tk2f one launch; names prefixed `_moe_`; T>32 = its kernel_fn (moe_twf: gemm13+gemm2 in one persistent launch); layer.py patches moe_norm2 to
  trigger dependents); `g10_src.cu` = proj gemv10 (r-proj d894f4e, m=1), `g12_src.cu` = proj gemv12 (r-proj e71d015,
  m=8), `g13_src.cu`/`g14_src.cu` = proj gemv13 (m=1)/gemv14 (m=8) over lossless 12-bit BF16 (r-proj 3f173eb, bit-exact escape headers; packs `_pack13/_pack14` in `_g12_prepare`, used for qkv+pooler (G13_CFG/G14_CFG) when the weight compresses, built in `_g10_lib`), `lt_src.cu` = proj cuBLASLt explicit-algo GEMM (LT_CFG fixed H200 picks, `_lt_prepare`); layer.py `_g10_lib` builds g10/g12 in one load_inline and text-patches them (PDL trigger, PDL-secondary launch).
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
0. Watch r-moe / r-proj Handoffs; re-copy + rebuild + test (+ tools/tk2f_stress.py, tools/esc_probe.py) + interleaved
   bench vs variants/sub_v22.py + check. Only submit on a check beating the board by more than noise.
1. Experts are 50-60% of every T>=16 shape (tk2f d32 101 of 217 us H100): moe's lane.
2. d32 sliding is the weakest shape (1.33x): critical path = qkv 19, attn ~26, o 9, boundary 4, pooler||router 38,
   tk2f 101, norm2 5, unpool 27, boundary 4 (H100 us). FUSED_ROUTER_TOKENS would win ~3-5 us at T>=16 with a GEMM
   config for [T,4608] FP32 out better than cuBLAS's heuristic (try proj lt_src with Ctype/Dtype 32F + a config sweep).
3. FP32-x port of proj's router2 (r-proj e605717) for T=8: ~2 us on v1 (router delays the pooler start).
4. Glue fusions (each ~1-3 us at d1/v1): rmsnorm into tk2f x staging, boundary into a GEMV epilogue (last-CTA).
5. Routing risk: diag check f5fdc570 once failed a timed rep's routing (3.18e-2 > 0.03). Not seen since.

## Dead ends (details in NOTES)
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
