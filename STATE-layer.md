# STATE-layer (muse_decode_layer): short current picture; NOTES-layer.md is the long log

Updated 2026-10-25 ~22:30Z (end of session 4: v10 scored 1.728x; main v14 checked 1.753x, submit after 00:00Z).

## Board
- **Best scored: 5fc7a807 = v10 (64bfdcf), 1.728x, 13,094 us/step.** Board #1 2.285x (10-25 19:30Z).
- **main = v14 (d531473): check 50820dee 1.753x (12,900 us). FIRST THING after 00:00Z 2026-10-26: submit main
  (`okc submit muse_decode_layer/submission.py --json`; code identical to the checked one).**
- Budget 2026-10-25: 9 of 9 submits used (cap 9, keep 1 spare), 14 of 20 checks used.
- v14 per shape, H200 check 50820dee (SGLang / ours, us): d1 170.2/81.2, d1f 181.6/92.3, d16 266.2/161.3,
  d16f 440.4/240.5, d32 263.9/193.6, d32f 623.6/335.8, v1 260.4/130.7, v1f 268.1/139.8, v16 522.9/321.7,
  v16f 697.5/404.0, v32 709.6/444.6, v32f 1068.2/630.4. Judge noise: same code moves up to 8% at d32/v32.
- Per load (geomean over loads; load = 46 sliding + 16 full): v10 d1 1.92, d16 1.74, **d32 1.51**, v1 1.87,
  v16 1.63, v32 1.61. d32 is the weakest load: experts 108 (tk2 gemm13 71 + gemm2 37) + cuBLAS GEMMs ~65.
- Floors (us): d1 40/44, d16 77/137, d32 85/208, v1 62/66, v16 127/189, v32 144/267.

## Code (muse_decode_layer/)
- `layer.py` = the layer. Inlined by **`python build.py` -> submission.py** (never edit submission.py by hand):
  `attn_src.py` = ~/attn submission.py @82866c6 (kernel_fn -> attn_kernel_fn, constexpr np2 fix, csl::decode patched
  to trigger PDL dependents); `moe_src.py` = r-moe 0f549b6 (names prefixed `_moe_`; layer.py patches moe_norm2 to
  trigger dependents); `g10_src.cu` = proj gemv10 (r-proj d894f4e, m=1), `g12_src.cu` = proj gemv12 (r-proj e71d015,
  m=8); layer.py `_g10_lib` builds both in one load_inline and text-patches them (PDL trigger, PDL-secondary launch).
  Re-copy: `git -C ~/r-moe show <c>:muse_moe_w4a16/submission.py > moe_src.py` (check the 68-int route meta
  matches `_moe_bucket`, and that the patch anchors in layer.py/build.py still match: asserts fire at import).
- Pipeline: qkv `_linear` (gemv10 T=1, gemv12 T=8, else cuBLAS) -> attn -> o `_linear` -> `_boundary` -> side stream:
  router + `_routing` || pooler `_linear` -> `_rmsnorm_gain` -> experts (`_moe_kernel_fn` = tk2 at T<=32, own
  route/moe_gemm/moe_big path at T>32) -> unpool `_linear` (1024-row slice at T<=8) -> `_boundary`.
  PDL: GEMV_PDL (gemv triggers; glue after it launches with PDL at T in PDL_TOKENS) and GEMV_PDL_SECONDARY
  {(tokens, sliding): keys} = GEMVs launched as PDL secondaries.
- bench.py `test` / `bench [file] [filter]` / `prof`; tools/timeline.py <file> "<load> <layer>" (per-kernel
  start/end of one replay). variants/ (gitignored) sub_v9/v11/v12/v13/v14 builds for A/B; regenerate from git.

## Where the time goes (judge check 50820dee profile + H100 timeline, v14)
- decode 1 sliding (81 us H200): gemv10 qkv+pooler ~30, experts tk2 24+14 + norm2 4.5, o 4.4, unpool 6.6,
  boundaries 2x~3.5, attention 4.2, rmsnorm 1.6; router 4.2 + routing 3.7 on the side stream (router delays the
  pooler start ~3.5 us on H100: gemv10 pooler needs whole SMs). Remaining gap to floor 40 = mostly HBM-bound GEMVs
  and experts (at ~70% BW) + ~15 us of glue.
- decode 32 sliding (194): experts tk2 108 (gemm13 71, gemm2 37), cuBLAS GEMMs ~61 + 2 splitK reduces 4, glue ~20.
- verify 16/32: experts moe_big ~300 / ~300-450, cuBLAS GEMMs ~70-90. Full layers: attention 100-300 (frozen lane).

## Next (ranked)
0. Submit main after 00:00Z (see Board). Then watch r-moe / r-proj: moe is on big-M (TMA) and the >8-token decode
   experts (d16/d32: the weakest load); proj on gemv12 m16/m32 (cuBLAS still wins there). Re-copy + rebuild + test
   + interleaved bench vs variants/sub_v14.py + check on every handoff.
1. d32 (weakest load, 1.51): GEMMs at T=16/32 are cuBLAS at 61 us vs ~40 floor (proj's lane, item 2 in their STATE);
   experts 108 vs ~56 floor (moe's lane).
2. More PDL secondaries: none left at T=1/8 on our GEMVs (qkv is graph start; pooler would block the router). For
   moe's tk2 kernels (1 CTA/SM, 83 KB smem) a secondary cannot co-reside; ask moe if they want PDL-friendly W prefetch.
3. Glue fusions (each ~1-3 us at d1/v1): boundary into the o / unpool GEMV epilogue (last-CTA pattern), rmsnorm
   into moe x staging (needs patched moe CUDA), router last-CTA top-8 (drops the routing launch from the side path).
4. Routing risk: diag check f5fdc570 once failed a timed rep's routing (3.18e-2 > 0.03). Not seen since (10 runs).

## Dead ends (details in NOTES)
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
