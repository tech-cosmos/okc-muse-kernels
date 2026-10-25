# STATE-layer (muse_decode_layer): short current picture; NOTES-layer.md is the long log

Updated 2026-10-25 ~21:40Z (end of session 3: probes only, no judge runs; main unchanged = v8).

## Board
- **Best scored: a7650536 = v8 (main @4c21a41), 1.576x, 14,353 us/step, layer rank 207-209.** Board #1 2.285x.
- **main = v9 (fc6c0ea)**: v8 + proj gemv10 at T=1. check 9261f31b 1.595x; submit dd2b45c6 1.566x (noise at T>=32,
  see NOTES s3; d1 gain held: 97-98/105 vs 104-105.5/113.5 us). Submit the NEXT real change on top of v9.
- Budget 2026-10-25: 8 of 9 submits (cap raised to 9 by the lead), 10 of 20 checks used.
- v9 per shape, H200 check 9261f31b (SGLang / ours, us): d1 170.5/97.0, d1f 180.3/105.0, d16 262.1/177.8,
  d16f 434.6/257.1, d32 255.7/217.8, d32f 617.0/365.7, v1 260.5/153.5, v1f 269.6/162.4, v16 514.3/328.6,
  v16f 673.5/415.5, v32 740.2/454.9, v32f 1102.4/623.5. Judge noise: same code moves up to 8% at d32/v32.
- Floors (us): d1 40/44, d16 77/137, d32 85/208, v1 62/66, v16 127/189, v32 144/267. Weights: sliding 46, full 16.

## Code (muse_decode_layer/)
- `layer.py` = the layer; `g10_src.cu` = proj's gemv10 CUDA (r-proj d894f4e; re-copy `_G10_SRC` from their
  submission.py when they improve it; configs in layer.py G10_CFG); `attn_src.py` = ~/attn submission.py @82866c6; `moe_src.py` = r-moe 6a1cda0
  (`git show <commit>:muse_moe_w4a16/submission.py > moe_src.py`; check the route meta size, now 68 ints, matches
  `_moe_bucket`). **`python build.py` writes submission.py** (inlines both; attention kernel_fn -> attn_kernel_fn +
  constexpr np2 fix; moe names prefixed `_moe_`). Never edit submission.py by hand.
- Pipeline: qkv `_linear` (gemv10 at T=1, else cuBLAS) -> attn kernels -> o F.linear -> `_boundary` -> side stream: router + `_routing`
  (+ moe bucketing <= 16 tokens) || pooler F.linear -> `_rmsnorm_gain` -> moe experts (all T now; Marlin code kept,
  MARLIN_TOKENS = ()) -> unpooler F.linear (1024-row slice <= 8 tokens) -> `_boundary`.
- bench.py `test` / `bench [file] [filter]` / `prof`; tools/timeline.py <file> "<load> <layer>" (per-kernel
  start/end of one replay). variants/sub_v7.py = v7 build for A/B.

## Where the time goes (judge, v8)
- decode 1 (105 us): cuBLAS GEMMs ~50 (qkv+pooler 33, o 7, unpool 5, 3 splitK reduces 5), experts 35 + norm 3.7
  (~BW floor for 3 experts), boundary 2x3.8, attention 4.3, rmsnorm 1.5. Router/routing hidden on the side stream.
- decode 32 sliding (218): experts 131 (gemm13 86, gemm2 46), GEMMs ~65, attention 10, glue ~20.
- verify 16/32: experts (moe_big) ~290/~290-450, GEMMs ~70-90. Full layers: attention 100-300 (frozen lane).
- Layer-side glue is at the launch floor (empty kernel ~2.4 us). The remaining big levers are other lanes:
  BF16 GEMMs at 55-65% of HBM peak (proj), experts at decode 16/32 + big-M (moe).

## Next (ranked)
0. Biggest lever in the whole layer = experts at T>=16 (2.3-3x their byte floor, see NOTES s3); that is moe's lane:
   re-copy moe_src.py the moment they post a faster moe_big / decode kernel. If moe doesn't take the big-M redesign
   (Request in COORD), it's worth doing here: wgmma with dequantized A in registers, B = x via TMA, more rows per warp.
1. Watch ~/COORD.md Handoffs / r-moe and r-proj commits: re-copy moe_src.py on every moe speedup (T=1..256 all
   use it), re-copy g10_src.cu when proj improves gemv10 or extends it to m=8 (verify 1 = T=8; gemv7 loses there),
   rebuild, `bench.py test`, `bench.py bench` vs variants/sub_v8.py (gitignored, regenerate from git), check.
2. gemv10 is ours at T=1 now, so its epilogue can take glue: unpool GEMV + boundary 2 (last-CTA), o GEMV +
   boundary 1; each saves a launch + gap (~2-3 us at d1).
3. Small layer fusions (each ~1-3 us): pre-expert rmsnorm into moe gemm13's x staging for T<=32 (needs a
   patched copy of moe CUDA: x*r*g rounded to BF16 = identical to now); remove splitK reduces once GEMMs are ours;
   boundary into the unpool GEMM epilogue at T<=8 (last-CTA pattern) if we own that GEMV.
4. Routing risk: diag check f5fdc570 once failed a timed rep's routing (3.18e-2 > 0.03). Not seen since (6 runs).

## Dead ends (details in NOTES)
- s3: proj gemv7 as layer GEMVs (ties cuBLAS m1, loses m2-8); pooler before router fork; router grid sweeps (noise);
  L2 bulk prefetch inside boundary CTAs (-0.7 us) or concurrent on a side stream (+3 us); moe_big config sweep.
- Multi-CTA boundary (atomics + spin barrier); one-pass boundary (4 sums): both slower.
- PDL on glue after cuBLAS: no gain. Side-stream moe bucketing at T>=128: +29..67 us on the judge.
- L2 prefetch of the next weight on a side stream (load kernel or cp.async.bulk.prefetch.L2): GEMM speeds up but
  attention is delayed/slowed by the same amount; a side-stream fork at graph START costs ~9 us.
- GEMV with PDL preloading W before griddepcontrol.wait: overlaps but the producer slows equally (zero-sum).
- torch.mm(out_dtype=float32) instead of F.linear: same speed.
