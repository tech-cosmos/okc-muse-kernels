# STATE-layer (muse_decode_layer): short current picture; NOTES-layer.md is the long log

Updated 2026-10-25 ~21:00Z (end of session 2).

## Board
- **Best scored: 12744959 = dbe255a (v7), 1.555x, 14,550 us/step, layer rank 220.** Board #1 2.285x (9,900 us),
  #20 2.03x. Budget 2026-10-25: 6 of 6 submits, 9 of 20 checks used (all spent on submits today: none left).
- **QUEUED: main (v8, c33532e + comment-only change) checked fc3e60d0 = 1.581x [1.557, ...] > 1.555 board.
  SUBMIT IT FIRST after 00:00 UTC 2026-10-26** (`okc submit muse_decode_layer/submission.py --json`).
- v8 per shape, H200 (SGLang / ours, us): d1 172.7/105.5, d1f 183.2/113.7, d16 269.8/183.7, d16f 444.0/260.4,
  d32 268.2/218.3, d32f 631.3/369.6, v1 267.7/157.9, v1f 281.0/169.3, v16 545.3/341.8, v16f 720.1/427.3,
  v32 735.6/469.5, v32f 1074.9/631.9. (Judge noise: SGLang's own times move +-5% between runs.)
  Per load ~: d1 1.63, d16 1.55, d32 1.42, v1 1.69, v16 1.62, v32 1.61.
- Floors (us): d1 40/44, d16 77/137, d32 85/208, v1 62/66, v16 127/189, v32 144/267. Weights: sliding 46, full 16.

## Code (muse_decode_layer/)
- `layer.py` = the layer; `attn_src.py` = ~/attn submission.py @82866c6; `moe_src.py` = r-moe 6a1cda0
  (`git show <commit>:muse_moe_w4a16/submission.py > moe_src.py`; check the route meta size, now 68 ints, matches
  `_moe_bucket`). **`python build.py` writes submission.py** (inlines both; attention kernel_fn -> attn_kernel_fn +
  constexpr np2 fix; moe names prefixed `_moe_`). Never edit submission.py by hand.
- Pipeline: qkv F.linear -> attn kernels -> o F.linear -> `_boundary` -> side stream: router + `_routing`
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
1. After 00:00 UTC: submit v8 (see above).
2. Watch ~/COORD.md Handoffs / r-moe and r-proj commits: re-copy moe_src.py on every moe speedup (T=1..256 all
   use it now), rebuild, `bench.py test`, local bench vs variants/, check. Proj: their gemv7 BF16 m=1 is ~3-5%
   faster than cuBLAS on H200 (qkv 13.3 vs 14.0, pooler 20.5 vs 21.1): worth taking when they post it, also
   their router hi/lo kernel for m 9-32 (but my router is off the critical path, so low value).
3. Small layer fusions (each ~1-3 us): pre-expert rmsnorm into moe gemm13's x staging for T<=32 (needs a
   patched copy of moe CUDA: x*r*g rounded to BF16 = identical to now); remove splitK reduces once GEMMs are ours;
   boundary into the unpool GEMM epilogue at T<=8 (last-CTA pattern) if we own that GEMV.
4. Routing risk: diag check f5fdc570 once failed a timed rep's routing (3.18e-2 > 0.03). Not seen since (6 runs).

## Dead ends (details in NOTES)
- Multi-CTA boundary (atomics + spin barrier); one-pass boundary (4 sums): both slower.
- PDL on glue after cuBLAS: no gain. Side-stream moe bucketing at T>=128: +29..67 us on the judge.
- L2 prefetch of the next weight on a side stream (load kernel or cp.async.bulk.prefetch.L2): GEMM speeds up but
  attention is delayed/slowed by the same amount; a side-stream fork at graph START costs ~9 us.
- GEMV with PDL preloading W before griddepcontrol.wait: overlaps but the producer slows equally (zero-sum).
- torch.mm(out_dtype=float32) instead of F.linear: same speed.
