# STATE-proj (muse_bf16_proj) — short current picture; NOTES-proj.md is the long log

Updated 2026-10-25 ~20:20 UTC (session 1).

## Board
- **Best: submit 0b870721666d40c7 = main 087192f, 1.0699x (CI 1.054-1.086), rank 926.** #1 1.610x, ~#10 1.53x:
  the leaders must be near the floor on most shapes, so a real GEMV kernel is worth ~1.4x more.
- Overall kernel board: tech-cosmos rank 456 after this (was 815).

## Per-shape H200 (us), SGLang / ours (0b870721); floor in TASK.md
| load | qkv_gate | o | pooler | unpooler | router |
|---|---|---|---|---|---|
| decode 1 (m1) | 13.94/14.15 | 6.75/6.93 | 21.54/21.87 | 7.10/7.26 | 10.56/**5.38** |
| decode 16 | 14.37/14.08 | 6.82/6.92 | 21.50/21.68 | 18.56/18.85 | 14.66/**10.88** |
| decode 32 | 14.34/14.18 | 7.01/7.01 | 21.79/21.86 | 18.66/18.94 | 15.33/15.31 |
| verify 1 (m8) | 14.37/14.20 | 6.77/6.67 | 21.57/21.55 | 6.98/6.85 | 26.05/**7.58** |
| verify 16 (m128) | 17.33 | 7.62 | 25.28 | 20.06 | 16.26 |
| verify 32 (m256) | 19.58 | 9.76 | 30.85 | 23.94 | 17.54 |
Floors: qkv 7.9, o 3.5, pooler 14.0, unpooler 1.75 (m<=8) / 14.1, router 1.8; m256: 9.8/4.5/17.4/17.4/2.7.

## Code on main (submission.py is GENERATED: edit dev/sub_template.py + dev/*.cu, run `python dev/build_sub.py`)
- Router m<=16: native true-FP32 split-K kernel `dev/router.cu` (FP32 W, FP32 FMA, smem x, red.global + last-block
  finalize; workspace/counters self-zeroing). Configs (mp,kb,rw,nwarp): m1 (1,2048,1,8), m<=8 (8,512,2,8), m<=16 (16,512,2,8).
- Router m>16: SGLang's cuBLAS hi/lo. BF16 layers: F.linear (cuBLAS) = SGLang.
- bench.py: `python bench.py test` (9 judge tests + 30 scored shapes), `python bench.py bench [filter]` (graph, cold L2).
  `SUB=dev/cand.py` selects another file. dev/sweep.py, dev/rt.py: config sweeps.

## Budget 2026-10-25: checks used 2/20, scored 2/10.

## What I learned (H100 local, cold L2 = 512 MB read, graph replay minus empty graph)
- Pure streaming read (dev/rd.py, grid-stride 1056x256 thr, 4 x 16B loads in flight per thread): 8 MB 5.2 us,
  16.7 MB 8.2, 37.7 MB 14.5, 67 MB 23.6 -> ~3 us fixed + 3.2 TB/s. cuBLAS: qkv m1 18.2, pooler 27.8 (5.8 us fixed).
  So on the H100 the ceiling for BF16 m<=32 is ~20% below cuBLAS; on the H200 the leaders show it's more.
- My BF16 GEMV prototypes are NOT faster than cuBLAS yet (all in dev/):
  gemv.cu (1 block per stripe x K chunk, regs) ~cuBLAS at m1, bad at m32; gemv2/3 (TMA cp.async.bulk pipeline,
  stream-K) ~9 us fixed cost, per-SM ingest ~18 GB/s regardless of stages (!); gemv4 (warp-level stream-K, regs,
  x frags from global) ~cuBLAS+1 us at m1, terrible at m>=16 because x fragments come from L2 per tile (58 MB L2
  traffic for 17 MB of W at o m16). The split-K flush (atomics + threadfence + counter) costs ~2 us in gemv4.

## Next (ranked)
1. BF16 GEMV for m<=32 that beats cuBLAS: block-tiled (SP stripes x K chunk per block, many blocks/SM), x staged
   in smem ONCE per block (issue x loads FIRST), W via register-direct 16-B loads of the packed layout (dev/gemv4
   _pack layout), grid-stride-like address order, intra-block smem reduction, split-K only as needed. Start from
   the rd.py access pattern (it is the fastest reader measured) and add the math. Compare vs cuBLAS per shape.
2. Router m=32/128/256 (15-17.5 us vs floor ~2-2.7): one tensor-core hi/lo kernel with x frags shared by hi and lo
   tiles (interleave hi/lo in the packing), block-tiled as in 1. m=8/16 router: 7.6/10.9 vs 1.8 floor: also room.
3. m=128/256 (compute-heavier): wgmma kernel or better cuBLAS/cuBLASLt algo choice; DeepGEMM/FlashInfer in judge.
4. Handoff to layer agent: router kernel interface pj_router(W_fp32, x, out, ws, cnt, *cfg).
