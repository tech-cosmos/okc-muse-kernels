# STATE-proj (muse_bf16_proj) — short current picture; NOTES-proj.md is the long log

Updated 2026-10-25 ~21:10 UTC (session 2).

## Board
- **Best: submit fd765502babc4c69 = 25e165b, 1.1251x (CI 1.108-1.142), rank 831.** #1 1.61x, ~#10 1.53x.
- main = 955bbc6 (= 25e165b + BF16 m=1 on gemv7): check c62f7f1d 1.1235x, same within noise (m1 shapes +3-6%).

## Per-shape H200 (us), SGLang / ours (check c62f7f1d); floor in TASK.md
| load | qkv_gate | o | pooler | unpooler | router |
|---|---|---|---|---|---|
| decode 1 (m1) | 13.97/13.29 | 6.69/6.66 | 21.10/20.47 | 6.82/6.39 | 10.56/**5.17** |
| decode 16 | 14.02/14.03 | 7.10/7.18 | 21.55/21.50 | 18.35/18.49 | 14.53/**7.37** |
| decode 32 | 14.35/14.72 | 6.88/6.93 | 21.76/21.89 | 18.56/18.71 | 15.30/**8.91** |
| verify 1 (m8) | 14.14/14.37 | 6.85/6.88 | 21.31/21.43 | 6.94/7.06 | 26.11/**8.05** |
| verify 16 (m128) | 16.96/17.20 | 7.57/7.55 | 24.54/25.19 | 19.87/19.75 | 16.42/**10.32** |
| verify 32 (m256) | 19.46/19.38 | 9.38/9.30 | 30.66/30.94 | 24.08/23.80 | 17.50/**11.18** |
Floors: qkv 7.9, o 3.5, pooler 14.0, unpooler 1.75 (m<=8) / 14.1, router 1.8; m256: 9.8/4.5/17.4/17.4/2.7.
The BF16 shapes are ~85% of our time and still at cuBLAS speed: that is where the remaining ~1.4x is.

## Code on main (submission.py is GENERATED: edit dev/sub_template.py + dev/*.cu, run `python dev/build_sub.py`)
- Router m<=8: true-FP32 split-K kernel dev/router.cu. m 9-32: hi/lo mma kernel dev/gemv7.cu (NA=2, cluster DSMEM
  split K), configs ROUTER_HL. m>32: one cuBLAS GEMM vs stacked [hi; lo] (FP32 out) + add.
- BF16 m=1: dev/gemv7.cu configs BF16_GEMV; other m: F.linear (cuBLAS) = SGLang.
- bench.py: `python bench.py test` (39 cases), `python bench.py bench [filter]` (graph, cold L2; SGLang H200 times
  filled in -> prints a projected geomean). dev/g5.py (KER=5/6/7 sweeps, env DEEP/WIDE/ILV), dev/g8.py (gemv8).

## Budget 2026-10-25: checks used 4/20, scored 3/10.

## What I learned (see NOTES for numbers)
- H100 pure grid-stride read (dev/rd.py): 37.7 MB 14.85 us, 67 MB 23.8 us. cuBLAS m1: 18.2 / 28.0. Every GEMV
  structure I built (gemv5-8: register or cp.async-ring W, cluster/DSMEM or atomic split-K, one-wave or many-block
  grids, sequential-sweep layout, 32-128 KB/SM in flight) lands at cuBLAS +-5% for m1 and is worse for m>=16.
- H100 -> H200 ratios carried over 1:1 for the m1 kernel (no hidden H200 headroom for this design).
- m>=16 with swap-AB + x in smem: x traffic = m/RB of W traffic; need RB >= 128-256 rows per block, x pipelined.
- Clusters: grid must be <= cudaOccupancyMaxActiveClusters * cs or it runs 2 waves (cs 8/16 are tight).

## Next (ranked)
1. BISECT from the fast reader: take rd.py's grid-stride kernel (1056x256 thr, 4x16 B loads/thread in flight,
   W read in plain row-major order) and add m=1 math on CUDA cores (x row in smem as FP32/BF16, FMA, warp-shuffle
   reduce, atomics per row chunk). If it stays near read speed, grow it (m<=8 with packed B, then mma); if adding
   the math slows it, that's the culprit to profile. Target qkv m1 ~15 us H100 (~11 on H200).
2. m16/32: block = 128-256 rows (each warp 2-4 stripes sharing B frags), x chunks pipelined in smem; or try
   cuBLASLt algo search / DeepGEMM / FlashInfer GEMMs (in the judge image only) for m=16..256.
3. Router m<=8 FP32 (5.2-8 us vs 1.8 floor) and m128/256 (10-11 us vs 2.2-2.7): native hi/lo with RB=256 blocks.
4. Handoff (posted in COORD): router hi/lo kernel for layer.
