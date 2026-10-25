# STATE-proj (muse_bf16_proj) — short current picture; NOTES-proj.md is the long log

Updated 2026-10-25 ~21:45 UTC (session 3).

## Board
- **Best: submit 22109e72563f4cc3 = 42dac99, 1.2334x (CI 1.215-1.252), proj rank 379.** (#1 1.61x, #10 ~1.53x.)
- Previous: 202a2485 (134eda6, lt v9) 1.2111x; a55d6b55 (e71d015) 1.1757x.
- main = 42dac99 (+ notes).

## Per-shape H200 speedup vs SGLang (submit 22109e72; SGLang us in bench.py SGLANG_H200)
| load | qkv_gate | o | pooler | unpooler | router |
|---|---|---|---|---|---|
| decode 1 (m1) | 1.25 | 1.07 | 1.22 | 1.66 | 2.09 |
| decode 16 | 1.01 | 1.07 | 1.16 | 1.03 | 2.11 |
| decode 32 | 1.02 | 1.07 | 1.16 | 1.01 | 1.75 |
| verify 1 (m8) | 1.03 | 1.05 | 1.16 | 1.13 | 3.42 |
| verify 16 (m128) | 1.08 | 1.02 | 1.16 | 1.00 | 1.57 |
| verify 32 (m256) | 1.14 | 1.03 | 1.19 | 1.02 | 1.56 |
Floors: qkv 7.9, o 3.5, pooler 14.0, unpooler 1.75 (m<=8) / 14.1, router 1.8; m256: 9.8/4.5/17.4/17.4/2.7.
SGLang H200 us: qkv ~14 (m<=32) 17 (m128) 20 (m256); o ~7/7.2/9.2; pooler ~21.5/24.8/30.7; unpooler 7 (m<=8) 18.7/20/24.

## Code on main (submission.py is GENERATED: edit dev/sub_template.py, dev/*.cu, dev/lt_cfg.txt; run `python dev/build_sub.py`)
- Router: m<=8 true-FP32 dev/router.cu; m 9-32 hi/lo mma dev/gemv7.cu; m>32 one cuBLAS GEMM vs stacked [hi; lo].
- BF16 m=1: dev/gemv10.cu (grid-stride GEMV, fixed configs BF16_G10).
- BF16 m>=8 (8/16/32/128/256): **prepare() `_tune`** times, per (layer, m), as CUDA-graph replays after a 256 MB read
  flush (cold L2, like the judge), F.linear vs gemv12 (BF16_G12: m8 all, pooler m16) vs 5 cuBLASLt configs
  (BF16_LT from dev/lt_cfg.txt), keeps the fastest in st["pick"]; kernel_fn dispatches via `_run`. Prints "tune ..."
  lines (visible in the judge log: profiles/*.log). On the H200 lt wins every shape except unpooler m32 (F.linear).
- dev/lt.cu: cuBLASLt matmul with explicit algo config (algo 66 = sm90 kernel; tile, stages 35=auto, split-K,
  reduction, swizzle, custom option, cluster shape). Separate load_inline (`-lcublasLt`): if it fails, F.linear.
- Sweep tooling: dev/lte_lib.py (load_inline helpers: lt_enum = all 45,792 algo-66 configs; lt_desc decodes a blob),
  dev/lte.py `python dev/lte.py 16:qkv_gate 256:pooler ...` (screen all configs, 2 reps eager w/ flush, refine top 40
  in graphs; saves /tmp/pj/lte_<m>_<layer>.pt), dev/lt_gencfg.py (-> lt_cfg.txt, top 5 distinct per shape).
  Run with SUB=<abs path to submission.py> and cwd /tmp/pj (scripts expect /tmp/pj/lt.py: copy dev/lte_lib.py there).
- bench.py: `python bench.py test` (39 cases), `python bench.py bench [filter]`.

## Budget 2026-10-25: checks 12/20, scored 7/10 (lead: max 9 submits/day). Resets 00:00 UTC.

## What I learned (numbers in NOTES)
- A graph's kernel launch costs ~1.8-2.6 us over an empty graph (more blocks -> more). Reader slope 3.24 TB/s (H100).
- Read pattern: only high-occupancy grid-stride or block-contiguous (threads interleaved at 16 B) reads approach
  peak; per-warp contiguous streams spread over the buffer are ~10-15% slower.
- x staging traffic = blocks * (m*k_chunk*2): keep blocks fat (1024 threads) and few. Issue x BEFORE W (x requests
  queued behind a 17 MB W prologue took 5-8 us).
- float atomicAdd on smem = CAS loop. Cluster size 4 at 1 block/SM: only 30 clusters fit (2 waves for 32).
- H200 gains exceed H100 gains for these kernels (H100 +1% -> H200 +6% at qkv m8). Don't drop marginal H100 wins.
- gemv12 m8 qkv (S=4): end phase (slot sum, stcg partials, fence, counter, re-read) ~2.8 us; warps finish with a
  ~1.8 us spread inside a block. H200 gemv12 at m16: qkv 0.99, o 0.91, pooler 1.08, unpooler 0.95; m32 0.72-0.93.
- Dead ends this session: TMA bulk x staging (slower than cp.async), shared float atomics, 4-block clusters,
  R=2 row groups per warp (spills), gemv9 warp stream-K, CUDA-core math at m>=8, cuBLAS swap-AB.
- unpooler m8 on gemv12 with 32 blocks: H200 5.71 once, 7.04/7.27 twice (same code): few-SM configs are fragile.

## Next (ranked)
1. Remaining gap is mostly unpooler m16-256 (~1.0x, cuBLAS ~3.6 TB/s already: near practical) and qkv m16/m32 (1.01;
   floor 7.9 vs ~13.7): a native kernel that hits m1-gemv10 speed at m16/32 (gemv10 qkv m1 = 11.5 on H200).
2. Tuner: more candidates for weak shapes (qkv m16/32: sweep was noisy, 2-rep screen; try a 6-rep screen of the top
   500), and router m>32 (stacked hi/lo FP32-out GEMM) through cuBLASLt configs too (needs an FP32-out lt variant).
3. gemv12 split-K end phase (~2.8 us): DSMEM cluster reduction instead of global partials+counter.
4. Router m<=8 (5-8 us vs 1.8 floor).
