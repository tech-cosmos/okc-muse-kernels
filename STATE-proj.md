# STATE-proj (muse_bf16_proj) — short current picture; NOTES-proj.md is the long log

Updated 2026-10-26 ~03:30 UTC (session 9).

## Board
- **Best: submit 71145e34fe424844 = a1f0af7, 1.307x (3,695.6 us), proj rank 196** (router3 at router m9-128; its check
  a478daea gave 1.3137). (#1 1.62x, #10 ~1.54x.) main = a566ba9 (+ m256 router tuner: H200 picks cuBLAS, check 1a978122 tie).
- Before: 6d0fc4d4 (405439e) 1.2742x; d8160f04 (e605717) 1.2636x; f9cd875d (21bec6b) 1.2482x.
- 12-bit header format is now bit-exact (human 23:30Z): header stores true/scale, kernels subtract 2^-15, round-trip
  checked in _compress (fallback per layer); `python bench.py esc` = adversarial escape test. Run it after any 12-bit change.
- Budget 10-26 used: checks 5/20, scored 4/10 (cap 9).
- H200 unpooler-1k now (check 1be6c8f5): m1 3.75 (g17), m8 5.99 (g17).

## Router H200 now (check a478daea, a1f0af7): m1 3.79, m8 4.72 (router2), m16 5.55, m32 6.36, m128 9.00 (router3),
## m256 11.18 (cuBLAS splitK GEMM 6.4 + splitKreduce 2.0 + hi+lo add 1.6). SGLang: 10.5/26.4/15.0/15.5/16.8/17.7.
## router3 (dev/router3.cu, sweep dev/r3.py): split-K mma GEMM, hi+lo into one accumulator, partials -> float4 atomics
## into out zeroed by a tiny PDL primary kernel. m256: red volume (2M floats ~4 us) + mma.sync compute: only ties cuBLAS
## (prepare() picks per GPU since a566ba9: ROUTER_R3_TUNE).
## Per-shape H200 (check 7ba50b0a, 21bec6b): m1 qkv 11.2 o 6.3 pool 15.8 unp 4.6 router 5.1; m8 qkv 13.3 o 6.8 pool 16.6
## unp 6.6 router 8.1; m16 qkv 13.6 pool 19.1 unp 17.9; m32 qkv 13.8 pool 18.9 unp 18.9 (m>=16 = cuBLASLt, uncompressed).
## Older table (submit 22109e72, before compression):
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
- Router: m<=8 true-FP32 dev/router2.cu (block = 2 whole rows, x in regs, no split K; dev/r2.py sweeps); m 9-32 hi/lo mma dev/gemv7.cu; m>32 one cuBLAS GEMM vs stacked [hi; lo].
- BF16 m=1: tuner picks gemv13 (compressed; qkv, pooler) or gemv10 (o, unpooler-1k).
- BF16 m=8: tuner incl. gemv14 (compressed; wins qkv, pooler on H200), m16 pooler also gemv15 (not picked on H200).
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

## gemv16 (session 6, NOTES has the breakdown)
- W = mma A operand, x via ldmatrix, x staged by TMA bulk multicast to mc=2 blocks sharing the k-chunk, S=1 path writes
  outputs directly, S>1 pushes partials into the owner's smem (one cluster barrier). BF16_G16 in dev/sub_template.py.
- H100: unpooler m16 23.0 vs cuBLAS 25.5, qkv m16 16.9 vs 18.1, pooler m16 25.2 vs 27.5. H200: ~par with lt (gains smaller).
- Remaining cost over a bare reader (19.8 us at unpooler m16, H100): x staging ~1.5-2 us, math ~1.3 us, end ~0.5 us.

## Lossless 12-bit weights (session 5): the big lever
- m=1 was already AT a bare reader's speed: only fewer bytes help. _compress in dev/sub_template.py: sign|mantissa byte
  + 4-bit exponent code (window 16 below max exp), escapes (~1/row) fixed from a per-row header. 25% fewer bytes.
- gemv13 (m=1, CUDA cores) and gemv14 (m=8, gemv12 + decode) are in; tuner picks per shape. Not yet: m16/m32 (needs a
  better kernel than gemv12's structure), m128/m256 (cuBLASLt; would need own wgmma + decode), router (FP32: exponent
  compression gives 28/32 bits only).

## Next (ranked)
-3. **Lane since 10-26 01:45Z: the layer's GEMMs at T=16/32** (cuBLAS there: d32 qkv 12.6, o+unpool 26.5, pooler 20.5
   + 2 splitK reduces, H200). Candidate = dev/gemv18.cu (session 9, NOTES has the timeline lessons): TMA-ring streamer
   over 12-bit weights, stream-K. Its GEMM part beats cuBLAS on H100 at every m16/m32 shape (warm x: pool m16 24.2 vs
   27.9, qkv m16 15.8 vs 17.2, unp m32 25.3 vs 26.0) but the in-kernel completion (fence/counter/last pass) adds
   3.5-5 us. Do: completion outside (PDL convert kernel; in the layer fused into the next op: hand off to layer under
   Requests), x pieces / PDL W prefetch, then BF16_G18 in the tuner + one check for H200 numbers.
-2. Router m256 (11.2 vs floor 2.7) and m128 (9.0): needs less split-K red volume AND less L2->SM traffic: x by TMA
   multicast to the n-tile blocks of a cluster of 2 (gemv16 has the machinery), or wgmma (compute ~3.5 us on mma.sync).
-1. gemv16 on the H200 only ties lt: cut x staging further (the critical path is x latency at kernel start: W cannot
   start computing before x lands; ideas: start computing atoms whose x rows came first with per-row mbarriers; FP32->
   nothing: x must stay BF16) and the math cost (2 row groups per warp sharing the ldmatrix x fragments). m32: needs
   S=2 x mc=2 (clusters of 4, <=120 blocks) or a smaller-x scheme.
0. Compressed m16/m32 kernel that beats cuBLASLt (decode 16/32 loads: qkv 1.0x, unpooler 1.0x, pooler 1.12-1.16x):
   bytes -25% => ~-3.5 us pooler/unpooler, -2 us qkv per shape if the kernel reaches reader speed. Idea: gemv13-style
   grid-stride over whole rows (no split K: k=8192 rows) with tensor cores (mma m16n8k16, W as A = 16 rows, x as B),
   x chunk in smem; or split-K with a cluster/DSMEM reduction.
   Also o m1/unp1k m1 (k <= 4096: gemv13 fixed costs lose to gemv10).
1. Remaining gap is mostly unpooler m16-256 (~1.0x, cuBLAS ~3.6 TB/s already: near practical) and qkv m16/m32 (1.01;
   floor 7.9 vs ~13.7): a native kernel that hits m1-gemv10 speed at m16/32 (gemv10 qkv m1 = 11.5 on H200).
2. Tuner: more candidates for weak shapes (qkv m16/32: sweep was noisy, 2-rep screen; try a 6-rep screen of the top
   500), and router m>32 (stacked hi/lo FP32-out GEMM) through cuBLASLt configs too (needs an FP32-out lt variant).
3. gemv12 split-K end phase (~2.8 us): DSMEM cluster reduction instead of global partials+counter.
4. Router m<=8 (5-8 us vs 1.8 floor).
