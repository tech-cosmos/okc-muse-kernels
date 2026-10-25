# STATE-proj (muse_bf16_proj) — short current picture; NOTES-proj.md is the long log

Updated 2026-10-25 ~21:45 UTC (session 3).

## Board
- **Best: submit a55d6b55a2604db3 = e71d015, 1.1757x (CI 1.158-1.193).** (#1 1.61x, #2-#11 1.53-1.56x.)
- Previous: 3c377a1a (ec649e1, m1 gemv10) 1.1739x; fd765502 (25e165b) 1.1251x.

## Per-shape H200 (us), SGLang / ours (check 1c6ae4b6 = e71d015); floor in TASK.md
| load | qkv_gate | o | pooler | unpooler | router |
|---|---|---|---|---|---|
| decode 1 (m1) | 14.05/**11.50** | 7.01/**6.55** | 21.78/**18.23** | 7.04/**4.54** | 10.54/**5.32** |
| decode 16 | 13.68/14.04 | 6.53/6.55 | 21.44/21.54 | 18.74/18.68 | 14.85/**7.38** |
| decode 32 | 14.29/14.18 | 6.75/6.68 | 21.58/21.37 | 19.04/18.72 | 15.15/**8.57** |
| verify 1 (m8) | 14.14/**13.36** | 6.85/6.74 | 21.26/**19.06** | 6.96/**5.71** | 26.30/**7.91** |
| verify 16 (m128) | ~17.0 cuBLAS | ~7.4 | ~25 | ~20 | 16.4/**10.3** |
| verify 32 (m256) | ~19.4 cuBLAS | ~9.2 | ~31 | ~24 | 17.5/**11.1** |
Floors: qkv 7.9, o 3.5, pooler 14.0, unpooler 1.75 (m<=8) / 14.1, router 1.8; m256: 9.8/4.5/17.4/17.4/2.7.
Still at cuBLAS speed: all BF16 shapes at m = 16, 32, 128, 256 (4 of 6 loads). That's where the remaining gap is.

## Code on main (submission.py is GENERATED: edit dev/sub_template.py + dev/*.cu, run `python dev/build_sub.py`)
- Router: m<=8 true-FP32 dev/router.cu; m 9-32 hi/lo mma dev/gemv7.cu; m>32 one cuBLAS GEMM vs stacked [hi; lo].
- BF16 m=1: dev/gemv10.cu (grid-stride over chunks of whole rows, W row-major in place, fat 1024-thread blocks
  share one x copy in smem; FP32 FMA). Configs BF16_G10.
- BF16 m=8 (and pooler m=16): dev/gemv12.cu (mma.sync m16n8k16, x = A, W = B repacked by _pack12 into 512-B atoms
  [ks][rg][atom][lane]; block = (k-chunk, NG 8-row groups) x WG warps each; x chunk cp.async'd into smem first, in XP
  pieces through a ring of XR buffers (pooler m16: S=1, full K, ring 8/2); per-warp smem slots summed; S>1: FP32
  partials in st["ws12"] + counter st["cnt12"], last block of a rg writes BF16). Configs BF16_G12
  (m, kcu, wg, pf, bk, xp, xr).
- Other m: F.linear (cuBLAS) = SGLang.
- bench.py: `python bench.py test` (39 cases), `python bench.py bench [filter]`. Sweeps: dev/g10.py, dev/g12.py
  ("M,KCU,WG,PF,XP,XR;..." bk...; skips invalid configs), dev/g11.py (dead variant). Phase timers: /tmp/pj/dbg12.py style (globaltimer per block).

## Budget 2026-10-25: checks 8/20, scored 5/10 (CLAUDE.md: max 6 submits/day). Resets 00:00 UTC.

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
1. gemv12 split-K end phase: release/acquire atomics instead of fence+atomic; or red.global into ws + one counter;
   or no split K where n/8 divides 128 (o, pooler m8 already S=1). Also slot sum cost (WG-deep).
2. m16/m32 (decode 16/32 = 2 loads): gemv12 loses (m32 badly). Per 512 B of W a lane does TT*2 LDS.128 + TT*2 mma.
   Ideas: wgmma (m64nNk16 with W as A from smem via TMA, x as B: N = m tokens = 16/32) so x is read from smem by the
   tensor core directly and W streams via TMA; or keep x A-fragments in registers across atoms (k-major loop inside a
   warp for a fixed k slice over many rgs: x frags loaded once per k slice, W atoms of all rgs at that k).
3. m128/256 (verify 16/32): cuBLAS ~20-30% over floor; would need a wgmma/TMA kernel (or DeepGEMM in the judge).
4. Router m<=8 (5.3-7.9 us vs 1.8 floor): FP32 weights 8 MB; a gemv10-style FP32 kernel might give ~4.
