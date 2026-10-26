# NOTES-layer (long log, newest at bottom)

## Session 1 (2026-10-25 19:17Z-)
- submit 53d9d188 @224f7fd (baseline modules): 1.0005x [0.985, 1.016], 22,608.9 us; 13/13. First board row.
- check 7e3e63a1 @367fbbf (baseline + ~/attn kernels): FAILED tests 4 and 8: `Unsupported function referenced:
  next_power_of_2` in _attn_body's PF path (full layers): SGLang's import leaves triton.next_power_of_2 a plain function.
  Fixed in build.py with our own triton.runtime.jit.constexpr_function.
- v2 (2a21d76): own glue. H100 microbench (graph, cold L2): one tiny kernel 2.4 us; boundary T=1 5.8, T=256 11.1;
  router: cuBLAS FP32 F.linear T=1 8.1 / T=8 23.6 (!), Triton FMA split-K BT=next_pow2(T) BE 2 BK 1024 KS 8: 5.9 / 11.0;
  hi/lo tl.dot T=16..256: 7.2-12.1; routing kernel 5.5; pre-expert norm 3.7.
  FMA router with BT=8 at T=1 was 25 us (3-D broadcast product wastes 8x): size BT to T.
- multi-CTA boundary (grid NC x T, atomics + spin row barrier, NC 2-16): 6.6-8.3 us vs single-CTA 16 warps 5.0 (T=1) / 5.3 (T=16): DEAD.
- check e4801e13 @2a21d76 (v2): 1.206x [1.188, 1.224]. submit d2f1498b @same: **1.2086x, 18,716 us**.
  v2 judge per shape (sglang/ours us): d1 172.2/143.3, d1f 182.1/150.0, d16 262.9/234.1, d16f 433.9/309.5, d32 262.5/225.0,
  d32f 618.5/381.2, v1 276.0/216.0, v1f 283.4/223.3, v16 535.9/502.4, v16f 678.0/554.7, v32 712.9/676.4, v32f 1074.5/858.8.
- diag check f5fdc570 (NEVER submit; prepare-time profile of the Marlin expert path on the judge): at T<=8 SGLang's
  fused_post_expert_norm_weighted_sum runs a torch fallback = ~14 elementwise/reduce kernels (the unexplained
  at::native kernels in every small-T profile); at T>=32 one Triton kernel. moe_align: 2.7 us (T=1) .. 7.5 (T=256).
  That check FAILED a timed repetition's routing (decode 1 sliding rep 44: "dropped expert 38 ... 3.18e-02 above its
  9th best (> 0.03)"). Local stress (tools/route_stress2.py, 3 router seeds x 40 inputs): 0 failures, worst pick gap
  ours 0.004-0.012 vs the FP8-query reference's own 0.004-0.005. Looks like FP8-query noise (SGLang has it too) but
  it is a real risk on every run: if it recurs, look at the score error upstream of the router.
- v3 (665a04e): own _post_expert_wsum (all 8 slots as one 2-D tile, 16 warps): H100 4.8 us T<=8, 5.2 T=32, 10.4 T=256
  (serial-slot version 8.8 us); used up to 16 tokens. check 281e3b43: 1.307x. submit ebf817c1: **1.3122x, 17,239 us**.
  v3 per shape (sglang/ours): d1 171.9/117.6, d1f 182.2/125.2, d16 260.4/196.6, d16f 441.6/283.3, d32 263.4/228.6,
  d32f 618.6/384.5, v1 267.8/179.1, v1f 278.9/189.7, v16 534.9/499.1, v16f 699.1/582.3, v32 719.2/680.5, v32f 1081.5/865.6.
- v4 (f819050): moe agent's experts (moe_src.py = r-moe bb49efd, inlined by build.py with every top-level name
  prefixed _moe_ via tokenize; local ids with -1 passed as ids, start 0, count 32) at every T except 32 (Marlin there:
  their board had bs32 0.97x). Local H100 full bench now meaningful (no fallback): geomean 1.223 vs SGLang-H200 numbers.
  check 63a88d7f: 1.452x [1.430, 1.474]; submit 7d829b64: **1.4415x, 15,692 us**.
  v4 per shape (sglang/ours): d1 169.5/106.2, d1f 179.4/113.5, d16 262.0/182.5, d16f 436.0/262.5, d32 256.9/219.7,
  d32f 619.7/383.0, v1 267.3/164.6, v1f 272.7/170.7, v16 529.9/393.4, v16f 726.2/494.4, v32 714.8/577.5, v32f 1065.1/739.5.
- Local timeline (tools/timeline.py, H100 decode 1 sliding, v4): qkv cuBLAS 20.9 + splitK reduce 1.7 (38 MB at ~1.8
  TB/s), csl 4.8, o GEMM 10.6 (17 MB!), boundary 4.3, [side: router_fma 5.6, routing 16.7 (slow only because it
  shares the GPU with the pooler; off the critical path)], pooler 27.2 + 2.0 reduce, prenorm 2.0, moe_route 1.9,
  gemm13 26.7, gemm2 15.3, moe_norm 3.8, unpool 6.2 + 1.7, boundary 3.7 = 137 us.
  routing num_warps 1/2/4 in isolation: 5.0/5.8/5.2 us (kept 2; it is off the path).
- v5 (b8b71f2): moe route launch on the side stream after routing (H100 137 -> 132) + moe at T=32: check c17d1581
  1.430x: decode/verify 1 -2.6/-7.5 us, decode 32 moe 222.0 vs Marlin 219.7 (keep Marlin), verify 16/32 sliding
  +29/+40 and full +15/+67 (local H100: no difference, 600 vs 600 us).
- v6 (5a95d4a): Marlin at 32, side-stream bucket only <= 16 tokens: check c4fe1669 1.460x, verify back to 391/578.
  So side-stream moe_route at T=128/256 HURTS on the H200 (judge) though not on the H100. submit 1a7c4997:
  **1.4586x, 15,508 us** (5th submit of 2026-10-25; 1 left today).
  v6 per shape (sglang/ours): d1 171.9/104.0, d1f 180.0/110.4, d16 258.6/178.5, d16f 431.9/255.2, d32 255.6/219.0,
  d32f 606.1/378.4, v1 262.8/157.1, v1f 281.3/174.1, v16 518.3/391.5, v16f 690.9/471.9, v32 705.3/578.3, v32f 1044.5/737.8.

## Session 2 (2026-10-25 20:24Z-)
- v7 (dbe255a): moe_src.py = r-moe db182ea (big-M kernel moe_big for M>32; route meta grew 66 -> 68 ints, so
  `_moe_bucket` allocates 68). Local H100: verify 16 399->351 / 572->506, verify 32 603->486 / 871->742 us.
  check 440361a9: 1.564x [1.541, 1.588]. submit 12744959: **1.555x, 14,550 us, layer rank 220** (6th submit of 10-25).
  v7 per shape (sglang/ours): d1 173.9/103.7, d1f 182.5/110.0, d16 259.7/177.3, d16f 444.8/262.8, d32 261.0/225.0,
  d32f 614.1/368.1, v1 272.1/164.3, v1f 281.8/171.2, v16 525.0/336.9, v16f 700.1/424.2, v32 699.5/448.3, v32f 1071.7/623.9.
  Per load: d1 1.67, d16 1.54, d32 1.34 (weakest; Marlin ~125 of 225 at sliding), v1 1.65, v16 1.59, v32 1.61.
- DEAD: L2 prefetch of the o weight on a side stream (tools/l2pf.py, l2pf2.py). Standalone, a GEMM after its weight
  is L2-resident is much faster (o 13.9 -> 5.8 us H100). In the layer: (a) Triton evict_last load kernel forked after
  qkv: o GEMM 8.9 -> 5.9 but attention starts 7 us later (prefetch CTAs hold the SMs): +4.5 us net; (b) bulk
  cp.async.bulk.prefetch.L2 from 64 one-warp CTAs: o 6.7, attention +2 us later and slower: +1.5 us net;
  (c) any fork at the very START of the graph (side stream root node) delays the first main-stream kernel: +9 us.
- DEAD: GEMV with PDL that loads its W tile before griddepcontrol.wait (tools/pdl_gemv.py): in a graph it does start
  during the producer, but the producer (latency-bound load chain) slows by as much as the overlap gains (HBM queueing):
  zero-sum (23.4 vs 23.7 us). Also the plain Triton GEMV is 11 us vs cuBLAS 9.7 for o at m1.
- DEAD: torch.mm(out_dtype=float32) for the GEMMs: same time as F.linear (tools/mm_variants.py).
- DEAD: one-pass boundary (both norms from 4 simultaneous row sums): 4.6/4.3 vs 4.3/3.9 us at T=1, mixed at T=256.
- Glue at decode 1 is at the launch floor: an empty kernel is ~2.4 us, boundary 3.8-4.3. Remaining layer-side lever
  is launch count (14 kernels on the critical path at T=1, 3 of them cuBLAS splitK reduces).
- v8 (c33532e): moe_src.py = r-moe 6a1cda0 (cheaper dequant), moe kernels at T=32 too (MARLIN_TOKENS = ()).
  Local H100 1-5 us better than v7. check fc3e60d0: **1.581x** (v7 check 1.564x); decode 32 sliding 218.3 on moe vs
  225.0 Marlin (440361a9), so Marlin is no longer used anywhere. Not submitted: 6/6 submits used on 2026-10-25.
  QUEUED for after 00:00 UTC. v8 per shape in STATE-layer.md.

## Session 3 (2026-10-25 20:44Z-21:40Z): no judge runs (no candidate beat noise); probes only
- Budget correction: okc shows 6/10 scored used for 10-25 but CLAUDE.md caps submits at 6/day, so v8 still waits
  for 00:00 UTC. 11 checks left today, none spent.
- DEAD: proj's gemv7 (r-proj 955bbc6) as the layer's BF16 GEMMs (tools/g7bench.py, H100 graph+cold L2, incl. launch):
  m1 qkv 20.8 vs cuBLAS(+splitK reduce) 20.8, o 12.2/11.8, pooler 30.5/31.2, unpool 11.4/12.0; m2/m8 g7 is 10-25%
  SLOWER. So owning the GEMVs buys no time by itself; it only pays if glue gets fused into them (~2-3 us each).
- DEAD: pooler before the router fork (event fork): the router's 1024 CTAs then wait behind the pooler and routing
  becomes critical (moe gemm13 starts 2.8 us later). Router grid sweeps (RPLAN be,bk,ks,warps at T<=8) all within
  local noise (same config 192 vs 199 us on verify 1: GPU shared); the 3.5-4 us pooler start delay at d1 stays.
- DEAD: L2 bulk prefetch (cp.async.bulk.prefetch.L2 via tl.inline_asm_elementwise, one 1 KB chunk per thread) in extra
  CTAs of boundary 1, of the pooler weight: bench d1 -0.7 us at 8 MB, 0 at 16/24 MB (boundary itself grows
  4.3 -> 6.2 us: issue cost). Gotcha: a 64-element tensor in a 512-thread program issues 8x (replicated layout).
  tools/pfprobe.py: a 4 us 1-CTA kernel + prefetch CTAs saves only ~2 us on the next 64 MB GEMV.
- DEAD: prefetching the NEXT weight concurrently on a side stream during a big GEMV (tools/pfprobe2.py): +3 us
  (fork + HBM contention). Every big kernel here is HBM-saturated enough that overlap is zero-sum or worse.
- Layer routing per scored shape (tools/route_hist.py, reference front, sliding, seeds 99/7/11): local routes /
  distinct local experts: d1 2-3/2-3, d16 23-24/6-10, d32 48-55/9-11 (top expert 13-20), v1 10-11/6-9,
  v16 163-181/17-18 (top 45-49), v32 318-353/18-21 (top 91-97). 27 MB per expert (w13 16+2, w2 8+1).
  => expert byte floors on H200: d1 ~17 us (ours ~39), d32 ~56 (ours 131), v16 ~100 (ours ~290), v32 ~112 (ours 335).
- moe_big variants on that routing (tools/big_sweep.py: NS,NW,MB in (8,8,2) default, (4,8,2), (4,16,2), (8,16,1),
  (4,8,3), (4,16,3), (8,4,4)): default best (v16 221.6 us route+13+2, v32 328.3; others +5-30%). Outputs identical.
  Rough budget of gemm13 at T=256 (H100 ~214 us): HBM 107, mma 52 (34 GFLOP incl. NG rounding 5->8 groups),
  smem B-fragment reads ~72 (each warp re-reads x for only 16 rows), dequant ~41: they add up instead of overlapping
  (one __syncthreads per 64-k chunk keeps the 8 warps in lockstep). Posted to moe as a Request.
- Lead raised the submit cap to 9/day (CLAUDE.md 05f0479). Submitted v8 (main @4c21a41, 13/13 local): a7650536
  **1.5760x [1.552, 1.600], 14,353 us, layer rank 207** (board best now).
- v9 (fc6c0ea): proj's gemv10 (r-proj d894f4e, grid-stride m=1 GEMV, W in place, FP32 accumulate; g10_src.cu inlined
  by build.py as a raw string, own load_inline) for qkv / o / pooler / unpool slice at T=1 only (layer `_linear`).
  H100 standalone (graph+cold L2 incl. launch, tools/g7bench.py): qkv 18.6 vs cuBLAS 20.9, o 11.5/11.8, pooler
  27.5/30.7, unpool 9.4/11.2. Layer H100 decode 1: 118.8/142.5 vs v8 123-128/148-151.
  check 9261f31b: 1.5948x [1.571, 1.619]; submit dd2b45c6: 1.5663x [1.543, 1.590] (board keeps 1.576).
  d1 held in both runs (H200 sliding/full: v8 104.3-105.5/113.4-113.7 -> v9 97.0-98.0/105.0-105.2), but the submit
  run had d32 236.1 (218-222 in the other 3 runs), v32 476.9/657.8 (450-469/624-632): shapes v9 doesn't touch, so
  judge/input variance at T>=32 (expert counts vary per input). Not resubmitting unchanged code (rule).

## Session 4 (2026-10-25 21:16-22:30Z)
- v10 (64bfdcf): moe_src.py = r-moe 0f549b6 (= their c7dce17 + the >8-token expert fix): TMA bulk-copy decode kernel
  moe_tk2 (route fused into gemm13, 3 launches) at T<=32 via `_moe_kernel_fn` directly (TK2_MAX_TOKENS; side-stream
  bucketing now only for 32 < T <= BUCKET_SIDE_MAX_TOKENS = never). H100: d1 114.9/134.1 (v9 118.8/142.7), d16
  230.5/332.7 (247.2/352.0), d32 221.0/448.8 (255.2/495.0), v1 180.1/192.1 (191.5/203.7).
  check ba297a3a **1.709x**; submit 5fc7a807 **1.728x, 13,094 us** (9th submit of 10-25: none left today).
  H200 per load (SGLang/ours load time ratio): d1 1.92, d16 1.74, d32 1.51, v1 1.87, v16 1.63, v32 1.61.
- v11 (a14b5d1): proj's gemv12 (r-proj e71d015 dev/gemv12.cu -> g12_src.cu, inlined; packed copies via `_pack12`
  in prepare; shared ws/cnt: all gemv12 calls are on the main stream) for qkv/o/pooler/unpool slice at T=8 (verify 1).
  H100 v1 174.6/187.4 (v10 180.1/192.1). check c99cf8af 1.711x: v1 136.8/147.3 vs 141.7/150.0 (rest noise).
- v12 (012e3af): PDL around our GEMVs. (a) gemv10/12 do griddepcontrol.launch_dependents at start; boundaries +
  rmsnorm launch with PDL at T in (1, 8) (they already preload weights/stream before gdc_wait). (b) the unpool
  gemv10 at T=1 launches as a PDL SECONDARY (cudaLaunchKernelEx + programmatic serialization; griddepcontrol.wait
  after its first W loads, before x) of moe_norm2, which is patched (layer.py, on _moe_CUDA_SRC) to trigger at start:
  W loads overlap norm2 (norm2 reads ~128 KB: no HBM contention, unlike s2's attention->o dead end).
  H100 d1 109.2/128.6 (v11 114.2/133.6), v1 -3.5. check f55079a3 1.723x: d1 84.7/92.5 vs 88.3/97.9.
- v13 (build.py patch): csl::decode (and its twin) trigger dependents at start; o gemv10 at T=1 a PDL secondary:
  d1 sliding -1..-3 us H100 (3 interleaved A/Bs; attention 4.5 -> 6.2 us but o finishes earlier).
- v14 (d531473): gemv12 PDL-secondary path (L2 prefetch of its W slice per CTA, then griddepcontrol.wait). Used for
  o at T=8 on sliding layers only: v1 sliding -1.9; verify-full o +1.3 and T=8 unpool tie (not used; table
  GEMV_PDL_SECONDARY). check 50820dee **1.753x, 12,900 us**: d1 81.2/92.3, v1 130.7/139.8. SUBMIT after 00:00Z.
- DEAD: router on few fat CTAs (persistent over K, BE 4-16) + pooler gemv10 on the remaining SMs (68-116 blocks):
  pooler + router are HBM-bound together (72 MB in ~26.5 us whichever split), best tie (moe start 28.4 vs 28.8 us
  after boundary). Background: gemv10 pooler = 64 regs x 1024 threads = the whole RF, so any router CTA blocks it.
- DEAD: _red_fin (decode full's last attention kernel) triggering dependents early: decode 1 full +2.5 us.
- H100 noise now: other agents' runs move the same file by up to 4 us at d1: A/B interleaved, >= 2 rounds.

## Session 5 (2026-10-25 22:11-23:10Z)
- v15 (3508986): moe_src.py = r-moe c39a176 (moe_tw: wgmma big-M experts, dequantized NVFP4 A in registers, TMA B
  from a gathered K-permuted x). T > 32 experts now go through `_moe_kernel_fn` (route, gather, tw gemm13/gemm2,
  norm). H100 v16 333.8/481.7 (v14 338.7/494.6), v32 417.8/674.7 (471.1/726.9).
  check 58f0ef6f **1.8176x [1.790, 1.845], 12,445 us**: d1 80.7/92.5, d16 162.9/234.1, d32 190.3/343.3,
  v1 135.7/144.6, v16 297.5/394.0, v32 382.3/553.1 (H200 us, sliding/full). tools/ckparse.py summarizes a check json.
- DEAD: moe_tw at decode T=16/32 instead of tk2 (TW_TOKENS): H100 d16 +14, d32 +29 us.
- DEAD: Triton split-K skinny GEMM with last-CTA reduce for T=16/32 (tools/skgemm.py): ties cuBLAS at best
  (qkv m16 18.0 vs 18.2, m32 19.7 vs 18.3 us H100 cold); ~2.1 TB/s is what both reach for 38 MB incl. launch.
- DEAD: moe_tk2 gemm13 as a PDL secondary of _rmsnorm_gain (TK2_PDL13, layer.py patches tk2: producer streams W and
  routing runs before the consumers' griddepcontrol.wait; launch via cudaLaunchKernelEx): works (tk2 starts 0.8 us
  after rmsnorm starts) but rmsnorm is only 1.7 us and the safe ordering (join side stream BEFORE rmsnorm, rmsnorm
  without PDL, because tk2 reads topk_ids before its wait) loses the rmsnorm's own PDL: H100 tie.
- DEAD: pooler first + router as its PDL secondary at T=8 (ROUTER_AFTER_POOLER), and the pooler on a priority -1
  stream (POOLER_HI_PRIORITY): the pooler starts 8 us earlier, but the router starves beside it (10 -> 34 us) and
  routing becomes critical: v1 +2..3 us. Pooler + router are HBM/SM-bound together (as s4).
  BUG FOUND on the way (eager tests, 2nd call): a main-stream tensor written on the side stream (p) reused the block
  _route had just freed (its split-K partials, same 64 KB): race. Rule: allocate side-written tensors BEFORE any
  main-stream temporaries are freed.
- DEAD: L2 prefetch of the FP32 router W (8 MB) from extra boundary-1 programs: router at T=8 is still 9.8 us (it is
  issue-bound: [8,2,1024] broadcast-multiply-reduce), v1 +0.9; d1 -0.8 on H100 but judge d1 -2% (noise level).
  Router configs (tools/router_sweep.py): current (BE 2, BK 1024, KS 8, 4 warps) is the best; tl.dot(ieee) FP32
  (tools/router_dot.py) slower (T=8 14.5 vs 11.4). proj's CUDA FP32 router (pj_router): T=8 9.47 vs ours 11.4 us,
  T=1 6.91 vs 5.87 (H100): could save ~2 us at v1 only; not integrated.
- v16 (99f31b3): proj's cuBLASLt with explicit algo configs (lt_src.cu = r-proj dev/lt.cu @f7b1c8b, own load_inline
  -lcublasLt, configs validated vs F.linear in prepare) at proj's H200 picks: check f8e06892 1.8112x (noise vs v15).
  pooler m256 pick is +11.5 us in the layer on H100 (beside the router_split): never used.
- v17 (5cf795f, main): lt only for pooler m128, qkv m128/m256: check 842b5e2c 1.8045x. Per-shape vs v15 check:
  v32 +1-2% (consistent with H100 qkv m256 -1.9 us), rest noise: identical code moved d1 2.118 -> 2.069, d32 1.392 ->
  1.341 between checks. Three near-identical codes: 1.818 / 1.811 / 1.805 => judge total noise ~+-1%.
- v18 (2e79b56): moe_src = r-moe 682dd87 (moe_tw RT=4, their judge 1.985 -> 2.033x): H100 layer worse (v16 337/482 vs
  332/478, v32 430/681 vs 417/675); check b55c49e3 1.7978x: v32 sliding 1.816 (v15/v17 1.880/1.895). Reverted to
  c39a176 (f2cb10b; submission.py byte-identical to v17 5cf795f). Told moe (COORD Requests).

## Session 6 (2026-10-25 23:05-~00:05Z)
- v19 (9de3707): moe_src.py = r-moe b42c782 (moe_twf: gemm13+gemm2 of the big-M path in ONE persistent launch,
  per-expert done flags; route now zeroes meta[68..99], so layer `_moe_bucket` allocates meta[100]). H100 v16
  333/480 -> 320/466, v32 419/676 -> 414/669 (2 interleaved rounds). check d1690865 **1.835x**: v16 297.5 -> 266.0,
  v32 382.3 -> 353.2 (sliding, H200 us).
- Judge profile v32 (T=256): moe_twf 204, moe_norm "45" = mostly its PDL wait for twf's tail. H100 timeline
  (tools/timeline.py): twf 117 -> 370 us, moe_norm starts at 332 (first twf CTA exits) and really takes ~10 us:
  ~38 us twf tail with SMs going idle (told moe).
- DEAD: twf at decode T=16/32 (layer TW_TOKENS path rewritten to twf): H100 d16 +22, d32 +25 us vs tk2.
- DEAD: L2 prefetch of the unpool W (first 16/24/40 MB, cp.async.bulk.prefetch.L2, 1/grid slice per CTA) issued by
  each twf producer when the work counter runs out (patched kernel + launcher signature): v16/v32 +1..5 us, monotone
  in PF size. The counter runs out near the start of the tail, so the prefetch contends with the last items' loads.
- v20 (164ea45, main): proj's gemv13 (m=1) / gemv14 (m=8) over lossless 12-bit BF16 weights (r-proj 21bec6b:
  g13_src.cu/g14_src.cu extracted from its submission, _compress/_nib/_pack13/_pack14 copied; GEMV_PDL trigger
  patched in; g14 shares g12_ws/g12_cnt on the main stream) for qkv_gate + pooler at T=1 and T=8; falls back to
  gemv10/12 per weight if _compress returns None (> 8 escapes in a row). H100 d1 106.7/128.4 -> 100.7/122.5,
  v1 171.1/185.3 -> 165.7/179.8. check 85410ccd **1.856x, 12,188 us**: d1 80.4 (83.2), v1 135.9 (138.2).
- DEAD: gemv14 for o (verify full; sliding o is a PDL secondary, kept g12) and unpool_slice at T=8, proj's cfgs
  (1,2,4,128,1) / (4,8,4,32,1): verify 1 +1.3 / +0.6 us H100.
- Not done (small): proj's pj_router at T=8 takes BF16 x; our router deliberately reads the FP32 boundary output
  (routing-precision risk), and the win is ~2 us at v1 only (~0.25% total).
- DEAD (tie): gemv13 for the T=1 unpool slice as a PDL secondary (patched: W prologue before griddepcontrol.wait,
  x after; cudaLaunchKernelEx): 4 interleaved rounds -0.2..-0.5 us (round 2 -2.6 was noise). Lesson: a stale
  cpp_sources declaration makes `_g10_lib` fall back to cuBLAS silently (tests still pass, d1 +24 us): always check
  the kernel names in `bench.py prof` after touching the GEMV build.
- 2026-10-26 00:11Z submit 81e501e9 (v20, 164ea45): **1.8815x, 12,023 us** (was 1.728x): d1 81.8/89.6, d16
  162.6/238.9, d32 191.7/339.5, v1 138.4/148.0, v16 279.5/354.9, v32 349.7/527.5. Budget 10-26: 1 submit, 0 checks.

## Session 7 (2026-10-26 00:12-~01:30Z)
- v21 (f57c3be; local d5b0dec before rebase): moe_src.py = r-moe 60d4a66 (= 128b11a after moe's rebase): moe_tk2f, decode
  gemm13+gemm2 in ONE launch (per-segment done counters st['cnt'], zeroed by moe_norm2; the layer's one moe state is
  stream-ordered, fine). moe's race fix (gemm2 waits for all gemm13 rows) included. tools/tk2f_stress.py: 600 random
  routings T=1/8/16/32, tk2f vs 3-launch tk2 bitwise equal. H100 (2 interleaved rounds) d1 101.4->97.5, d16 230->226,
  d32 221->217, v1 165->161 us. check f72eff30 **1.8834x, 12,011 us** (board v20 1.8815): d1 77.8/85.1, d16 160.8/233.2,
  d32 196.4/346.5, v1 123.9/138.4, v16 280.3/371.6, v32 367.2/542.9. v1 sliding -12 us vs v20's check; d16/d32 flat
  on the judge (H100 said -4); v16/v32 (unchanged code) +3% = noise. Total = tie.
- v22 (3fab554): proj 3f173eb's bit-exact 12-bit escape headers (human decision 10-25T23:30Z): g13_src.cu/g14_src.cu
  re-extracted verbatim (tools: extract _G13_SRC/_G14_SRC from r-proj submission), _compress = proj's (true/scale
  headers, empty slot 2^-15, prepare-time round-trip check -> None = gemv10/12 fallback). tools/esc_probe.py: real
  qkv/pooler W with injected 0 / -1e-20 / 3e-39 escapes, one-hot x at the escape col: g13 and g14 outputs bit-equal to
  W[:, col] (0 mismatches); random x rel err 1.6e-3 (BF16 output rounding). 13/13 tests.
- FUSED_ROUTER_TOKENS (522da14, OFF): SGLang's router above 8 tokens is BF16 h x BF16 hi/lo split (baseline.py), so
  pooler + router = ONE GEMM of h against [pooler; hi; lo] (4608 x 8192, FP32 out; routing sums hi+lo columns via
  strides, rmsnorm rounds the pooler part to BF16 first). No side stream. H100 tie: d32 pooler-to-rmsnorm 38.0 vs 38.3
  us (GEMM 30.8 + splitK reduce 2.1 + routing 3.4 + rmsnorm 1.7); bench d32 +1.5, v16 full +5, v32 -2..-10. Microbench
  (cold L2): fused = pooler alone + 3 us at every T (the router's 8 MB at HBM rate) and no orientation helps. To win it
  needs a better GEMM than cuBLAS's heuristic for [T, 4608] FP32 out (proj's lt with Ctype 32F + tuning).
- Not done: proj's router2 (r-proj e605717, m8 9.8 -> 6.5 us H100) takes BF16 x; SGLang's router input up to 8 tokens
  is FP32 (FusedPostNormResidualPreNorm writes it), so the layer needs an FP32-x variant; x in registers doubles
  (M=8, T=512: 128 regs for x alone): split tokens 2 x 4 per block, or x via L1 per j. ~2 us on v1 only.
- Ops lesson: `okc check` blocks ~20-40 min when the judge queue is busy: run it with run_in_background and a 1 h
  timeout; if the client dies, `okc jobs` + `okc status <job> --json` recover the result.
- check 511378a9 (v22, 3fab554) **1.9046x, 11,876 us**: d1 77.3/88.0, d16 148.7/228.7, d32 190.4/341.0, v1 126.0/138.8,
  v16 269.1/365.7, v32 365.5/540.7. -> submit ca8f833a (v22) **1.8924x, 11,953 us** (new best; was 1.8815): d1
  76.4/87.2, d16 147.6/237.7, d32 189.0/337.9, v1 123.1/143.6, v16 271.7/356.0, v32 353.0/529.1.
- DEAD: cuBLASLt heuristic search (64 algos, FP32 out) for the fused [T, 4608] pooler+router GEMM: nothing beats
  torch.mm's pick at T=16/32/128/256 (H100 cold L2; /tmp sweep: lt_setup/lt_info/lt_run over AlgoGetHeuristic).
  nvcc gotcha: a static std::vector<cublasLtMatmulHeuristicResult_t> crashes nvcc (internal error); use an array.
- v23 (358d87c): T=8 FP32 router = r3_router (layer.py _R3_SRC, built in _g10_lib; ROUTER3_CFG {8: (4, 2, 512)}):
  proj's router2 with FP32 x, M=4 tokens per block over blockIdx.y (x float4 in registers: 64 regs), 2 rows per
  block, 512 threads, no split K (part = logits, KS=1). H100 standalone m8 8.74 vs _router_fma 11.52 us (err 1e-7
  vs FP64); m1 tie (6.91 vs 7.10, not used). Layer H100 v1 162.1/177.3 -> 157.6/172.6 (3 rounds). Timeline: pooler
  (side g14) now starts 5 us after the router (was 8). check f2db2454 **1.9088x, 11,851 us**: v1 123.3/138.4.
- submit dae762e4 (v23, 358d87c): **1.8852x, 11,999 us** (below v22's 1.8924 = noise: SGLang's own reference times moved
  2-3% between runs, e.g. v32 sliding 692 vs 704). Board best stays ca8f833a (v22) 1.8924x. v23 stays on main (real
  H100 -4.6 us at v1). Don't resubmit v23 unchanged; it rides along with the next real change.

## Session 8 (2026-10-26 01:28Z-)
- Scoring arithmetic (bench.py WEIGHT, judge): score = geomean over the 6 loads of SGLang/ours, each load = 46 sliding +
  16 full layers. So 1 us saved = 46/load_us of that load: d1 sliding 1 us = +0.16% total, v1 sliding +0.10%, d16
  +0.07%, d32 +0.05%, v16 +0.04%, v32 +0.03%. Small-T glue microseconds are worth 3-5x big-T ones.
- moe_src.py = r-moe d899a76 (TK2F 204: 2 k chunks per ring stage; moe's H200 bs8-32 -3..4%). All build.py/layer.py patch
  anchors still matched. tools/tk2f_stress.py 0/600 bad. H100 layer: tk2f d1 36.0 -> 35.1 us.
- gemv17 (proj dev/gemv17.cu @405439e) for the unpool slice at T=1/8 (layer.py _G17_SRC, built in _g10_lib), with W
  loads moved before griddepcontrol.wait (PDL secondary) and a launch_dependents trigger. H100 in-layer config sweep
  (tools/g17sweep.py): all configs within 0.3 us of gemv10/gemv12 (PDL hides it). Kept (1,4,512)/(8,4,256) for
  proj's H200 standalone gain (-0.9/-0.5).
- n2u (layer.py _N2U_SRC, T=1): moe_norm2 fused into the unpool-slice GEMV. 128 blocks x 512 threads x R=8 rows: each
  block loads its W rows, then recomputes the token's norm2 from y32 (<= 8 x 16 KB from L2), rounds to BF16 (as
  moe_norm2's output), dots. Block 0 zeroes moe's tk2f counters (st["cnt"], norm2's other job). The layer calls
  moe_tk2f_launch itself (_experts_n2u). Bitwise equal to norm2 -> gemv17 (tools/n2u_eq.py). H100 d1 sliding
  96.7 -> 95.4, full 118.6 -> 116.8. R=4 +1.3 (256 blocks' redundant y32 reads), R=16 +1.5 (fewer W streams), PDL
  secondary of tk2f +2..3 (tk2f triggers at its start: n2u blocks squat on SMs).
- DEAD: second boundary as n2u's last-block epilogue (atomic counter, threadfence, L2 prefetch of its 5 vectors):
  H100 d1 +2.7 us vs the PDL Triton _boundary, which has its vectors loaded before the GEMV ends. Kept as
  N2U_BOUNDARY=False.
- RACE (unexplained, reverted): gemv17 as a PDL secondary of moe_norm2 at T=8 (GEMV_PDL_SECONDARY (8, False) =
  unpool_slice; H100 v1 -0.5..-0.7 us) fails test 8 (batch 1, ctx 1M, full, q_len 8) on repeats 2 and 3 (rel err 0.40),
  rep 1 passes; v26 passes all reps. Don't re-enable without finding it.
- Nondeterminism (benign): at d1 full, the attention output differs between identical runs in ~1 of 1024 elements on
  ~15% of seeds (1 BF16 ulp): ~/attn's FP32 atomic merge (red plan). Not a race.
- check d3c8c88d (v26, 45bb3ef) **1.9152x, 11,811 us**: d1 74.0/83.9, d16 150.4/228.1, d32 182.3/334.4, v1 123.7/132.0,
  v16 264.5/357.0, v32 348.6/518.4. -> submit be766193 **1.927x, 11.74 ms** (board best; was 1.8924): d1 75.9/83.3,
  d16 152.1/234.1, d32 194.1/339.6, v1 121.5/130.8, v16 283.0/365.2, v32 371.9/547.4.
- v29 (9199ea7): pooler at T=16/32 = 2-slab split-K strided torch.bmm (FP32 out, no copies: views of h and W), the
  slabs summed in _rmsnorm_gain (NS/SS, ROUND to BF16 = the pooler's dtype) instead of cuBLAS's splitK kernel + its
  reduce. Standalone H100 m16/m32 25.8/26.1 vs F.linear 28.0/28.1; layer d16/d32 -2.3 us on both layer kinds; 4
  slabs worse; T=128/256 noise (not used).
- DEAD: moe_norm2 as PDL secondary of tk2f (moe_set_pdl(6)): H100 d16/d32/v1 sliding +1.1..1.4 us. bmm split-K for o /
  unpool at T=16/32 standalone: unpool 25.4 -> 26.0 (S2), o 8.8 -> 9.5: cuBLAS non-split is better there.
- check 0092b908 (v29, 9199ea7) **1.9492x, 11,605 us**: d1 73.2/85.0, d16 153.2/225.5, d32 180.8/337.7, v1 119.8/138.7,
  v16 273.0/359.5, v32 369.1/543.2 (part of the gain over d3c8c88d is SGLang's reference moving +2-3%). Submitted.
- submit 3fca3609 (v29): **1.9139x, 11,819 us** (noise: d32 sliding 188.1 vs check 180.8, v1 sliding 128.5 vs 119.8).
  Board best stays be766193 (v26) 1.927x. v29 stays on main (-2.3 us d16/d32 on the H100); don't resubmit it unchanged.

## Session 9 (2026-10-26 02:46Z-)
- Routing robustness (lead 01:45Z): tools/route_judge.py = judge-like stress: per scored shape, capture kernel_fn once
  as a CUDA graph, then per fresh input copy it into the captured inputs, flush L2, replay, check_routing vs
  ref_front; also records the worst pick gap (limit 0.05) and worst "drop-lead" (a dropped reference-top-8 local
  expert's lead over the 9th; limit 0.03, 0.04 > 256 tokens), and the same for the FP8-query reference's own top 8.
  v29 (main), router seeds 11/12/13 x 12 shapes x 25 inputs = 900 replays: **0 failures**. Worst drop-lead ours /
  fp8ref per shape: d1 0.006/0.006, d16 0.021/0.010, d32 0.012/0.012, v1 0.019/0.019, v16 0.020/0.020, v32
  0.018/0.020. Ours tracks the FP8-query reference almost token for token (often identical maxima): the error
  upstream of the router is our attention's FP8 queries + FP8 P (allowed; FA3 does the same), not our glue
  (boundary/h/router precision add nothing visible). So there is no glue numerics to match. Residual risk: the FP8-
  query reference itself reaches drop-lead 0.020 at v16/v32 with 25 inputs, so a 0.03 tail on a very long run is
  possible for any FP8-query implementation; d1's f5fdc570 failure (0.0318 at T=1, where the worst seen here is
  0.006) is 5x outside this distribution: most likely a bug in that old diag build (Marlin-path profile, session 2),
  not reproducible on current code.
- v30 = moe e8d47eb (SA: arithmetic e4m3 scale decode in tk2f/twf when prepare() finds all block scales positive
  normal; the layer's own T=1 moe_tk2f_launch call passes cfg + 100000 when st["sa"]). H100: tie (moe saw the gain on
  H200 only: issue-bound there). tk2f_stress 0/600, n2u_eq: 3/40 diffs at d1 full also with identical toggles
  (known attention atomic-merge nondeterminism), sliding 0.
- v31/v32 = + proj router3 (pr3_src.cu = r-proj a1f0af7 dev/router3.cu, built in _g10_lib, PR3_CFG): router above 8
  tokens as one split-K mma kernel (hi+lo into one accumulator, float4 atomics into zeroed out). H100 layer: T=128
  v16 sliding/full 318/466 -> 313/460; T=16/32 +1..3 us (72 KB-smem CTAs beside the pooler bmm) -> v32 = T=128 only.
  route_judge v16 x 2 seeds: identical margins to v29.
- check 50cb4a5b (v32, 062c6c6) **1.930x, 11,721 us**: d1 74.5/83.6, d16 148.6/228.5, d32 190.5/342.5, v1 121.5/138.3,
  v16 269.9/365.1, v32 378.5/550.5.
- v33 (KV_PF): the qkv gemv13 at T=1 also L2-prefetches the sliding window's K/V rows: grid thread i < pfn reads
  page_table[0, mp - 2049 + i] after issuing its W prologue, prefetch.global.L2 of the K and V row (text patch of
  proj's g13 in _g10_lib; pooler call passes pfn = 0). Probe first: warming those rows (torch gather before the
  graph's qkv) cut csl 6.1-7.3 -> 3.8-4.3 us. With KV_PF: csl 4.1-5.3, qkv unchanged, o GEMV (PDL secondary of csl)
  ends 2.6 us earlier, timeline end 98.4 -> 95.8; A/B (5 rounds, one process) d1 sliding 97.17 -> 95.94. v1's vtk
  barely gains from warm KV (5.7 -> 5.4): not done there.
- DEAD (H100 A/B): ROUTER_L2_PF_TOKENS (1,) -0.4 (noise) and (8,) +0.9/-0.5; T=8 pooler g14 on 64 fat blocks (so the
  router could share SMs) +2..6 us. Measured but not built: unpool slice warm in L2 before n2u: n2u 7.1 -> 5.8 and the
  boundary after it 8.3 -> 6.9 (d1 ~-1.4 us), but keeping 8 MB in L2 through tk2f's 81 MB stream would need evict_first
  hints on moe's bulk copies + an evict_last prefetch: idea for later.
- v1 front (H100 timeline): r3_router's 256 x 512-thread blocks (64 regs) fill every SM's register file, so the
  pooler g14 (128 x 1024 threads, 61 regs: a whole SM each) starts only as router blocks drain: 5 us after the
  boundary. Fusing the router rows into the pooler kernel would fix it (~-3 us v1).
- check 6aaff5a2 (v33, bb6fb77) **1.9377x, 11,674 us**: d1 73.2/85.2, d16 148.1/229.6, d32 177.3/309.3, v1
  120.2/131.7, v16 264.6/348.6, v32 357.0/537.2. -> submit e5512e00 **1.9459x, 11,624 us** (new best; was be766193
  1.927x; overall #52): d1 73.5/82.0, d16 149.9/238.6, d32 187.7/323.8, v1 118.8/131.5, v16 282.8/363.1, v32
  361.3/533.9.
- Next big layer-owned item (STATE Next 1): pooler + router (+ top-8 by the last router block) in one kernel at T=1/8.

## Session 10 (2026-10-26 03:51Z-)
- v34 (f1beb08): moe_src.py = r-moe c33acae (tk2f race fix per lead 03:45Z; GS16 flags off). tests 13/13,
  tools/tk2f_stress.py 500 (now takes reps) 0/2000. Check 80eddcdc 1.9445x (tie with v33 6aaff5a2 1.9377 / scored
  1.9459). Submit 0b9801d2 1.9231x (noise-low): board best stays e5512e00 (v33, old race). Asked lead in COORD whether
  to withdraw e5512e00 (only rows with the fix would then be on the board at 1.927/1.923).
- v35 (08f61b2): proj gemv19 (g19_src.cu, r-proj dev/gemv19.cu @6996d0c; one load_inline with the rest, -DG19_ONLY2)
  for the pooler + full unpool at T=16/32 (FP32 out; _rmsnorm_gain ROUND / _boundary RB round to BF16 = SGLang's BF16
  GEMM output). Pooler g19 on all 132 SMs starves the side-stream _router_split (1 g19 block/SM, ~180 KB smem): routing
  ends late, H100 d16 +4 us; on 96 blocks -2. HANG lesson: the epoch barrier counter must be per grid size (old - old %
  G + G): changing the grid on a shared counter deadlocked; _g19 now keeps one counter per grid. H100 d16 -6, d32 -3.5.
  Check 5cf5247d 1.9238x: per-shape ratios show no H200 gain (d16 sliding 1.721 vs 1.709-1.745 history, d32 1.433).
  proj's H200 diag (NOTES-proj s10): every m32 loses; m16 pooler wins with AP8 (1024,8,3,xb1) 15.7 vs 19.0.
- v36 (4017c16): g19 only at T=16: pooler AP8 (16,1024,8,3,npre 3,xb1) on 108 blocks, first _boundary triggers its
  dependents at start (BOUNDARY_TRIG) so the pooler streams W during it; unpool m16 (1024,4,4). H100 interleaved d16
  219.3/324.2 (bmm) -> 213.2/316.5 (104 blocks), 112 214.9, 120 221.4 (router starved), 132 220.2. Without the unpool g19
  full +4.7. Check ee80a8f8 **1.9566x** (best check; d16 1.747/1.912). Submit 543b6ca9 1.9395x (noise). Judge per-shape
  speedups for identical code spread ~+-1.5% (SGLang's own times move 2-6%): single checks can't resolve <2% changes.
- New tools: tools/ab_flag.py <file> <filter> <rounds> "<stmt>"... (interleaved A/B of module flags in one process/state),
  tools/scored_test.py (check_implementation on scored shapes x seeds; the test list has no T=32), tools/replay_test.py
  (route_judge-style graph replay on fresh inputs, full output check).

## Session 11 (2026-10-26 04:52-05:35Z)
- v37 (d411cf4): proj pr1 (r-proj dev/pr1.cu@149d115 -> pr1_src.cu, inlined by build.py as _GPR1_SRC, built in
  `_g10_lib`): T=1 pooler (gemv13) + FP32 router + routing top-8 in ONE launch, PDL secondary of the first _boundary
  (TRIG at start); no side stream at T=1. H100 interleaved d1 94.7/117.4 -> 92.3/115.5 us; verify-1 unchanged.
  + moe_src = r-moe 2466fac (moe_twm, _moe_TWF_MIX 1; H100 v16 +3, v32 -4.5 us). Tests 13/13, route_judge d1 0/240
  (3 seeds x 40 x 2), replay_test 0/180.
  Check 7b02fb2c **1.947x** (score 11,618 us): d1 72.5/82.2, d16 145.0/221.1, d32 176.8/328.8, v1 126.0/131.5,
  v16 271.3/362.5, v32 358.2/534.3. = tie with v36 (1.9566 / 1.9395 runs): pr1 on H200 d1 -0.4/-1.5 us (noise);
  twm ~ twf (v16 sliding twm 145.8+44.3 vs twf 163+40; v32 224.5+44.8 vs 228-233+40-44). Not submitted.
- PR1_BND (off): the boundary fused into pr1 as a per-block prologue (each block recomputes h from the o GEMV's BF16 out
  + FP32 stream in registers, 2 block reductions; block 0 writes stream1/h). H100 d1 +2.0/+1.0 us; with s/alpha/beta/gain
  loaded before griddepcontrol.wait (BPre, pooler role) +1.2/+0.3. Cause (likely): 132 blocks x 144 KB redundant L2
  reads beside the W stream, and pr1 loses the boundary's 5 us of free W prefetch. Would need cluster/DSMEM sharing
  (each block of a 4-cluster computes 1/4 of h) to cut the redundant reads; est. gain <= 2-3 us d1.
- v38 (d664b23): proj pr8 (r-proj dev/pr8.cu@3be546b -> pr8_src.cu) at T=8: gemv14 pooler blocks each also take a
  router item, extra finisher block sums K slices + routes (PR1_TOKENS (1, 8), PR8_CFG {8: (8,8,4,128,1,0)},
  pr1_cnt int32[32]). H100 interleaved v1 157.7/172.6 -> 156.1/170.8 us. route_judge v1 0/120, replay T=1/8 0/60.
  Check 0602febf **1.936x** (SGLang's own times +2-4% this run): d1 74.2/84.5, d16 151.8/228.9, d32 179.8/336.5,
  v1 125.0/138.3 (v37 126.0/131.5; profile: pr8 22.2-22.7 us vs g14 pooler 16.6 + r3 5.6 + routing 3.2 overlapped;
  tk2f full 55.4 -> 57.2), v16 279.7/367.3, v32 361.0/529.5. Unresolved by one check; kept on main (H100 says win).
  Not submitted (needs > 1.9459). Budget 2026-10-26 at 05:55Z: checks 12/20, submits 8/9.

## Session 12 (2026-10-26 05:19-05:55Z)
- H200 idle budget (lead 05:00Z ask), posted under COORD Requests "layer -> all (s12 05:30Z)". Per-shape gap to the
  judge floor x score weight (%/us sliding|full: d1 .161|.056, d16 .072|.025, d32 .056|.020, v1 .096|.033,
  v16 .041|.014, v32 .031|.011): every load is 8.0-9.4% of score from its floor (sum ~50%). Top items: (1) moe big-M
  twm at v16/v32 (187/235 us vs ~124/140 bytes floor at 4.0 TB/s + drain) ~8%; (2) end-of-layer tail (norm2/n2u +
  unpool + boundary) ~4%; (3) boundary 1 + rmsnorm glue ~3.5%; (4) tk2f d16/d32 ~2.7%; (5) T<=8 GEMV ramps ~2.5%,
  T>=16 cuBLAS no-PDL gaps ~1.6%. H100 timelines of all 6 sliding shapes: /tmp/tl_all.py (3 cold replays, last one).
- check e6e0a00b (9762bf6, v39 = v38 + XN below) 1.926x: d1 74.3/83.6, d16 153.7/233.1, d32 190.2/341.5, v1 120.5/137.7,
  v16 284.9/374.2, v32 364.7/539.5 (SGLang d1 170.6: low this run). Noise-level; but see the XN lesson.
- DEAD UPF: moe_tk2f L2-prefetches the 8.4 MB unpool slice (cp.async.bulk.prefetch.L2, 1/grid each) for n2u/g17:
  from the producer after its last stage +0.6..1.5 us (competes with the tail stream); from tid 0 at CTA end: tie
  (n2u 7.2 -> 6.2 us at most: n2u is not W-bound; it's launch/latency after tk2f frees the SMs).
- DEAD XN: the pre-expert rmsnorm inside moe_tk2f (prologue: per-token row sums over all 8 consumer warps + gain to
  smem while the producer's first copies fly; gemm13 x loads scale + round to BF16). Output = baseline up to its own
  nondeterminism (pr1/pr8/tk2f sums: off vs off differs too, ~1 in 20 inputs). First version (in-consume barrier
  per segment): +2.7 d1 / +8.8 v1. Prologue version vs the same binary with XN off: -0.4..-0.8 us, BUT adding the
  XN code to the shared tk2f instantiation made the plain kernel 1-2 us slower at d16/d32 (H100 bench v38 vs v39,
  2 rounds each: d32 216.2/444.0 -> 217.3/444.9, d16 216.7/318.7 -> 218.3/319.8; regs 160 -> 168). As a separate
  template instantiation (plain kernel unchanged): XN +0.2..0.7 us at d1/v1. The rmsnorm launch is already hidden
  (PDL after pr1); its ~1 us is not on the critical path the way the timeline suggests. Reverted: main = v38 code.
  LESSON: A/B kernel-code changes against the ORIGINAL binary (variants/sub_vNN.py from git), not a flag inside the
  modified binary: shared instantiations change for every T.

## Session 13 (2026-10-26 05:55-07:10Z)
- v40 (e964348): moe_src = r-moe 258d49d (TK2F_DYN 84204: tk2f writes y32 in route-slot order pre-scaled, moe_norm3 =
  moe_norm2<true>; T=1 stays 204 for n2u). The layer's UNPOOL_PDL anchor ("__global__ void __launch_bounds__(512)
  moe_norm2(") still matches (now under a template line, patches both). H100 ab_flag 3 rounds: d16 217.2/319.8 ->
  215.9/319.0, d32 216.2/444.2 -> 215.4/443.3. POOLER_BMM[32] = 4 slabs: tie (dead).
- H100 d32 sliding timeline (v40): qkv splitK 16.9 + reduce 2.0, nds 13.8, o 9.0, boundary 4.0, router_split 6.9 ||
  pooler bmm 32.4 (s143; _routing starved until it ends), rmsnorm 2.0, tk2f 97.7, norm2 4.0, unpool 26.7, boundary 4.3.
  Non-expert bytes ~197 MB (59 us H100 floor) vs ~119 us: the m32 GEMMs are the layer-side gap (proj lane, g19 m32).
- v41 (4afa8ea): TK2F_LATE_TRIG 3 (layer.py text patch of moe_src's tk2f): the start-of-kernel
  griddepcontrol.launch_dependents is removed and every CTA triggers when it reaches its gemm2 rows (between the two
  tk2f_consume calls); n2u (T=1, N2U_CFG (8, 1)) and moe_norm2/3 (T>1, g_pdl 4 -> 6) are PDL secondaries. Dependents
  now land on SMs freed by drained tk2f CTAs (the old start trigger let them co-reside and slow tk2f: s8 dead ends).
  H100 d1 timeline: n2u starts 2 us before tk2f ends; tail (tk2f end -> boundary end) 10.0 -> 8.6 us. Variants (H100
  bench, 2 rounds, us): 1 = trigger at CTA end: d1 -0.5/-0.2; 2 = gemm2 start, T=1 only: d1 -0.9/-0.7; 3 = 2 at all
  T + norm2 PDL: d1 -0.9/-0.8, d16 -2.7/-1.1, d32 -0.5/-1.0, v1 -0.7/0.0 (geomean proxy 1.540 -> 1.546). Tests 13/13,
  tk2f_stress 0/600, scored_test 0, replay_test 0.
  Check 36073e6e **1.9526x** (11,585 us): d1 70.9/80.7 (best ever; v36 72.1/83.7), d16 147.7/222.3, d32 177.2/330.0,
  v1 120.4/126.8, v16 267.9/361.8, v32 354.1/550.3. Beats board 1.9459 by noise-level 0.35%: NOT submitted (last spare
  of 9 today); submit first thing after 00:00Z 2026-10-27 (or today if a later check clearly wins).

## Session 14 (2026-10-26 06:20-07:15Z)
- Tool: per-CTA globaltimer stamps (generator /tmp/l14/mk_stamp.py: g_st in moe_tk2f start/gemm2/end, g_sn in n2u
  start/after-wait/after-y-sync/end, read back by exported moe_stamps/n2u_stamps; recreate from this description).
  H100 d1 sliding, v41: last tk2f CTA ends 34.2 us after tk2f start (CTA ends spread 31.0..34.2, gemm2 starts
  20.7..25.2); n2u blocks start 32.8..34.4, pass griddepcontrol.wait at 35.2 (PDL release ~0.9 us after the last
  tk2f CTA), y32 loaded + synced 38.1, end 40.4: 6.2 us after tk2f. The wait -> y-sync part was 3 dependent L2 round
  trips (ids -> inv/gs -> Y).
- **FOUND (SASS audit, /tmp/l14/sass_pdl.py: loads scheduled before ACQBULK = griddepcontrol.wait per kernel):**
  (a) `__ldg` / ld.global.nc loads are hoisted ABOVE griddepcontrol.wait by ptxas (treated as invariant). g17 read x
  with __ldg: as a PDL secondary it read stale x = the s8 "unexplained race" (actually deterministic: replay_test v1
  15/15 fail, NaN/Inf on eager repeats). (b) the opposite: ptxas SINKS plain/nc register loads issued before the wait
  to after it (acquire semantics), so "W loads before griddepcontrol.wait" did nothing in n2u (all W LDGs after
  ACQBULK) and in g17-with-ldcg. cp.async.bulk.prefetch.L2 (UBLKPF) and LDGSTS stay in place; Triton gdc_wait
  kernels (_boundary, _rmsnorm_gain) keep their loads before ACQBULK. Also an ld.acquire spin sinks earlier loads.
  Audit of the current libs: pr1/pr8/g10/g12/g19/pr3 and moe twm/tk2f/norm2/3 have only weights/inputs (or pr1's
  routing, complete) before their waits: no hazard. n2u<4,false> (unused) hoisted inv loads: hazard if ever used.
- v42 (cfd3bc5): N2U_SLOT (T=1 tk2f PF&4 = 84204, y32 in slot order pre-scaled; n2u reads ids/tw before the wait,
  one L2 trip after it; bitwise = 204), N2U_PFW (n2u bulk-prefetches its 64 KB of W rows into L2 before the wait),
  g17 x __ldcg + W bulk prefetch + g17 as PDL secondary of moe_norm3 at T=8 (sliding and full). Bitwise = v41 (d1/v1,
  5 seeds). H100 interleaved (tools/ab_files.py, whole files, 3-4 rounds): v1 154.6/169.7 -> 153.4/168.3; d1 within
  noise (stamps: tail after tk2f 6.2 -> 5.0-5.4 us). Tests 13/13, replay 0/288 (seeds 11, 5), tk2f_stress 0/1200.
  Check 9db0d4eb 1.9545x; **submit 371a17ee 1.9558x (11,566 us) = new board best, overall #49**: d1 71.4/82.3,
  d16 145.1/223.9, d32 185.9/334.2, v1 120.5/132.4, v16 264.9/368.3, v32 368.0/544.3.
- v1 tail now (H100): tk2f end -> norm3 end 3.7 -> g17 end 4.6 -> boundary end 3.3 (was 13.0 total, now 11.6).
  g17 rows/block sweep in PDL mode: (8,4,256) stays (8/16 rows x 512 threads +3..9 us).
- DEAD: UPF2 (first 4..32 finishing tk2f CTAs L2-prefetch the unpool slice at T=1: tie..+0.9 us; HBM is not idle in
  the drain). N2U_BOUNDARY re-test: still +2.8 us. BND_SPIN (second boundary as a 1-block CUDA kernel, PDL secondary of
  n2u WITHOUT griddepcontrol.wait, spinning on a store counter): +2 us (ptxas sank its vector preloads below the
  acquire spin) and 1/10 replay fails at d1 full (race not found): reverted.
- Next ideas: T=8 norm3+unpool fusion needs per-route sum-of-squares (tk2f epilogue atomics) + split-K unpool with
  FP32 out (boundary RB path) to avoid 16 MB of redundant y32 reads; est. v1 -3..4 us. Any kernel edit that wants
  loads before griddepcontrol.wait: use cp.async.bulk.prefetch.L2 or LDGSTS, check SASS with /tmp/l14/sass_pdl.py logic.

## Session 15 (2026-10-26 07:10Z-)
- v43 (928381a) = v42 + moe_src r-moe 499bde6 (moe_twm item order 76 at T=256). H100 interleaved vs v42 (5 rounds):
  v32 414.4/667.8 -> 404.0/652.3 us, rest same. Check 89325570 1.9468x: v32 ours 354.2/541.9 vs 365.6/545.6 in
  9db0d4eb (absolute -11/-4 us as moe predicted), but SGLang's own v32 came in 8% lower this run (673.9 vs 727.7):
  ratio noise. d1 73.0/82.2, d16 149.6/221.7, d32 181.2/323.6, v1 119.1/130.5, v16 273.8/369.3. Not submitted
  (does not beat 1.9558; v42 vs v43 differ only at v32).
- DEAD n2u8 (dev/n2u8_s15.patch): T=8 moe_norm3 + unpool slice in ONE kernel on clusters of 8 CTAs (CTA rank r
  normalizes token r into smem, the cluster reads all 8 rows over DSMEM, each CTA dots its rows). H100 v1 +3.4/+3.3
  us at best (R 9 x CS 8 = 120 CTAs; CS 4/2 worse: each extra token per CTA is ~4-5 us of serial phase 1). Reasons:
  only 15 clusters of 8 fit at 1 CTA/SM (512 thr x 128 regs), a cluster cannot be placed while tk2f's CTAs hold the
  GPC (loses the PDL early start + W prefetch that g17 gets), phase 1 alone ~= moe_norm3, and the k-reduction over
  512 threads is shuffle-bound (64 values x 5 shuffles x 16 warps; butterfly reduce-scatter = 62 shuffles helps).
  Lessons (SASS): a C select between two array elements by lane bit became dynamically indexed LOCAL memory;
  loops with two induction variables did not unroll (array to local); conditional y loads inside per-slot branches
  were serialized (8 round trips): load unconditionally, then select. barrier.cluster.arrive.release = MEMBAR.ALL.GPU.
- DEAD bndc (dev/bndc_s15.patch): both boundaries as a CUDA kernel on a cluster of 8 CTAs per row (1024 columns
  each, the two sums over DSMEM, stores after the last barrier). Standalone cold: T=1 7.6 vs Triton 8.4 us, T=8 7.7
  vs 8.4, T=32 9.4 vs 8.8, T=256 27.9 vs 15.7. In the layer worse everywhere (d1 +0.8, v1 +0.8, d32 +4.6, v32 +30):
  the first boundary ends 6.3 us after the o GEMV (Triton 5.0), the second 3.4 after n2u (Triton 2.2).
  Cluster size 16 needs cudaFuncAttributeNonPortableClusterSizeAllowed (launch error otherwise).
- DEAD qkv on g19 at T=16 (+ .to(bf16)): H100 (16,1024,4,4,2,2) tie (16.9 + 1.3 cast vs cuBLAS 17.1 + 1.8 splitK
  reduce), (8,3,3,1) +4 us. No first-op penalty exists (H100 micro: the same cuBLAS qkv as the 2nd GEMM in a graph
  17.3 vs 18.1 us first): cuBLAS m16 qkv is just 2.1 TB/s on H100 (H200 12.6 + 1.8 us). Asked proj (Requests).
- DEAD pr1 npf (pooler chunks L2-prefetched before its wait) 6/8/12 vs 4: H100 d1 tie..+0.6.
- DEAD csl overlapped with the qkv GEMV (attention starting on q/k/v rows before the gate rows): g13 fills the
  register file at 1 CTA/SM, so csl's 16-CTA clusters cannot co-reside without taking GPCs from the GEMV.
- DEAD (H100 and H200) UPF3: L2 prefetch of the unpool W for the T=32 cuBLAS unpool / T=16 g19 unpool. H100 d32: at
  tk2f CTA end unpool 26.5 -> 22.5 but tk2f +7.8 (stragglers lose bandwidth); in moe_norm2/3 after its wait (bulk:
  the CTA retires only when its bulk prefetches complete, norm +5.6; per-line prefetch.global.L2: same) +1..2 us.
  G10_PFL2 (gemv10 o W bulk L2 prefetch before its wait, d1): H100 +1.8 (csl slowed). DIAG check b1ee542c (174fdf1)
  1.9476x: d32 ratio 1.460 -> 1.407 / 1.915 -> 1.830 (-4%), d16 -1.5%, d1 sliding 2.347 -> 2.333; controls flat.
  So the "spare HBM during issue-bound tk2f on H200" does not exist for prefetches (or L2 thrash). Flags kept, off.
- DEAD BND_ROWS: _boundary with 2/4 rows per program at T=128/256 (weight vectors loaded once per program): H100 v16
  +3.4/+3.3 (2 rows), +12/+10 (4 rows); v32 +0.5/+0.2, +5/+6. Fewer CTAs = less bandwidth; the L2 re-reads are free.
- **v44 (de763eb) = v43 + proj's g19 s15 escape-fixup rework (r-proj dev/g19_s15_layer.patch) + g19 at T=32 for the
  pooler (32,1024,8,2,2,1) on 108 blocks and the unpool (same cfg) + qkv at T=16 on g19 (16,1024,4,4,4,2), FP32 out
  + .to(bf16) (1.2 us cast) + _moe_TWM_ORDER_M = {} (moe s16 H200 proxy: order 76 loses ~3.6 us at v32; my two judge
  runs were within noise).** H100 vs v43: d16 213.8/315.9 -> 212.1/313.9, d32 212.3/442.9 -> 207.2/435.3.
  Check c6a00da7 **1.9608x** (> board 1.9558): d1 72.1/80.3, d16 146.9/220.6, d32 167.4/317.6 (was 181.2/323.6),
  v1 122.4/129.2, v16 273.8/360.5, v32 345.9/519.3. H200 profile d32: g19 pooler+unpool 49.0 (2 calls) vs cuBLAS
  pooler 19.8 + o/unpool 26.5; d16 qkv g19 ~11.2 + cast 1.2 vs cuBLAS 12.6 + reduce 1.8. Submit held for later
  candidates (1 submit left; must be used before the sandbox ends ~19:10Z).
- DEAD (H100, 8 rounds) qkv T=32 g19 (32,1024,4,4,4,2) + cast and o T=32 g19 (32,1024,4,4,2,2): d32 sliding +2.6..3.4,
  full tie (5-round runs had shown -2.2: noise; use >= 8 rounds for 2 us effects). Pooler grid at T=32: 108 = 96 <
  120/132 (+12 us: side-stream router starved). qkv g19 at T=8 (m=8): the pick fails prepare's check (unsupported);
  asked proj for an M=8 mode-3 g19.
- Session 15 summary: v43 (moe twm order) and v44 (proj g19 fixups + T=32 pooler/unpool + T=16 qkv, twm order reverted
  per moe's proxy) on main; v44 check c6a00da7 1.9608x, submit HELD for the day's last bundle (STATE ACTION).

## Session 16 (2026-10-26 08:52Z-)
- Budget at start: 1 submit, 2 checks (verified `okc problems`). v44 submit still HELD per s15 plan (bundle later, submit before ~17:30Z).
- H100 timelines of v44 (d1/d16/d32/v1 sliding, d1 full) re-read: boundaries are not idle HBM time (the pooler
  streams W during them via TRIG/PDL); boundary micro (after a cuBLAS mm, cold vs params pre-warmed): 4.6 us at
  T=1/16/32 either way, so its params' cold misses are not the cost.
- DEAD: KV rows pre-warmed in L2 before the replay (gathered kc/vc[window rows]): csl d16 8.5 cold vs 8.8-10.2 warm,
  nds d32 10.8 vs 12.3, csl d1 3.9 vs 3.9 (d1 already has KV_PF). Attention at d16/d32 is latency-bound: no KV_PF
  extension to g19 qkv.
- H200 profile v44 (c6a00da7): d32 sliding g19 pooler+unpool 49.0 us for 2 x 50.6 MB (~2.1 TB/s): biggest non-expert
  gap (~20 us/layer at d32); posted to proj (Requests s16) to try gemv22 (wgmma) at N=16/32. d32 full: _gd_full 167 +
  _red_fin (waits) = attention ~at KV floor.
- H200 vs H100 difference found: _router_split 5.6/6.5 us on H200 (d16/d32) vs 18-19 on H100 -> pooler G19_GRID 108
  (H100-tuned) likely leaves SMs idle on the H200. Candidate 120 (variants/sub_v45g.py): correct (tests 13/13, scored
  decode 2 seeds, replay decode 0/60), H100 loses as expected (d16 212.5 -> 219.3, d32 204.6 -> 217.4). Queued for the
  bundle check (STATE Next -4); not on main.

## Session 17 (2026-10-26 09:09Z-)
- Budget at start: 1 submit, 2 checks (`okc problems`). Board: #53 at 1.956x (#1 2.380x, #50 ~1.961x). No new
  proj/moe commits since 08:36Z (proj gemv22 for T=128/256 is uncommitted WIP in r-proj dev/).
- H100 timeline d1 sliding (v44): the first boundary ends 5.3 us after the o GEMV, the second 2.4 after n2u. Checked:
  _boundary's Triton SASS keeps all 18 weight/stream LDGs before ACQBULK (only the branch loads after); pr1's SASS: W
  loads (LDG.CONSTANT) before ACQBULK, x read after it: no hoisting hazard. Hypothesis "pr1's L2 prefetch flood slows
  the boundary": npf 0 does end the boundary 1.3 us earlier (timeline), but pr1 ends at the same time (it is
  HBM-bound from its start); ab_flag 10 rounds npf 4/0/2: d1 sliding 91.53/91.60/91.51, full 113.19/113.20/113.06 =
  tie. DEAD on H100; not worth a judge slot.
- H200 v44 profile (c6a00da7) notes: the m=1 GEMVs run at ~2.5-3.2 TB/s on the H200 (qkv g13 11.0 us for 28 MB, pr1
  21.4 for ~59 MB, o g10 6.4 for 17 MB) vs ~4 TB/s practical: ~11 us at d1 sliding+full (~2.4% score) is in the
  m=1 GEMV efficiency (idle-budget item 5, proj's lane). v1 tk2f is 70 us (vs d16's 63): moe's lane.
- Plan for the last 2 checks: see STATE (s17 TIMELINE). No check or submit this session.

## Session 18 (2026-10-26 09:16Z-)
- Budget at start: 1 submit, 2 checks. Board #53 at 1.956x (#50 1.961x, #1 2.380x). proj gemv22 still uncommitted WIP.
- v45 (2dddb88): prepare() tunes the pooler g19 grid at T=16/32 over (108, 114, 120, 126, 132) by timing whole-layer
  CUDA-graph replays (cold L2, 40 interleaved reps, median) on synthetic sliding decode inputs (`_synth_input`,
  ctx 3000, random data; shapes only), 0.12 s; result in state["g19_grid"], printed to the judge log. H100 picks 108
  (T=16 202/203/208/209/209, T=32 318/319/330/330/329 us), stable over 3 runs; A/B vs v44 both orders = parity.
  Forced 114/120/126/132: tests 13/13 + scored decode ok, replay decode 16/32 0/60 (120, 132).
- **Check ea9421fa (2dddb88) 1.9634x**: the H200 ALSO picks 108: T=16 108 185.6 / 114 186.2 / 120 193.2 / 126 195.2 /
  132 195.3 us; T=32 232.6 / 234.0 / 243.8 / 246.1 / 245.7. **DEAD: G19_GRID 120 (s16 queued idea) loses 8-11 us on
  H200** (the router-speed argument was wrong: g19 itself slows with more blocks at 1 CTA/SM beside the router, or
  the router is starved anyway). Per shape: d1 72.8/80.4, d16 148.4/221.7, d32 180.5/320.9, v1 122.4/131.3, v16
  284.3/370.1, v32 367.0/527.4 (= v44 compute; d32 sliding 180.5 vs v44's 167.4 is judge noise).
- Lesson/tool: the in-prepare tuner is a FREE H200 measurement channel: any check can carry whole-layer A/B timings of
  variants (printed in the log) at ~0.1 s each. Use it to measure H200-only questions inside the last checks.

## Session 19 (2026-10-26 09:30Z-)
- Budget at start: 1 submit, 1 check. Board #53 at 1.956x (#50 1.961x, #1 ~2.38x). proj gemv22 still uncommitted WIP.
- v46-v48 (189f79d, c8056e5, d36e095): generalized s18's grid tuner into `TUNE` / `_tune` / `_opt(state, T, name,
  default)`: option sets per T, timed per (T, sliding) on `_synth_input(batch, dev, ctx, q_len, sliding)` (now also
  verify blocks and full layers; full at ctx 80000; our attention ignores scheduler_metadata), 40 interleaved reps,
  median; a candidate that raises host-side is skipped. Pooler g19 grid fixed at 108 (s18 H200 result). New G19_CFG
  picks (qkv T=32, o T=16/32; configs already built) with G19_DEFAULT_OFF; G19_ON now includes "o".
  H100 tuner picks (noisy, +-1 us): T=1 default (sometimes rb 32); T=16 sliding o g19 (-1.3), full default; T=32
  default; T=8 default; T=128 sliding lt qkv off (-1.4), full default; T=256 full lt qkv off (-0.3).
  ab_files decode 8 rounds v45 vs v46: d1 91.54/112.92 vs 91.37/112.82, d16 213.4/314.1 vs 212.4/316.4, d32
  205.7/435.4 vs 205.7/434.9; verify 1 154.2/168.6 vs 155.0/168.7 (= parity). Tune 0.9 s.
- Caveat: synthetic routing is random-uniform (not "dist soft"): options that change the experts would be
  mis-measured; all current options are GEMM/glue choices. Synthetic T=1 sliding 114 us vs real d1 91: local
  expert count differs; only the relative timings matter.
- No check/submit this session (holding the last check for the gemv22 bundle; see STATE).

## Session 20 (2026-10-26 09:38Z-)
- Budget at start: 1 submit, 1 check (`okc problems`). (Note: earlier sessions' clock stamps ran ahead: s19 "10:05Z" was
  really ~09:35Z. Use `date -u`.)
- v49 (b564697): proj gemv22 (r-proj a17318c dev/gemv22.cu -> g22_src.cu, inlined by build.py; own load_inline, +30 s
  build) as TUNE options at T=128: g22:qkv_gate / g22:pooler / g22:unpool / all three, cfg (128, 4, 3, 32), FP32 out
  (consumers already round FP32: qkv .to(bf16), pooler rmsnorm ROUND, unpool boundary RB). `_pack22` (proj's) + `_pack19(tm=1)`;
  `_g22_prepare` checks each pack vs F.linear. Dispatch inside `_g19()` (checked first, only if TUNE picked it).
  H100 standalone (proj's dev/g22.py, cold L2): qkv m128 21.9 vs cuBLAS 21.1, o 14.9/10.2, pooler 30.9/27.5, unpool
  28.8/27.3; m256 all +10..12 us (compute side: not built). H100 in-layer tuner: T=128 sliding default 433.1, g22 qkv
  434.8, pooler 437.2, unpool 433.6, all 443.2 -> keeps default (full the same). Forced all-g22 at verify 16: scored 6/6
  ok, replay 0/30 sliding + 0/30 full. Tests 13/13.
- Why it may win on the H200: v45 profile verify 16 sliding: the T=128 GEMMs (nvjet 128x64 x2 27.5, 88x64 22.8, 48x64
  16.9) ~67 us vs ~35 us floor for 12-bit bytes at 4 TB/s; H100 TMA streams only ~2.9 TB/s (proj).
- No check yet: holding the last check so moe/proj handoffs posted by ~14:30Z can ride along (see STATE).

## Session 21 (2026-10-26 09:45Z-)
- Budget at start: 1 submit, 1 check. Board #53 at 1.956x (#1 2.380x). No new proj/moe handoffs since s20.
- H100 d1 sliding timeline (v49): g13 qkv 0-13.7, csl 14.2-18.5, o g10 15.2-24.2, boundary 21.2-29.6, pr1 22.1-47.3,
  rmsnorm 34.1-48.3, tk2f 48.6-86.9, n2u 82.5-91.2, boundary 87.7-93.6 us.
- v50: g13 PF=2 instantiation (2,4,1,2) as TUNE option g13alt:qkv_gate at T=1 (qkv = 2304 rows = 4.36 rows per
  virtual block, one row in flight each at PF 1: latency-bound). REG 62 (PF 1: 60), no spill. H100 tuner: sliding
  115.1 default vs 116.4, full 80.7 vs 82.7 (loses); a free H200 A/B in the last check. Forced: scored decode 1/16 x3 ok.

## Session 22 (2026-10-26 10:05Z-)
- Budget at start: 1 submit, 1 check. Board #53 1.956x (#1 2.380x). No new proj/moe handoffs.
- v51 (1e457c2): GEMV config alternatives generalized: `G10_ALT`, `G13_ALT`, `G14_ALT` lists, `_cfg(state, T, kind, key, cfg)`
  picks TUNE option "<kind>alt:<key>" (1-based index). New options: T=1 g10alt:o = (1024, 4, 4, 1, 1, 132); T=8
  g14alt:qkv_gate = (2, 2, 8, 32, 1) / (2, 2, 4, 48, 1). H100 tuner (us, sliding/full): T=1 default 114.4/80.4, g10alt o
  114.7/82.2 (PF 2 on 264 blocks: 125.5/91.3, dropped); T=8 default 147.2/169.5, g14alt 1 147.0/169.9, 2 147.3/170.3.
  Forced each: scored decode 1 + verify 1 x3 ok, replay_test 30 inputs " 1 " 0/30 on all four. Free H200 A/B in the last check.
- v52 (9621c71): proj s16 (r-proj 2480c53) T=8 qkv gemv14 without split-K: G14_ALT qkv = (8,8,4,96,1), (8,8,4,96,2),
  (8,8,8,96,1) (kcu 8 needs its own pack: state["g14k"][(key, kcu)]); my v51 kcu-2 alts dropped (ties). H100 tuner T=8
  sliding default 147.4 / alt1 147.0 / alt2 146.5 / alt3 147.8; full 171.8 / 170.6 / 170.5 / 170.8 -> picks alt 2.
  Forced each: verify 1 x3 ok, replay 0/30. G22_CFG = {} (proj's H200 check fb2fe4ee: g22 loses at every m128/256:
  qkv m128 22.1 vs lt 18.8, pooler 29.6 vs 24.4) and its TUNE options removed: -30 s build.
- Not ported: proj G13_CPA (cp.async ring m1): proj's tuner says it ties gemv13 VT 512, which tied on H200 (6b50e844).

## Session 23 (2026-10-26 10:13Z-)
- Budget at start: 1 submit, 1 check. Board #53 1.956x (#1 2.380x, #50 ~1.961x). No new moe/proj handoffs.
- Source of H200 data: r-proj profiles/*.log "tune <key> m<T>: ... F:<F.linear> lt:<cfg i> ..." lines (30 judge runs each,
  standalone incl. launch). Medians F -> best lt (us): qkv m16 17.48 -> 16.83 (cfg idx 3), m32 17.82 -> 17.25 (idx 3),
  o m16 10.04 -> 9.77, m32 10.30 -> 9.98, m128 10.94 -> 10.73, m256 12.80 -> 12.54, unpooler m128 23.21 -> 22.96,
  pooler m16/32 24.4 -> 22.0 (but g19 in-layer is the default there), pooler m128 29.3 -> 25.3 (= our LT_CFG).
- v53 (66fef4c): those lt cfgs as TUNE options (LT_DEFAULT_OFF; T=16 qkv option also turns g19 qkv off). H100 tuner:
  all within +-1.5 us (T=16 full picks lt:o -0.1, T=32 full lt qkv+o -1.4, T=128 sliding lt:o -2.7, full lt:unpool -0.5;
  noise level). Forced all: scored d16/d32/v16/v32 x2 ok, replay 0/20 each.
- v54 (3b5a373): pr1 pooler PF 2 (template instantiation (2,4,2,4)) as TUNE option pr1_pf (alone / with rb 32): H100
  +0.5..0.9 sliding, tie full; PF 2 + npf 8 +3 us (dropped). Forced: scored decode 1 x3, replay 0/30.
- v55 (b36caa7): lt pooler/unpool at T=16/32 instead of g19 (proj diag19 H200: m32 lt 20.3 vs g19 21.6, unpool 19.3 vs
  20.3 after a 3 us PDL predecessor). H100 in-layer: +4..7 us (g19's PDL W prefetch wins): free H200 A/B only.
  Forced: scored d16/d32 x2, replay decode 0/20. Tune 5.2 s, prepare 13.1 s total.

## s24 (2026-10-26 from 10:24Z real UTC)
- v56 (d36131b): proj s16 Handoffs (10:10Z) lt picks: ("pooler", 256) (66,18,...,0,2) and ("unpool", 256) (66,20,...,6,5) as
  LT_DEFAULT_OFF TUNE options at T=256; ("qkv_gate", 32) (66,27,35,3,2,0,2,4) as a new LT_ALT entry (option ltalt:qkv_gate 1,
  validated vs F.linear in _lt_prepare into state["lt_alt_ok"]). proj's o m16/m32 and unpool m128 picks equal ours already.
  H100 tuner: pooler m256 lt +11..12 us (beside the side-stream router, same as s5's other m256 algo), unpool m256 -1.6
  sliding / +3 full, qkv m32 alt tie. Forced (TUNE = only those): scored d32/v32 x2 ok, replay 0/20 x4.
- v57 (7d14445): tuner confirm round: after the min-of-medians pick, re-time default vs pick 40 reps interleaved; keep the
  pick only if it beats the default by TUNE_MARGIN 0.1%. H100: T=32 full and T=256 full fall back to default (pick gains
  0.3/0.2 us vanished), the rest confirmed (T=8 full g14alt 1 -2.4, T=128 full lt o+unpool -2.8, T=1 full -1.0).
- proj g22 is dead (H200); no new moe handoff. Last check still planned ~14:30Z.

## s25 (2026-10-26 10:30Z-11:20Z real UTC)
- Found: the prepare() tuner's synthetic stream was pure noise -> near-uniform routing (T=256 local counts [43,23,20,19,..],
  29 experts) vs the scored loads' skew ([91-97,73-76,32-50,..], 18-21 experts; route_hist.py). Expert-kernel options
  could not be judged. Fix v58 (7ab88e4): `_synth_stream` (TUNE_SKEW): each token's 8 experts = local j w.p. SYNTH_P[j]
  (geometric-ish shares from route_hist, 20 hot local experts) + random off-GPU fill; stream = sum of their unit router
  rows * sqrt(8192/8) + 0.3 noise. Synth now: T=256 362 routes/23 experts [96,70,49,29,19,..], T=128 204/21 [54,36,24,..],
  T=32 39/11, T=16 28/13 (real 48-55/9-11, 23-24/6-10). Weights only, prepare-time, outputs discarded.
- v58: TUNE option twm_order (76) at T=128/256. v59 (46fb62d): + twm_order 4, twm_tail 32 at T=256.
  H100 tuner (skewed): T=256 sliding default 432.5 / o76 418.2 / o4 431.9 / t32 431.3; full 667.3 / 654.0 / 665.8 / 665.5
  -> o76 (confirmed). T=128 o76 tie. Real scored inputs (ab_files verify 5 rounds, notwm vs v58): v32 415.5 -> 402.7,
  666.3 -> 651.9 us; v1/v16 tie. (Unskewed tuner had o76 as a tie: 474.6 vs 473.0.)
  H200 truth unknown (moe proxy: o76 +3.6 at v32; judge v43 89325570: -11/-4): the last check's tuner lines decide.
- Forced each option: scored verify 32 x2 ok, replay 0/20 (o76 also v16). Tests 13/13. Prepare (incl. tune) ~4 s warm.

## s26 (2026-10-26 10:38Z-) NU8: T=8 moe_norm3 + unpool slice fusion (lead 10:32Z item)
- Design (layer.py `_N2U_SRC` namespace nu8, `_experts_nu8`, `_nu8_on`): moe_tk2f (slot-ordered y32) -> nu8 = PDL secondary,
  128 blocks x 512 thr, block (row group of 64 unpool rows, k chunk of 512): W tile -> smem by LDGSTS before
  griddepcontrol.wait; after it the LOCAL slots' y32 chunks -> smem (one LDGSTS batch), h = BF16(sum_s f_s y_s w1)
  (moe_norm3 math/order/rounding, thread = k), mma.m16n8k16 (W rows = M, 8 tokens = N, FP32), 4 k-quarters summed in
  smem -> FP32 slabs [8][8][1024]; `_boundary` NS/SS adds the slabs, RB rounds to BF16. Per-slot sums of squares
  (norm3's tot over all 4096 k) are the crux; two ways, both kept: (a) moe_tk2f PF 20 (cfg 85204; layer text patch,
  w1 via moe_set_ss host static): gemm2 epilogue accumulates (y w1)^2 per segment, one atomic per 8 cols at the
  segment end, into y32 row 64 (zeroed by `_rmsnorm_gain` ZN); (b) NU8_X: nu8 blocks 0..7 (row group 0) add their
  chunk's partials, counter at y32 row 64 col 128, all blocks spin on it (blocks 0..7 dispatched first: no deadlock).
- Exactness: vs main rel diff ~1e-4 (BF16 ulp flips from FP32 sum order); tests 13/13, scored verify 1/16 x3 ok.
- H100 (tools/ab_files.py, 6 rounds, v1 sliding/full, tuner off): main 153.8/168.7, (a) 153.2/168.3, (b) 153.8/169.4,
  ideal (sums free: atomics compiled out, wrong output) 152.0/166.5. So the fused tail saves ~2 us but the sums cost it:
  (a) epilogue bisect: w1 load free, the per-row fmaf accumulate alone +1.1 us (!), flush atomics +1.3; (b) the
  exchange adds ~2 us of latency (atomics + acquire spin). Stamps (globaltimer, rel. to the first wait release):
  y chunks ready 0.9 med / 2.2 max, h done 3.5 max (was 1.2 us of chained LDS; fixed), end 4.2 max; wait release
  itself spreads 0..1.3 us over the blocks. Micro (cold, no PDL) 14 us is meaningless (W from HBM serial).
- Dead inside it: L2 prefetch of the late half's W tiles by the early blocks (tie); FMA dot (2.4 us) -> mma (fixed).
- Shipped as T=8 TUNE options only (H200 decides): {"nu8": True} (= X), {"nu8": True, "nu8x": False} (= PF 20),
  {"nu8": True, "g14alt:qkv_gate": 1}. Default off. Next idea if revisited: the sums for free = have tk2f write
  y32 with the w1 multiply folded... no: the cost is any extra epilogue work in tk2f's serialized row loop; a
  better place is a spare warp (tk2f's producer warp idles after its last stage) reading back the CTA's own y rows.

## s27 (2026-10-26 11:26Z-) twq options, TK2FX / TK2FJ (tk2f as PDL secondary of the pre-expert rmsnorm)
- v61 (6a2f780): moe s17 twq / twm_hot as TUNE options at T=128/256. moe's ready-made tools/s17/twq_layer.patch was cut
  against an older layer.py and REVERTS v60's nu8 X: applied only its moe_src.py part + the twq hunks of layer.py by hand.
  H100 tuner: T=128 picks twq 4 (sliding 361.0 -> 356.6, full 470.5 -> 454.5); T=256 keeps o76. Real ab_files 5 rounds vs
  v60: v16 316.2/462.9 -> 311.6/459.7 us, rest tie. Forced twq 4/12 and twm_hot(+o76): scored verify x2 + replay 0/20 each.
- v62 (5d79338) TK2FX at T=1/8 (pr1 paths, no side stream): moe_tk2f launched with the PDL attribute after the pre-expert
  _rmsnorm_gain, which triggers (TRIG2) right after its own gdc_wait (pr1, which wrote the ids tk2f routes on, has then
  completed). tk2f's routing + producer's first weight stages overlap the rmsnorm; its consumers griddepcontrol.wait
  before the first x read; x via __ldcg (SASS: LDG.E.128.STRONG.GPU after ACQBULK; the lut LDG.CONSTANT before it is a
  weight). One-shot host flag moe_set_tk2fpdl (tk2f_go reads and clears it). Without the attribute the wait is a no-op.
  H100 timeline d1: tk2f starts 0.4 us after pr1 ends (was 1.3), pr1 end -> layer end 43.6 vs 45.5 us.
  ab_files 6 rounds: d1 90.4/112.8 -> 89.9/111.8, v1 154.0/167.7 -> 152.1/167.8. Default ON, TUNE {tk2fx: False} at T=1/8.
- v63 (4276630) TK2FJ: the same at T=16/32 (side-stream router): join the side stream BEFORE the rmsnorm (the rmsnorm
  was not PDL there anyway: PDL_TOKENS = (1, 8)), tk2f its PDL secondary. ab_files 5 rounds: d16 212.4/313.3 ->
  209.9/313.8, d32 205.7/433.8 -> 203.7/432.0. Default ON, TUNE {tk2fj: False}. (moe's H200 stamps: tk2f d1 routed 1.5,
  first data 3.6 us after start: the H200 may gain more than the H100.)
- Tests 13/13; scored d1/v1/d16/d32 x3, replay 0/30 x2 seeds, forced T=8 with and without nu8.
- v64 (2ca2d30) RF_TRIG: attention's _red_fin (decode 1 full, 1 program) triggers its PDL dependents right after its
  own gdc_wait, so the o gemv10 (already a PDL secondary, W preload before its wait) overlaps the merge instead of
  launching at _red_fin's exit. (s4's dead variant triggered at entry: o then fought _gd_full.) attn_src.py edited
  (TRIG constexpr; layer sets attn global RF_TRIG per call from _opt "rf_trig"). ab_files d1 full: 113.0 -> 109.9 (8),
  112.3 -> 110.7 (4), 112.1 -> 110.5 (6). Default ON, TUNE {rf_trig: False} at T=1.
- DEAD: CB_TRIG = the same for verify 1 full (_combine triggers after its wait + o g12 as PDL secondary): +0.7 us
  (o's CTAs crowd the 512-program combine: combine 2.4 -> 6 us). Reverted.
- v65 (4fb5d01) BKT: at T=128/256 moe_route runs on the router's stream right after _routing (before the join); moe_src
  kernel_fn takes data["bucket"] = (meta, sroutes) (2-line edit in moe_src.py: re-apply on re-copy!). The old dead
  "side-stream bucketing at T>=128" fell back to the gemm13/gemm2 path (`bucket is None` condition). H200 v16/v32: router
  8-11 us vs pooler 23-32, so moe_route (1.8/2.7 us) leaves the critical path. H100 v32 -3.0/-3.2, v16 tie. Default ON.
- v66 (33608f5) GNORM: at T>32 the pre-expert rmsnorm is fused into the gather (moe_gathern<F32>, layer text patch into
  moe's lib; moe_src kernel_fn takes data["gather_norm"]/["gather_eps"] (edit in moe_src.py: re-apply on re-copy!)):
  each gathered row normalizes its source token from the pooler output (FP32 math, BF16-rounded input as ROUND).
  Outputs differ from the separate rmsnorm in 0.3-0.4% of elements (reduction-order ulps in p; main vs main is
  bit-identical at v32). H100 v16 0/-2.8, v32 -2.1/-2.0. Default ON, TUNE {gnorm: False}.
- Cold compile + prepare on this H100 (fresh TORCH_EXTENSIONS_DIR / TRITON_CACHE_DIR): import 92 s, prepare 162 s (incl.
  the tuner), 4.3 min total.
- Ideas not done: fuse the d16 qkv g19 FP32->BF16 cast (1.2 us) into csl (attention kernel reads BF16 qkv; ~0.1%);
  _routing + moe_route in one kernel at T>32 (now off the H200 critical path anyway).
- Bundle v66 vs v60 (ab_files 4 rounds, H100, real scored inputs): d1 91.3/112.9 -> 91.1/110.5, d16 211.1/315.5 ->
  210.9/314.2, d32 205.3/436.2 -> 203.4/433.1, v1 153.6/168.9 -> 153.2/167.7, v16 318.3/460.9 -> 309.3/455.5, v32
  402.8/649.8 -> 397.8/647.0 (~+0.9% on the score if the H200 agrees). Main v66: tests 13/13, tk2f_stress 0/600,
  route_judge 0/360, replay all shapes 0/240.
- Session end ~12:15Z. Next session: the last check of the sandbox at ~14:30Z on main (+ any handoff posted by 14:15Z,
  as TUNE options), then submit by 17:30Z if it beats 1.9558 (else v45 2dddb88, check ea9421fa 1.9634x).

## s28 (2026-10-26 12:07Z-) v67 QF32: attention reads g19's FP32 qkv at T=16 (proj s19 request)
- At T=16 the qkv GEMM is g19 (FP32 out); v44-v66 cast it with qkv.to(bf16) (H200 ea9421fa: 1.2-1.3 us, and it sat
  between g19 and attention). v67 (d51f68c): attn_src.py csl::decode templated on the qkv element type (float: float4 /
  float2 loads packed by __floats2bfloat162_rn into the same BF16 words; host dispatches on qkv dtype, 4 instantiations);
  every Gluon/Triton qkv load in the decode paths (_gd_full_body q/xp/k/kp/v, _gw_merge_rows, _norm_gate_store, _combine,
  _combine_hb, _combine_bal incl. REDO, _red_fin) does .to(bf16).to(fp32) (no-op for BF16 inputs: same compiled kernels).
  Verify paths (_gw_consumer_bal, _gw_verify, _attn_split/_new/_norm_rope, nds, nvs, vtk) untouched: layer.py passes FP32
  only when q_len == 1 and batch <= CSL_MAX_BATCH (16) (csl sliding / gdec full), else casts as before.
- tools/qf32_eq.py: FP32 qkv vs its BF16 cast through attn_kernel_fn, batch 1/2/8/16 (+ test configs), 3 seeds: sliding
  out and every K/V cache write bit-identical; full-layer out differs only like the bf16-vs-bf16 control (FP32 atomics).
- H100 ab_files 6 rounds (v66 vs v67): d16 sliding 210.59 -> 209.02, full 313.83 -> 311.26 us. Default ON, TUNE {qf32: False}.
- Tests 13/13; scored all 12 shapes x2 seeds 24/24; replay all shapes 0/240 (20 inputs x 2 router seeds).
- Not done: QF32 for nds (d32 sliding) — only matters if the H200 tuner picks g19:qkv at T=32 (it has not).

## s29 (2026-10-26 12:17-12:35Z): v68 QF32_NDS
- H100 timelines (main v67): d1 sliding tk2f end 93.2 -> n2u 97.7 -> boundary 100.0; d16 sliding tk2f end 192.4 ->
  norm2 196.1 -> g19 unpool 217.0 -> boundary 221.3; v1 tk2f 154.0 -> nu8 158.5 -> boundary 162.9. H200 ea9421fa d32
  sliding: tk2f 75.3, g19 pool+unp 50.2, qkv nvjet splitK 12.7 + splitKreduce 2.0, nds 10.0 + _combine_hb 1.6, norm2 9.3.
- v68 (d196c9e): nds (decode sliding batch >= 32) templated on the qkv type (qld8/qld4 round FP32 to BF16 RN), layer
  QF32 condition extended to batch >= NDS_MIN_BATCH (QF32_NDS). tools/qf32_eq.py now also batch 32/48 sliding:
  bit-identical out + cache. (Its 2 "bad" rows are batch 8/16 FULL, untouched code: FP32-atomic nondeterminism where the
  control happened to match; maxdiff 2e-7..6e-5.)
- torch.mm(x, w.t(), out_dtype=float32) at m32 qkv: same nvjet splitK + FP32 splitKreduce, bit-identical after
  rounding, 22.9 vs 23.2 us: no gain alone. The win at T=32 needs an FP32-native producer: g19 (tuner option) or proj s20.
- H100 ab_files d32 5 rounds: main 202.1/431.7, g19 qkv@32 + cast 205.0/434.1, g19 qkv@32 + QF32 204.4/431.5 us.
- Tests 13/13 (main and g19q variant), scored d32 x3 g19q / x2 main, replay g19q d32 0/40. No judge runs.

## s30 (2026-10-26 12:30-12:50Z): no build, main = v68 kept for the ~14:30Z last check
- Budget (okc problems): checks 19/20 used (1 left), submits 9/10 (1 left). GPU at 100% from moe/proj all session.
- H100 d1 sliding timeline (v68, shared GPU): g13 0-14.2, csl 14.5-19.4, o g10 15.7-24.5, boundary 22.4-30.3,
  pr1 23.1-49.7, rmsnorm 35.5-50.8, tk2f 50.0-92.1, n2u 88.0-97.0, boundary 92.4-99.0 (same shape as s21/s29).
- Considered and not built: _boundary fused into g19 unpool at T=16/32 via a second grid barrier (all blocks spin,
  blocks 0..T-1 do one row each). Est. saving 0.5-1.5 us (launch + PDL release minus the barrier) = <0.2% score;
  the shared H100 cannot resolve it before 14:00Z, and the last-block form lost +2.7 in s8. Tomorrow's sandbox idea.
- build.py on main reproduces submission.py byte for byte (v68, d196c9e). Plan unchanged (STATE).

## s31 (2026-10-26 12:48Z-): v69 = v68 + moe s19 twf_split TUNE options
- Board: layer #56 at 1.956x (#1 2.380x). Budget: 1 check + 1 submit left today (sandbox ends ~19:10Z).
- moe_src.py: applied r-moe 20b864e tools/s19/twm_split_moe_src.diff (moe_twm K-split combinable with order/hot via
  launch bit 22; split limited to light experts). Default path unchanged (TWF_SPLIT 0, order 0).
- layer.py: `twf_split` option (-> _moe_TWF_SPLIT_M[m]); TUNE[128] += {twf_split 66}; TUNE[256] += {twf_split 66},
  {twf_split 66, twm_order 76}. H200 tuner decides in the last check (moe proxy: 0-4 us either sign).
- Tests: bench test 13 PASS; forced A (twf_split 66 at 128/256) and B (twf_split 66 + order 76 at 256): scored verify
  x2 0 bad each, replay verify 0/20 per shape each. H100 tuner (forced, single candidate, shared GPU): A T=128 354.1/456.0,
  T=256 424.7/656.9; B T=256 420.5/654.0 us. Not A/B'd on H100 (H200 tuner decides). Committed as v69.

## s32 (2026-10-26 13:14Z-14:05Z): v70 = v69 + proj QSS + moe s20 twm AUTO K-split; LAST check + LAST submit of the day
- Applied r-proj cb318e6 dev/qss_layer.patch (qkv T=32 = 2 FP32 bmm slabs, attention sums them at load; TUNE qss, default
  on at 32) and r-moe 6f6c487 tools/s20/twm_auto_moe_src.diff (moe_src.py only); TUNE[128] += {twf_split 1022},
  TUNE[256] += {twf_split 1022}, {twf_split 1022, twm_order 76}. Tests 13/13; forced (1022 at 128, 1022+o76 at 256)
  scored verify x2 0 bad, replay verify 0/20 x4; d32 scored x2 0 bad, replay 0/20 x2. H100 ab_files d32 4 rounds
  v69 203.1/433.2 vs v70 204.3/432.8 (tie). Commit 909010f.
- check 0f065691 (909010f) 1.9754x (best check ever; was ea9421fa 1.9634): d1 69.4/78.8, d16 145.7/223.2, d32 178.1/330.1,
  v1 120.0/133.8, v16 277.1/365.8, v32 354.8/527.5.
- submit 703d3a7e (909010f) **1.9954x NEW BOARD BEST** (was 371a17ee 1.9558): d1 69.6/77.8, d16 143.3/227.7,
  d32 174.0/326.6, v1 117.7/128.2, v16 272.2/367.7, v32 356.4/550.5 (11,336 us/step).
- H200 tuner (check 0f065691 log; prepare-time picks, so they may differ per run):
  T=1 sl {pr1_npf 0}, full default; T=8 sl {nu8}, full {nu8, g14alt:qkv_gate 1}; T=16 sl {g19:o}, full default;
  T=32 sl {g19:unpool False, lt:unpool}, full default; QSS wins on H200: default(qss) 158.8/315.6 vs qss False 159.6/317.5;
  T=128 sl default, full {twq 4} (twf_split 1022 268.2/367.3 vs default 269.6/371.4); T=256 sl {twm_tail 32},
  full {twf_split 1022} 509.4 vs default 520.5 (sl: 1022 339.1 vs 336.6, 1022+o76 348.6: o76 loses on H200 again).
  For moe: twf_split 1022 wins full T=128/256 (-4/-11 tuner us), loses/ties sliding.
  layer tune T=1 sliding: default 57.5us | {'pr1_npf': 0} 56.2us | {'pr1_npf': 8} 59.8us | {'kv_pf': False} 57.8us | {'pr1_rb': 20} 58.5us | {'pr1_rb': 32} 59.0us | {'g13alt:qkv_gate': 1} 59.2us | {'g10alt:o': 1} 57.7us | {'pr1_pf': 2} 58.3us | {'pr1_pf': 2, 'pr1_rb': 32} 59.3us | {'tk2fx': False} 58.7us | {'rf_trig': False} 58.0us (confirm default 57.4 vs 56.6us) -> {'pr1_npf': 0}
  layer tune T=1 full: default 78.9us | {'pr1_npf': 0} 78.9us | {'pr1_npf': 8} 80.2us | {'kv_pf': False} 80.0us | {'pr1_rb': 20} 80.2us | {'pr1_rb': 32} 80.5us | {'g13alt:qkv_gate': 1} 81.3us | {'g10alt:o': 1} 81.2us | {'pr1_pf': 2} 80.4us | {'pr1_pf': 2, 'pr1_rb': 32} 81.2us | {'tk2fx': False} 80.2us | {'rf_trig': False} 80.8us -> default
  layer tune T=16 sliding: default 185.8us | {'g19:qkv_gate': False} 187.4us | {'g19:o': True} 184.9us | {'g19:qkv_gate': False, 'lt:qkv_gate': True} 187.0us | {'lt:o': True} 185.5us | {'g19:qkv_gate': False, 'lt:qkv_gate': True, 'lt:o': True} 187.1us | {'g19:pooler': False, 'lt:pooler': True} 189.0us | {'g19:unpool': False, 'lt:unpool': True} 199.3us | {'tk2fj': False} 185.7us | {'qf32': False} 187.6us (confirm default 185.2 vs 184.8us) -> {'g19:o': True}
  layer tune T=16 full: default 246.3us | {'g19:qkv_gate': False} 247.2us | {'g19:o': True} 249.3us | {'g19:qkv_gate': False, 'lt:qkv_gate': True} 246.3us | {'lt:o': True} 245.6us | {'g19:qkv_gate': False, 'lt:qkv_gate': True, 'lt:o': True} 246.8us | {'g19:pooler': False, 'lt:pooler': True} 250.8us | {'g19:unpool': False, 'lt:unpool': True} 248.3us | {'tk2fj': False} 245.5us | {'qf32': False} 247.4us (confirm default 245.9 vs 246.1us) -> default
  layer tune T=32 sliding: default 158.8us | {'g19:qkv_gate': True} 163.0us | {'g19:o': True} 162.6us | {'g19:qkv_gate': True, 'g19:o': True} 166.0us | {'lt:qkv_gate': True} 159.1us | {'lt:o': True} 158.8us | {'lt:qkv_gate': True, 'lt:o': True} 158.6us | {'g19:pooler': False, 'lt:pooler': True} 166.3us | {'g19:unpool': False, 'lt:unpool': True} 158.6us | {'g19:pooler': False, 'lt:pooler': True, 'g19:unpool': False, 'lt:unpool': True} 166.7us | {'ltalt:qkv_gate': 1} 159.3us | {'ltalt:qkv_gate': 1, 'lt:o': True} 159.0us | {'tk2fj': False} 159.9us | {'ltalt:qkv_gate': 1, 'lt:o': True, 'tk2fj': False} 160.2us | {'qss': False} 159.6us | {'qss': False, 'lt:qkv_gate': True} 160.2us (confirm default 159.1 vs 158.9us) -> {'g19:unpool': False, 'lt:unpool': True}
  layer tune T=32 full: default 315.6us | {'g19:qkv_gate': True} 319.5us | {'g19:o': True} 320.5us | {'g19:qkv_gate': True, 'g19:o': True} 336.3us | {'lt:qkv_gate': True} 315.9us | {'lt:o': True} 315.7us | {'lt:qkv_gate': True, 'lt:o': True} 316.6us | {'g19:pooler': False, 'lt:pooler': True} 323.5us | {'g19:unpool': False, 'lt:unpool': True} 316.0us | {'g19:pooler': False, 'lt:pooler': True, 'g19:unpool': False, 'lt:unpool': True} 321.7us | {'ltalt:qkv_gate': 1} 316.0us | {'ltalt:qkv_gate': 1, 'lt:o': True} 316.3us | {'tk2fj': False} 317.3us | {'ltalt:qkv_gate': 1, 'lt:o': True, 'tk2fj': False} 317.3us | {'qss': False} 317.5us | {'qss': False, 'lt:qkv_gate': True} 317.5us -> default
  layer tune T=8 sliding: default 114.3us | {'pr1': False} 116.0us | {'g14alt:qkv_gate': 1} 112.6us | {'g14alt:qkv_gate': 2} 112.6us | {'g14alt:qkv_gate': 3} 112.4us | {'nu8': True} 112.0us | {'nu8': True, 'nu8x': False} 113.9us | {'nu8': True, 'g14alt:qkv_gate': 1} 112.1us | {'tk2fx': False} 115.1us | {'tk2fx': False, 'g14alt:qkv_gate': 1} 113.4us (confirm default 114.1 vs 112.4us) -> {'nu8': True}
  layer tune T=8 full: default 144.2us | {'pr1': False} 146.8us | {'g14alt:qkv_gate': 1} 142.8us | {'g14alt:qkv_gate': 2} 142.9us | {'g14alt:qkv_gate': 3} 143.3us | {'nu8': True} 143.7us | {'nu8': True, 'nu8x': False} 144.6us | {'nu8': True, 'g14alt:qkv_gate': 1} 142.3us | {'tk2fx': False} 145.2us | {'tk2fx': False, 'g14alt:qkv_gate': 1} 144.4us (confirm default 144.4 vs 142.3us) -> {'nu8': True, 'g14alt:qkv_gate': 1}
  layer tune T=128 sliding: default 269.6us | {'lt:pooler': False} 276.7us | {'lt:qkv_gate': False} 271.0us | {'lt:o': True} 269.2us | {'lt:unpool': True} 269.0us | {'lt:o': True, 'lt:unpool': True} 268.4us | {'twm_order': 76} 271.0us | {'twq': 4} 269.2us | {'twq': 12} 270.7us | {'bkt': False} 269.5us | {'bkt': False, 'twq': 4} 269.1us | {'gnorm': False} 270.6us | {'gnorm': False, 'twq': 4} 271.4us | {'twf_split': 66} 268.8us | {'twf_split': 1022} 268.2us (confirm default 268.4 vs 268.4us) -> default
  layer tune T=128 full: default 371.4us | {'lt:pooler': False} 377.8us | {'lt:qkv_gate': False} 371.9us | {'lt:o': True} 368.7us | {'lt:unpool': True} 369.6us | {'lt:o': True, 'lt:unpool': True} 369.0us | {'twm_order': 76} 370.1us | {'twq': 4} 361.5us | {'twq': 12} 364.4us | {'bkt': False} 370.0us | {'bkt': False, 'twq': 4} 362.0us | {'gnorm': False} 371.2us | {'gnorm': False, 'twq': 4} 366.3us | {'twf_split': 66} 367.6us | {'twf_split': 1022} 367.3us (confirm default 372.8 vs 362.2us) -> {'twq': 4}
  layer tune T=256 sliding: default 336.6us | {'lt:qkv_gate': False} 346.4us | {'lt:o': True} 338.1us | {'lt:pooler': True} 347.6us | {'lt:unpool': True} 337.8us | {'lt:pooler': True, 'lt:unpool': True, 'lt:o': True} 347.2us | {'twm_order': 76} 346.3us | {'twm_order': 76, 'lt:unpool': True} 347.1us | {'twm_order': 4} 336.2us | {'twm_tail': 32} 333.7us | {'twq': 4} 346.0us | {'twq': 12} 345.6us | {'twm_hot': 1} 343.9us | {'twm_hot': 1, 'twm_order': 76} 343.0us | {'twm_hot': 1, 'twm_order': 76, 'lt:unpool': True} 341.1us | {'bkt': False} 341.2us | {'bkt': False, 'twm_order': 76, 'lt:unpool': True} 350.2us | {'gnorm': False} 338.8us | {'gnorm': False, 'twm_order': 76, 'lt:unpool': True} 348.0us | {'twf_split': 66} 341.0us | {'twf_split': 66, 'twm_order': 76} 339.6us | {'twf_split': 1022} 339.1us | {'twf_split': 1022, 'twm_order': 76} 348.6us (confirm default 337.4 vs 334.4us) -> {'twm_tail': 32}
  layer tune T=256 full: default 520.5us | {'lt:qkv_gate': False} 524.0us | {'lt:o': True} 520.3us | {'lt:pooler': True} 532.8us | {'lt:unpool': True} 521.0us | {'lt:pooler': True, 'lt:unpool': True, 'lt:o': True} 532.3us | {'twm_order': 76} 521.1us | {'twm_order': 76, 'lt:unpool': True} 521.4us | {'twm_order': 4} 521.5us | {'twm_tail': 32} 511.4us | {'twq': 4} 539.6us | {'twq': 12} 534.2us | {'twm_hot': 1} 521.2us | {'twm_hot': 1, 'twm_order': 76} 520.2us | {'twm_hot': 1, 'twm_order': 76, 'lt:unpool': True} 520.8us | {'bkt': False} 524.5us | {'bkt': False, 'twm_order': 76, 'lt:unpool': True} 523.9us | {'gnorm': False} 523.9us | {'gnorm': False, 'twm_order': 76, 'lt:unpool': True} 523.2us | {'twf_split': 66} 509.5us | {'twf_split': 66, 'twm_order': 76} 511.6us | {'twf_split': 1022} 509.4us | {'twf_split': 1022, 'twm_order': 76} 510.7us (confirm default 522.8 vs 511.8us) -> {'twf_split': 1022}

## s33 (2026-10-26 13:32Z-): v71 BND_SPLIT (no judge runs: today's 20 checks / 10 submits are spent)
- Board: #42 at 1.995x (703d3a7e), #1 2.380x. Judge json: per-run d1 times spread 61-92 us (routing differs per seed:
  local expert count), so the d1 median moves with the routing draw, not with jitter we can fix.
- Why the boundary: H100 d1 timeline, the first _boundary ends 5.6 us after the o GEMV (1 CTA of 512 threads per row
  moves ~224 KB: alpha/beta/gain/s/b loads + s_out/h/h32 stores; one SM's bandwidth bounds it). SASS check: its
  constant/s loads stay above ACQBULK (no sinking).
- v71 (a47269b) BND_SPLIT {(T, 1|2): SPL}: _boundary runs SPL programs per row; each loads the full row and computes
  both norms redundantly (same Triton reductions, so bit-identical: /tmp unit test SPL 1/2/4/8 x T 1/8/16/32 x
  RB/BGAIN/NB variants, 0 bad), but stores only its N/SPL columns (and loads only its gain slice).
  H100 ab_flag (same state, interleaved): d1 SPL4 90.43/110.89 -> 89.97/109.68 (6 rounds), 91.04/110.64 -> 90.54/109.68
  (8 rounds; first-only 90.98/110.08, second-only 90.42/110.06); v1 SPL2 152.84/166.24 -> 151.78/166.14, SPL4 151.66/166.27;
  d16 SPL4 208.66 -> 208.02 (5 rounds), 209.94/312.60 -> 208.48/310.68 (6 rounds), SPL2 209.40/311.42; d32 SPL2
  202.70/432.66 -> 201.14/431.28, SPL4 201.18/431.92. T=128/256 SPL2 LOSES (v16 +2.2/+1.7, v32 +5.2/+4.9): off there.
  Defaults: T=1 4, T=8 2, T=16 4, T=32 2 (both boundaries). Tests 13/13, scored decode x2 0 bad, replay 0/20 all shapes.
  NOTE ab_files (separate prepare() per file) showed d16 sliding +2.3: the two tuners picked different options; use
  ab_flag for changes that don't touch the tuner.
- tools/files_eq.py A B [filter] [seeds]: output diff of two files on scored shapes. v70 vs v70 itself differs at every
  T>=8 shape (FP32 atomics in experts/g19: maxdiff 0.016-0.05 in hidden), d1 is bitwise reproducible.
- Considered, not built: moe_norm2 at T=16/32 split over columns. tk2f-side sums of squares (PF & 16) costs tk2f +2.4 us
  (s26 NU8_X note), and a spin/ticket split inside the norm saves at most ~1 us (the 1-block-per-token norm2 is ~3.6 us
  H100 after tk2f's end incl. the PDL release).

## s34 (2026-10-26 13:44Z-): v72 O_PF (no judge runs: today's budget is spent)
- H100 d1 timeline (idle GPU, v71): g13 0-14.0, csl 14.5-18.9, o g10 15.5-23.6, boundary 21.4-26.3, pr1 22.8-45.5,
  rmsnorm 30.9-46.5, tk2f 46.0-83.6, n2u 79.5-87.9, boundary 84.0-89.8. d16: tk2f end 183.5 -> norm2 187.3 -> g19 unpool
  208.2 -> boundary 212.0. d32: qkv 16.7, nds 17.2-30.3, combine -33.2, o nvjet 33.7-42.4 (8.7 us for 16.8 MB).
- v72 O_PF {32: 0.5, 128: 0.5} (6baba9e): attention kernels take (pf_ptr, pf_bytes); nds/nvs's producer warp, after its
  last K/V load is issued, bulk-L2-prefetches its CTA's share of the first half of o's W (g19 o pack when the tuner
  picks g19:o). csl has the hook too (after its PV MMA) but is off (d16 loses). TUNE option {"o_pf": 0} at T=32/128.
  H100 ab_flag (same state): d32 sl 202.84 -> 201.36, 203.10 -> 201.76 (fraction 1.0: -0.2; 0.3/0.7 -0.7); v16 sl
  310.33 -> 309.50, 310.74 -> 309.18, 0.5 best of 0.3/0.5/0.7/1.0; v32 tie (off); d16 (csl, o g19) 0.5 +0.6, 1.0 +1.6 (off).
  Prefetch at nds START: o 8.7 -> 6.5 but nds 13.0 -> 16.0 (zero-sum; same as s2's side-stream results).
  NDS splits S 4/16 vs 8: +0.5/+2.8 (8 stays). Tests 13/13, scored sliding x2 0 bad, replay sliding 0/20 per shape.
- DEAD POOL_PF (first boundary's extra programs L2-prefetch 8/16/24 MB of the cuBLAS pooler W at T=128): H100 v16 sl
  300.18 -> 300.93/301.76/301.90 (the boundary's PF programs retire only when their bulk prefetches land, so the
  non-PDL pooler starts later; same reason UPF3_AT 1 lost). Prefetch only pays from a kernel that runs long anyway.
- DEAD O_PF on full layers (_gd_full, prefetch after each program's cp.wait_group(0), 1 chunk per lane): H100 ab_flag
  d1 full 109.10 -> 109.57 (0.5) / 110.33 (1.0); d32 full 430.22 -> 430.26 / 431.70; d16 full 309.90 -> 310.61 / 312.21.
  Full attention is HBM-bound (d16/d32) or o is a PDL-preloading g10 (d1). Also d16 sliding with cuBLAS o + O_PF 0.5:
  208.72 -> 208.41 (noise); g19:o without PF 207.97. Patch not kept (re-derive from this note if needed).
- Considered, measured, not built: split qkv into q/k/v (1280 rows) + gate (1024 rows, needed only after attention)
  so the gate GEMV hides behind latency-bound attention. H100 g13 first-op in a graph: 2304 rows 14.21, 1280 rows 10.37,
  gate 1024 rows 8.64 us (ramp-dominated). d1: -3.8 on qkv but the gate (8.6) outlasts csl (4.4): ~zero. d32 sliding:
  nds holds the SMs' smem (2 CTAs/SM), full layers are KV-HBM-bound: only d16 sliding (csl) could hide it (~-4 us H100,
  needs g19 split + csl ungated FP32 out + gated o GEMV or a cuBLAS-side gating pass). Est. <= +0.3%: parked.
- Tests on v72: bench test 13/13; scored_test all 12 shapes x3 seeds 0 bad; replay_test all shapes 0/20, seeds 11+12.

## s35 (2026-10-26 14:11Z-): v73-v75 (no judge runs: today's layer budget 20/20 checks, 10/10 submits is spent)
- v73 (f170e82) = v72 + moe s21 TK2F_L2H 1 (`python ~/r-moe/muse_moe_w4a16/tools/s21/patch_layer.py moe_src.py 1`):
  tk2f weight bulk copies carry L2 policies (evict_last for rows a later 8-token group of the same expert re-reads,
  evict_first otherwise). H100 ab_files vs v72, 4 rounds: d16 208.78/311.42 -> 205.54/306.79, v1 152.34/167.09 ->
  148.78/160.84, d32 -0.5/+1.2, T>=16 verify 0. Tests 13/13, tk2f_stress 0/600, scored_test all x3 0 bad, replay all 0/20.
- TUNER NOISE (important for reading A/Bs and judge runs): identical files' prepare-time tuners pick differently; at d1
  sliding v72 vs v72 = 89.09 vs 90.74 us (pick pr1_npf 0 vs default). Mode 2 (no hint on final rows) showed the same
  apparent d1 +0.5: it was the tuner, not L2H. Read the "layer tune" lines before blaming a change.
- v74 (13e5517): PR1_CFG[1] npf 4 -> 0 (tuner option {"pr1_npf": 4}): both logged H200 tuners (703d3a7e, 0f065691,
  same code 909010f) picked npf 0 at T=1 sliding (-2.3/-0.8 us). H100 d1 sl 87.96/88.10 vs v73 89.58/88.39.
- v75 (482a3b6): TUNE_BASE = per-(T, sliding) defaults under the tuner (`_opt`: tune pick > TUNE_BASE > code default),
  set to the picks both H200 logs agreed on: T=8 sl nu8, T=8 full nu8 + g14alt:qkv_gate 1 (-2.0/-2.1 both runs), T=16 sl
  g19:o. Tuner candidates now include the "off" variants. H100 tie (the tuners converge here). Picks that disagreed
  between the two H200 runs (not in base): T=32 sl lt:o vs lt:unpool, T=128/256 (twf_split 1022 -12 us in one run, twm_tail
  32 / default in the other): the synthetic T=256 timings vary ~3% between runs.
- Tests v75: scored_test all x2 0 bad; replay verify 1/16, decode 16 0/20 (seed 12).
