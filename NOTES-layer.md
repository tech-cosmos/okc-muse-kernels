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

## s36 (2026-10-26 14:21Z-): v76 (no judge runs: today's layer budget 20/20 checks, 10/10 submits is spent)
- v76 (5c5329d) = v75 + moe s21 TWM_L2H (r-moe 12d95a9): only the twm half of tools/s21/patch_layer.py applies
  (tk2f half is already in since v73; script used: /tmp/twm_l2h.py = the `if twm:` rep of patch_layer.py). moe_twm's
  weight bulk copies get evict_last when a later 64/128-token pass of the same expert re-reads them, else evict_first.
  Runs only at T>32. H100 ab_files with TUNE = {} in both files (variants/v75_nt.py, v76_nt.py; avoids tuner noise),
  4 rounds: v16 sl 308.32 -> 307.95, full 457.53 -> 454.30, v32 sl 411.57 -> 410.27, full 662.10 -> 662.57; T<=32 rows
  are the same code (spread +-1.4 us = noise floor of this A/B). Tests 13/13, scored verify x3 0 bad, replay verify 0/20.

## s37 (2026-10-26 14:40Z-): v77 fused pooler+router GEMM at T=256 (no judge runs: budget spent)
- H100 timelines of v76 (idle GPU): v32 sliding: qkv nvjet 21.6 | nvs+combine -> 39.8 | o 10.7 | boundary 8.9 | pooler
  nvjet 128x64 61.9-110.8 (48.9 us: 67 MB at T=256 is ~compute/tile-bound, shares SMs with router_split 12.7 + routing
  16.0 + moe_route 4.4 on the other stream) | gathern 4.4 | twm 254 | moe_norm tail ~10 | unpool 31 | boundary 9.
  v16 sliding: pooler 88x64 33.7 || pr3 router 30.8; twm 178. d16/d32: routing ends ~with the g19 pooler (no gain there).
- v77 (b66a869): FUSED_ROUTER_TOKENS (256,) (the s7 path: ONE torch.mm of h against [pooler; router hi; lo], FP32 out,
  routing sums the hi/lo columns) + now GNORM in that path: moe_gathern takes a row stride (x.stride(0)/2) and reads the
  strided FP32 pooler columns (rounded to BF16 first = the BF16-out pooler), moe_route right after the routing, all on
  one stream; build.py patches moe_src kernel_fn to skip .contiguous() when gather_norm is set. TUNE option "fr" at
  T=128/256 (FR_TUNE builds state["pool_router"], +75 MB): T=128 candidates {fr}, {fr, twf_split 66}; T=256 {fr: False}
  (+ twm_order 76 / lt:unpool combos). H100 fused GEMM (80x128 4x2 cluster) 39.7 us vs pooler 48.9 in parallel with the
  router; twm starts ~5 us earlier. ab_files tuner off (variants/*_nt.py): v32 sl 412.6 -> 406.4, full 663.3 -> 661.4
  (6 rounds); 1st run -2.4/-2.5; without the gnorm part -5.4/-4.4. Tuner (H100) at T=256: fr vs not -2.6 sliding,
  -2.3 full with twm_order 76 too. T=128: +3.5/0 and +8 (v16 loses; timeline twm itself +6 us there, unexplained): off.
  Tests: bench test 13/13 with fr forced at EVERY T>8 (variants/s37frall_nt.py) and on main; scored verify x3 0 bad;
  replay v32 0/20; route_judge v32 2 seeds x 50: 0 fails, gaps like the FP8 reference's. prepare() 6 s on H100.
- DEAD fr mode 2 (router hi/lo GEMM [T,512] first, then the pooler alone while routing + bucketing run beside it):
  router GEMM 8.4 + splitKreduce 2.1 us and the pooler ALONE is 38.8 us (= the fused 4608-col GEMM 39.7): v32 +6/+7,
  v16 +8 us. At T=256 the extra 512 router columns are ~free in the fused GEMM. Code not kept.
- Next ideas from this session's timelines: T=256 glue after the fused GEMM is routing 3.7 + moe_route 2.4 + gathern
  6.2 us H100 (serial, small grids: a PDL chain could save ~1 us); qkv at T=128/256 is 2x its bytes floor but proj
  closed that lane (s16/s28: own GEMMs lose to cuBLASLt).

## s38 (2026-10-26 14:53Z-): v78-v80 (no judge runs: budget for 10-26 spent; sandbox ends ~19:10Z)
- DEAD moe TK2F_L2H mode 4 (evict_last on every group of a multi-group expert; moe Handoffs s21 UPDATE 2) in the layer:
  H100 ab_files tuner off, 2 interleaved pairs vs mode 1 (v77): d16 sl 204.5/205.2 -> 206.7/208.0, d32 sl 200.7/201.4 ->
  202.4/203.1, d16 full -1, rest tie. moe s23 measured mode 4 = mode 1 in its copy. Kept mode 1.
- DEAD (upper bound measured): L2-prefetching the routed experts' weights during the d1 pooler (pr1). Oracle hack
  (variants/hack_pf.py: side-stream Triton cp.async.bulk.prefetch.L2 of the KNOWN local experts' w13q/w2q, forked
  after the boundary; ids from an eager run): best -1.2 us (30% of w13 of 3 experts = 15 MB: 88.74 -> 87.50); all of w13 of
  3 experts +3.6, 1 expert w13+w2 -0.9. HBM is ~saturated during pr1 (proj: g13/pr1 run at bare-TMA-stream speed), so a
  real version (router before the pooler) cannot pay.
- v78 (5c27e0a): g19_src.cu supports M = 8 (odd NT: one ldmatrix.x2 + mma per atom for the last n8 token group);
  G19_CFG (qkv_gate|o|pooler, 8) in G19_DEFAULT_OFF; TUNE[8] candidate {"g19:qkv_gate": True}. vtk::verify (T=8 sliding
  attention) templated on QT (float: BF16 RN at each load via csl::ld16/ld2, like csl's QF32); layer.py VTK_QF32 skips
  the cast when vtk will run. files_eq vs the cast: 0 diff. H100 v1 sliding: g14 qkv 15.8 vs g19 m8 16.7 us (+0.8 with
  the cast dropped; cast alone 1.2-1.6 us); configs swept (AP8 NS3/NS2, AP4 NS6, KC2048): (8,1024,4,4,4,2) best.
  (proj s29 had built an M=8 g19 too, also +1.2 on H100: g14 is mma.sync already.) H200 decides via the tuner.
  H100 o g19 m8: +2.9/+6 us (not a candidate); pooler g19 m8 = g14 pooler, but without pr8's fusion +3.4 us.
- v79 (b1f7a33): TUNE[8] += {pr1 off, g19:pooler}, {pr1 off, g19:pooler, g19:qkv_gate} (H200 decides; H100 +3.4).
  attn_src.py: csl/nvs/nds load_inline names now carry a source hash. REASON: proj's /tmp copy of r-layer built
  muse_csl_s96 from the OLD source into the shared ~/.cache/torch_extensions at the same time; my timeline run printed
  "csl build failed, using Triton" (template mismatch from the mixed build dir). Local-only hazard (judge = 1 process).
- v80 (e598e41): proj s32 evl_all_layer.patch (evict_last on all 9 layer L2 prefetch sites + KV_PF). H100 ab_files
  tuner off 5 rounds: d1 sl 88.66 -> 86.69, full 107.82 -> 107.01, v1 tie. Tests 13/13, scored d1 x3 0 bad, replay T=1/8 0/20.
- Looked at, nothing to build: (a) boundary at d16 "exposed 7.4 us" on the H100 timeline: BND_SPLIT (16,1) 2/8 and
  BOUNDARY_TRIG off are all noise (+-1.5 us). (b) csl alone (graph, cold L2): batch 1 5.2-5.3, 16 9.3-9.8, 32 (nds) 13.7 us
  (8/16 MB of random 64 B KV gathers + latency chain; attention lane). (c) verify full: _new_tokens/_combine_bal end 0.6/3.8
  us after _gw_verify on H100 (PDL-hidden; the H200 profile's 19-28 us rows are mostly waits). (d) n2u (d1 tail) is
  already W-in-registers + routing before the wait.
- H200 budget (profile 703d3a7e): biggest non-expert item = g19 m32 pooler+unpool at d32 (2 calls 50.2 us, ~2.0 TB/s;
  standalone 19.7/call vs bare 13.9): asked proj (Requests s38).

## s39 (2026-10-26 15:28Z-): v81-v83 = PDL edges in the routing glue (no judge runs: budget for 10-26 spent)
- H100 timelines of v80 (tuner off, variants/s40_nt.py): d1 sl 86.7 us (qkv g13 13.2 | csl 5.5 | o ends 4.8 after csl |
  bnd +2.5 | pr1 -> 44.8 | tk2f 45.1-80.2 | n2u +4.7 | bnd +1.8); v1 sl 150.1 (tk2f 82.9 us; nu8 +5.5; bnd +4.0); d32 full
  _gd_full 245 us = 655 MB KV at 2.67 TB/s (80% of H100 peak; d16 78%): no attention headroom worth the layer's time.
  v16 sl glue after the router: routing 3.6 -> moe_route 1.8 -> gathern 4.2 -> twm (7.9 us router end -> twm start).
- _routing standalone (graph of 20, H100): 2 warps 3.26-3.44 us/call, 1 warp 2.84-2.99, 4 warps 3.5-3.75; a packed
  int64 (order-preserving key bits << 8 | 255 - idx) single-max top-8 is bit-identical but NOT faster (2.99-3.13):
  the kernel is launch/latency-bound, not reduction-bound. 1 warp is bit-identical (300 seeds incl. exact ties).
- v81 (cacb271): moe_route launched as a PDL secondary of _routing at T > 32 (layer patch of moe_src: moe_route waits after
  its smem init and reads ids with __ldcg; moe_route_pdl_launch; _routing TRIG at start) + _routing on 1 warp at T > 32
  (ROUTING_WARPS_1_MIN_TOKENS 33). 1 warp at T = 16/32 lost d16/d32 sliding +1.3/+2.6 us in 2 runs (side stream beside the
  g19 pooler; unexplained), so 2 warps there. First ab_files run said v16 sliding -4.7 us; later runs showed that was
  noise (v16 sliding moves +-3 us between ab_files runs: twm's own replay spread).
- v82 (f970e18): T=128 _routing as a PDL secondary of pr3::router (pr3 triggers before its atomics' wait; _routing PDL
  waits first). H100 timeline v16 sl: router end -> twm start 7.9 (v80) -> 6.5 (v81) -> 5.7 us; tools/tlm.py
  (N cold profiled replays per file, median twm start / layer end; ONE file per process: several modules in one process
  share the .so statics and crash with an illegal address (harness artifact; one module with 12 captured graphs
  replayed interleaved is fine: /tmp/l39/multi.py)): twm start -1..2 us vs v80.
- v83 (7fad983): d16/d32 _routing as a PDL secondary of _router_split (TRIG at start). H100 ab_files 5 rounds: d16 -0.1/0.0,
  d32 -0.9/-0.5. On H100 the routing ends before the g19 pooler; it is for the H200 (pooler relatively faster there).
- DEAD: BND_SPLIT 2/4 at T = 128/256 (both boundaries): v16 sl +3/+8, v32 sl +6/+21 us (the big-T boundaries are
  bandwidth-bound; split programs re-read the row).
- All of v81-v83: tests 13/13, replay verify 0/120 + v16 0/80 + decode 0/90, route_judge v16 0/60 + decode 0/180, scored d32 2/2.

## s40 (2026-10-26 15:49Z-): v84 = proj s33 g13o (no judge runs: budget for 10-26 spent)
- v84 (00ce840): `git apply` of r-proj muse_bf16_proj/dev/s33/g13o_layer.patch (cut on v82, applies clean on v83):
  T=1 o on gemv13 (12-bit W, 12.6 MB) as a PDL secondary (W prologue before griddepcontrol.wait, x by __ldcg after);
  TUNE[1] option {"g13o": True}, default off. H100 ab_files tuner off 8 rounds: d1 sliding 87.13 -> 86.38, full
  108.24 -> 106.34 us. Forced (variants/v84o_nt.py = TUNE_BASE g13o at T=1): tests 13/13, scored d1 x3 0 bad, replay
  d1 0/20 + 0/20. Tuner on (H100): T=1 sliding picks g13o (86.6 vs 87.8), full picks pr1_rb 32 (g13o 90.0 vs 90.4 2nd).
- H100 timelines (tuner off, g13o forced): d1 sl qkv g13 14.2 | csl 5.8 | o (g13) ends 4.4 after csl | bnd +3.2 | pr1
  -> 45.8 | rmsnorm +1.1 | tk2f 46.1-81.0 | n2u +4.5 | bnd +1.9 = 87.4. v1 sl: tk2f 58.1-141.8 (83.7) | nu8 +6.5 | bnd
  +4.0 (T=8 boundary 2x the d1 one: 8 FP32 slabs per row + SPL 2).
- FIRST CHECK OF THE NEXT SANDBOX = main (v84): copy the 'layer tune T=1' lines (g13o vs default, sl and full) for proj.
- DEAD: T=8 second-boundary BND_SPLIT (8,2) 1/2/4/8: H100 ab_files 6 rounds v1 149.0-149.8 / 161.7-162.8 us (noise):
  the 4 us after nu8 is the PDL release + latency chain, not the row work.
- main (tuner on) replay T=1/T=8 shapes 0/10 each.

## s41 (2026-10-26 15:55Z-): validation of main v84 + v1 timeline (no judge runs: budget for 10-26 spent)
- main v84 (61fe6fe, submission.py == build.py output), tuner on, H100: bench.py test 13/13; tools/replay_test.py 10 reps x
  seeds 11,23 x 12 scored shapes 0/240; tools/route_judge.py same 0/240 (worst drop-lead gap 0.02). Ready as the next
  sandbox's first check.
- H100 tuner picks this run: T=1 sl g13o (86.6 vs 87.8 confirm), T=1 full kv_pf off (noise, 90.2 vs 90.4), T=8 sl/full
  nu8x off (149.4 vs 151.0 / 162.9 vs 163.7), T=128 full twf_split 66, T=256 twm_order 76.
- H100 timeline v1 sliding (variants/v84_nt.py, tuner off): g14 qkv 0-15.9 | vtk 16.3-23.3 | g12 o ends 31.1 | bnd 34.9 |
  pr8 58.4 (23.5 us for 48 MB 12-bit pooler + 8 MB FP32 router: 2.4 TB/s, ~4 us over the d1 pr1 rate; m8 decode is
  issue-bound, proj's g14 lane) | rmsnorm 59.7 | tk2f 58.7-142.7 (84.0: moe's lane, ~55% of the shape) | nu8 149.0 | bnd 153.0.
  No idle-HBM interval > 4 us in the front (o's W prologue already overlaps vtk).
- Considered, not built: pre-expert rmsnorm fused into pr1/pr8's last pooler block with tk2f as pr1's PDL secondary:
  tk2f already starts 0.3 us after pr1 ends (TK2FX hides the rmsnorm; tk2fx off costs only +0.2..1.2 us), and tk2f
  would then have to read the routing after its wait: expected tie.

## s42 (2026-10-26 16:30Z-): v85 = proj s34 DEC4 (no judge runs: budget for 10-26 spent)
- v85 (cf16011): `git apply` of r-proj muse_bf16_proj/dev/s34/dec4_layer.patch (clean on v84): the 12-bit exponent word
  (s & 0x07800780) | 0x38003800 as one 3-register lop3 (constants made opaque via gridDim.y >> 16 == 0) in g13/g14/g19/
  pr1/pr8. H100: bench test 13/13; tools/replay_test.py 10 x seeds 11,23 0/240; tools/route_judge.py 0/240; files_eq
  v84_nt vs v85_nt = the same atomic-order noise as v84_nt vs itself (d16/d32/v16 FP32 split-K adds; T=1/8 identical).
  ab_files A B B A (v84_nt v85_nt v85_nt v84_nt) 4 rounds: every shape within 0.5 us (H100 g19 shares HBM with the router).
- HARNESS NOTE: make tuner-off variants by inserting `TUNE = {}` right before `TUNE_REPS = 40` (a fixed line number
  breaks when submission.py grows: my first v85_nt put it ABOVE the TUNE dict = tuner on).
- H100 timeline d32 sliding (v85_nt): qkv 19.6 | nds 17.5 | o ends 47.2 | bnd 53.7 | g19 pooler 47.9-78.9 || router_split
  + routing end 77.8 | rmsnorm 81.2 | tk2f 79.6-178.1 (98.4) | norm3 ends 183.6 (+5.5) | g19 unpool ends 207.6 (+24.0) |
  bnd 211.8. Considered, not built: column-split norm3 at T=16/32 using tk2f PF 16 slot sums of squares (as nu8): the
  norm is one L2 round trip + PDL release, <= ~1 us to gain (~0.05%); fused pooler+router-hi/lo rows in g19 at T=32
  (pooler on 132 SMs, routing on the critical path after it): expected tie (s7 cuBLAS version tied), routing-order risk.
  The d32 gap is g19 m32 (consumer-bound, proj) and tk2f (moe).

## s43 (2026-10-26 16:36Z-): v86 = proj s35 DEC3 (no judge runs: budget for 10-26 spent)
- v86: proj's DEC3 12-bit layout (r-layer 10cde26, muse_bf16_proj/dev/s35): kernels via `python
  ../muse_bf16_proj/dev/s35/dec3_src.py g13_src.cu g14_src.cu g19_src.cu pr1_src.cu pr8_src.cu` (decw: prmt with 0x38
  + one lop3 per BF16 pair; old decode kept under -DPJ_DEC4), packers: proj's `_dec3` (with its prepare-time round-trip
  check) copied into layer.py and used by _pack13/_pack14/_pack19. g22 (off, G22_CFG = {}) keeps the old layout:
  `_pack19(..., dec3=False)` from _pack22. pr1/pr8 read the g13/g14 pooler packs (converted with them).
- H100: bench test 13/13; files_eq v85_nt vs v86_nt T=1/T=8 identical (d1 full x6 seeds identical; one transient
  1.6e-4 diff on d1 full seed 1 in the first run did not reproduce), d16/d32/v16 = the usual FP32 atomic-order noise;
  same g19/g13/g14 picks survive prepare's F.linear check (11 g19, 3 g13, 2 g14); replay_test 10 x seeds 11,23 0/240;
  route_judge 0/240. ab_files ABBA 4 rounds (v85_nt v86_nt v86_nt v85_nt): d1 -0.3/-0.4 us, every other shape +-0.2
  (H100 is bandwidth-bound here; proj's H200 proxy says -0.5..-1.6 us per g19 call at m16, -0.2..-0.5 at m32).
- FIRST CHECK OF THE NEXT SANDBOX = main (v86); compare per shape vs 703d3a7e.

## s44 (2026-10-26 16:42Z-): v87 = moe s29 moe_hot at T=256 (no judge runs: budget for 10-26 spent)
- v87: `patch -p1` of muse_moe_w4a16/tools/s29/hot_moe_src.diff on moe_src.py (clean, offsets only) + "moe_set_hot",
  "moe_hot_launch" in _moe__lib's list + `_moe_HOT_M` from `_opt(state, m, "hot", 0)` (beside twm_hot) + TUNE_BASE
  (256, True/False) {"hot": 32} + T=256 tuner candidates {"hot": 0}, {"hot": 24}, {"hot": 40}. moe_hot runs the 65..128-token
  experts (v32 has 2 of 73..98) in one N128 pass before moe_twm (its PDL secondary, which skips them); bitwise = hot off.
- H100 tuner (in bench.py test): T=256 sliding hot32 411.8 | hot0 425.8 | hot24 415.8 | hot40 411.0 -> picks twf_split
  1022 (+hot32 from base) 406.5; full hot32 647.7 | hot0 658.9 | hot24 643.9 | hot40 646.9 -> twf_split 1022. So hot G32
  = -14 us sliding / -11 us full on H100 (moe's H200 proxy -13..15 per v32 call). bench test 13/13.
- Validation (H100): tuner on (picks twf_split 1022 + base hot32 at T=256): replay_test 10 x seeds 11,23 all 12 shapes 0
  fails, route_judge same 0 fails; v32 only 20 x seeds 11,23,37: replay 0/120, route_judge 0/120 (worst drop-lead 0.018).
  Tuner off (variants/v87_nt = hot32 alone) and hot32 + twm_order 76 (v87o_nt): v32 replay 0/40 each.
- Not done: no tuner-off ab_files A/B (the in-tuner hot0 vs hot32 numbers above are interleaved in one prepare and agree
  with moe's 4-round layer A/B, -8 us sliding). Hot at T=128 left out (v16 experts <= ~49 tokens: launch would cost ~2 us).
- FIRST CHECK OF THE NEXT SANDBOX = main (v87); compare per shape vs 703d3a7e (v32 sliding/full should drop ~8-14 us on
  the H200), and read the T=256 tune lines (hot 0/24/32/40).

## s45 (2026-10-26 16:48Z-): no build, no judge runs (layer budget for 10-26 spent: 20/20 checks, 10/10 submits)
- Board 17:00Z: #45 at 1.9954x (703d3a7e); #1 colinlikescode 2.380x, #10 2.180x. main = v87 unchanged; submission.py ==
  build.py output (rebuilt, cmp identical). No new Handoffs/Requests for layer since s44.
- Re-read the 703d3a7e H200 profile: d1 sliding kernels g13 11.4 | csl 4.2 | g10 o 6.7 | bnd 2 x ~4.3 | pr1 20.8 (incl. its
  PDL wait) | tk2f 28.7 (incl. wait) | n2u 5.6 | rmsnorm 1.6; d32 sliding tk2f 82.6 (42%), g19 m32 x2 50.2, qkv nvjet 12.8,
  nds 10.2, moe_norm2 9.8, bnd 8.0, router_split 6.3, routing 4.0. Nothing layer-side left on the dead-end list's terms.
- STATE condensed to <= 150 lines with a concrete first-check plan for the next sandbox (check v87, read tune lines).
- Idea queued (STATE s45): T=1 pooler W partly SMEM-resident in early-launched pr1 CTAs during the latency-bound
  csl/o/bnd windows (L2 prefetch was zero-sum; SMEM residency untested).
- Checked the cheap form of the SMEM idea: pr1's pre-wait window is only the boundary (~3-4 us); pr1 already loads its
  first chunk (PF 1, U 2, NV 4 = ~6 MB over the grid) into registers there, and deeper prefetch in that window (npf 4/8)
  LOST on the H200 tuner (-0.8/-2.3 us for npf 0): it delays the boundary. So the idea needs pr1 to start during csl/o,
  i.e. a T=1 megakernel (o GEMV -> grid barrier -> bnd -> pooler/router, pooler W preloaded into SMEM during the o phase).
  Est. upper bound: ~20-28 MB of spare HBM in the csl/o/bnd window (~13.7 us H100) = -5..7 us d1 sl/full (~+1% layer
  score) minus the grid-barrier cost (~1-2 us). Multi-session build; not started.

## s46 (2026-10-26 17:07Z-): v88 = v87 + proj s36 pr8 bulk L2 prefetch (no judge runs: layer budget for 10-26 spent)
- `git apply -p1 muse_bf16_proj/dev/s36/pr8_bpf_layer.patch` (clean on v87): pr8_src.cu gets bulk_pf (one
  cp.async.bulk.prefetch.L2 per (row group, WS/WN) of atoms [PF, PF+12) issued after the wait), PR8_CFG {8: npf 212};
  probes (PRED/NOW/NOMATH/RLAST/RMID) are compile-time off. build.py rebuilt.
- H100: bench test 13/13 (tuner picks unchanged: T=1 g13o, T=8 nu8x off, T=256 twf_split). ab_files ABBA 6 rounds tuner
  off (v87_nt v88_nt v88_nt v87_nt): v1 sliding 148.4/148.8 -> 146.6/146.8 (-1.9 us), full 161.2/162.0 -> 159.5/160.5
  (-1.6 us), agrees with proj's 7-round -1.6/-1.6. replay_test 10 x seeds 11,23 verify 1: 0/40; route_judge same 0 fails
  (worst gap 0.013, drop-lead 0.009).
- FIRST CHECK OF THE NEXT SANDBOX = main (v88).

## s47 (2026-10-26 17:24Z-17:40Z): probe only, no build, no judge runs (budget spent)
- No new Handoffs since s46 (proj s37 and moe s32 posted notes only). Board 17:25Z: #46 at 1.9954x; #1 2.380x.
- H100 d1 sliding timeline, main v88 (idle GPU): g13 qkv 0-13.1, csl 13.6-18.7, o g13 14.6-22.7, boundary 21.4-26.6,
  pr1 22.4-45.8, rmsnorm 31.3-46.9, tk2f 46.3-80.3 (34.0), n2u 77.6-85.0, boundary 80.4-86.9 = 86.9 us. Same shape as s34
  (89.8): every edge is already PDL-overlapped; the gains since then are in pr1 (-0.6) and the chain start. No idle gap
  > 1 us. Biggest remaining d1 slices: tk2f 34 us (moe; ~2x its bytes floor) and pr1 23.5 us (proj; ~1.6x).
- Next sandbox: unchanged plan in STATE (first check = main v88).

## s48 (2026-10-26 17:40Z-17:50Z): housekeeping only
- Budget spent (layer 20/20 checks, 10/10 submits), sandbox ends ~19:10Z. Verified submission.py == build.py output,
  no /tmp input dependencies in tools/, added variants/ to muse_decode_layer/.gitignore. main = v88 unchanged.

## s49 (2026-10-26 17:56Z-18:05Z): notes only, no build, no judge runs
- Layer budget for 10-26 spent (20/20 checks, 10/10 submits); sandbox expires ~19:10Z, before the 00:00Z reset.
- Board 18:00Z: we are #46 at 1.9954x (703d3a7e = v70); #1 colinlikescode 2.380x, #10 2.208x, #50 above us is unchanged in kind.
- New since s48: proj s39 (side-stream HBM stream dilates a tiny-kernel chain by its whole time; only +3.8 us with
  1 TMA/SM in flight) and s40 (throttled pre-wait W prologue / pre-wait stream: dead). Nothing to integrate.
- Next sandbox: unchanged, STATE "FIRST THING" (check main v88, submit if > 1.9954x).

## s50 (2026-10-26 18:12-18:20Z): notes only
- Budget spent, sandbox ends ~19:10Z, push deadline 18:30Z: did not port moe s33 RG (r-moe 7ac1069) - it needs a
  merge onto the patched moe_src (strided moe_gathern at T=256, hot/L2H/route-PDL patches), not doable+validated in 15 min;
  value ~-1.1 us v16 (~+0.04%). Queued as STATE item 4. main v88 unchanged.

## s51 (2026-10-26 18:27Z): notes only
Budget 0/0 left (okc problems), sandbox ends ~19:10Z, past the 18:30Z push cutoff: no build started (a RG port onto the
patched moe_src would not finish and validate in time). Board 18:27Z: us #46 1.9954x (703d3a7e = v70); #1 2.380x; #10 2.208x
(up from 2.180x at 17:00Z). main = v88 unchanged; next sandbox: STATE "FIRST THING" list as is (check v88 first).

## s52 (2026-10-26 19:12Z-, sandbox okc-night4, session 1): v89 = v88 + RG (moe s33) merged with GNORM, T=256 TUNE option
- New sandbox works: v88 rebuilt from git, bench test 13/13. Layer budget for 10-26 still 0/0 (resets 00:00Z).
- moe_routegathern (layer.py moe patch, after moe_gathern): moe s33 RG (route + gather, one CTA per token, stable
  route order) merged with moe_gathern's pre-expert rmsnorm on strided FP32 / BF16 pooler rows: one launch instead of
  moe_route (PDL secondary of _routing) + moe_gathern, and each token's row is normalized ONCE (gathern: once per route).
  Fix vs moe's version: ids/X read with __ldcg (moe's reads ids via const __restrict__ after griddepcontrol.wait: the
  s14 .nc-hoist hazard). moe_src kernel_fn: data["rg"] skips moe_route and launches it in place of moe_gathern.
- tools/rg_eq.py: RG vs non-RG outputs + KV writes bitwise on v16/v32 + T=33/40/48, fr on/off, 3 seeds: every DIFF also
  differs in the non-RG control (baseline FP32-atomics nondeterminism at T=128 pr3 / T=33), one T=33 case 5.8e-5 only.
  RG forced: bench test 13/13, scored_test verify x3 seeds 0 bad, replay_test verify 32 0/40.
- H100 ab (forced tune at T=256, us sl/full): {} 389.7/630.3 vs rg 390.8/627.6 (tie); twf_split 1022 387.1/622.2 vs
  + rg 379.6/617.7; reversed 8 rounds: split1022 384.2/623.2 vs +rg 379.5/616.1, split66 384.1/623.9 vs +rg 378.6/615.6.
  => rg wins only together with twf_split: -4.6..5.5 sl, -7..8 full (~+0.2% score if the H200 agrees).
  T=128: rg (RG_DEFAULT) +1.8 us v16 (bucketing is hidden on the side stream there): no T=128 option.
- TUNE[256] += {rg}, {rg, twf_split 1022}, {rg, twf_split 66}. H100 tuner picked {rg, twf_split 1022} for both layer types.

## s53 (2026-10-26 19:25Z-, sandbox okc-night4, session 2): v90 = v89 + moe n4s1 NORM_EARLY
- moe 9898def's diff applied to moe_src.py with `patch` (11/16 hunks clean; failed hunks = moe-only context: the
  #define block, moe's routegather (layer has its own routegathern), moe's _lib/flags). By hand: `#define NORM_EARLY 1`
  in moe_src (layer's _moe__lib loads _moe_CUDA_SRC directly), python NORM_EARLY = 1, moe_norme_launch in the
  functions list, routegathern zeroes meta[100 + tid], every layer meta alloc 100 -> 132 ints (moe_route writes
  meta[100..131] now: a 100-int meta would be written out of bounds).
- Tests: bench test 13/13, scored_test verify x3 seeds 0 bad, replay_test verify 16/32 x2 seeds 0/40 each.
  files_eq v89 vs v90 (tuner off): v32 bitwise identical (3 seeds); v1/v16 diffs also appear in the v89-vs-v89
  control (known FP32-atomics nondeterminism: pr3 at T=128, T=8 path).
- H100 ab_files tuner off (variants/v89_nt, v90_nt: TUNE = {}, TUNE_BASE T=256 {hot 32, rg, twf_split 1022}, T=128 sl
  {twf_split 1022}), 3 runs x 8-12 rounds, both orders, v90 - v89 us: v32 sl -1.0/-1.3/-2.0, full -3.0/-3.8/-3.9;
  v16 sl +3.1/-2.1/-2.0 (position-sensitive), full -3.2/+0.4/+0.2; v1 tie (unchanged path). => main v90.

## s54 (2026-10-26 19:33Z-, sandbox okc-night4, session 3): v91 = v90 + __ldcg in moe_norm / moe_norm2 (moe n4s2 request)
- moe's SASS audit (r-moe 5960afc): moe_norm read ids via LDG.E.CONSTANT hoisted above griddepcontrol.wait (ACQBULK).
  layer's moe_src had the same code: ids/tw (moe_norm, moe_norm2) and inv (moe_norm2) now __ldcg. SASS after: in
  moe_norm/moe_norm2/gathern/routegathern only prepared w1 loads precede the wait. Cache-op only (bitwise same math).
- Tests: bench test 13/13, scored_test verify x3 seeds 0 bad, replay_test 0 fails.
- DEAD (oracle, s54): qkv split at d1 sliding = q/k/v GEMV (1280 rows, first op) + gate GEMV (1024 rows, 12.6 MB 12-bit)
  launched after csl as its PDL secondary, griddepcontrol.wait moved to the gate's END (so it streams beside csl; o =
  its PDL secondary), gating epilogue NOT implemented (output wrong: timing upper bound). tools/gsplit_oracle.py
  (`python tools/gsplit_oracle.py base|gs`, run from muse_decode_layer; GS_CFG env = g13 cfg of the gate, GS_PDL 1/2).
  H100 timelines: qkv' 9.2-9.6 us (-3.5..4), but the gate takes 8-11 us beside csl (~1.3 TB/s: ramp-bound small GEMV,
  shares SMs with csl and o's prologue) and o ends ~24 vs 23.4. ab_files tuner off, 8 rounds A B C A: base 86.32/86.22,
  gate 240 blocks (2,2,2,1) 86.82, (2,1,4,1,480) 87.61 us. Wait-before-x version (pdl 1): +4 us. The qkv->o window is
  ramp/latency-bound GEMVs at ~2 TB/s, not idle HBM: moving bytes into it doesn't pay (same lesson as s34/s38).

## s55 (2026-10-26 19:46Z-19:55Z, sandbox okc-night4, session 4): expert-weight L2 prefetch beside the d32 pooler = DEAD
- No judge runs (layer 20/20 checks, 10/10 submits until 00:00Z). main = v91 unchanged.
- Idea: g19 m32 pooler is consumer-bound (H100 30 us for 50 MB = 1.7 TB/s), so prefetch the routed experts' w13q into
  L2 (evict_last) beside it; tk2f (~HBM-bound) then reads part of its bytes from L2. Oracle: tools/expf_oracle.py +
  dev/expf_s55.patch (an _EXPF hook before/after _route on the side stream), prefetch kernel on its OWN stream joined at
  the end of kernel_fn (on the router's side stream it serializes the router: +13 us). Specs pre (oracle ids recorded
  untimed, launched at the fork) / post (real ids after the routing); unit 0 = expert prefix, 32768 = per row tile.
- H100 d32 sliding (tuner off, base 197.5 us): 16 x 128-thread prefetch blocks: pre 2 MB/expert 208.1, per-tile 8 KB
  208.2, 16 KB 213.5, post 219.8. Timelines: 40 MB prefetched -> tk2f 97.8 -> 90.8 us (20 MB -> -2..3.6), BUT the
  prefetch blocks sit on the ~24 SMs the pooler leaves to router_split (20.9 -> 33.6 us): routing ends 12-16 us later.
  With 2 blocks: issue-limited (bulk prefetch ~50 GB/s per SM), kernel runs 105-171 us and holds 2 SMs that tk2f's
  grid needs: tk2f 153-237 us. Real ids exist only at ~the pooler's end (routing ends 71 vs pooler 72.6 on H100), so
  the 'post' form has no window anyway. Upper bound even with free routing ~-7 us/40 MB (~0.3-0.5% score at d32) and
  it needs SMs nobody has spare. Dead unless the g19 pooler's own producer warps issued it AND routing came early.
- Real form also built and measured (dev/xpf_s55.patch: xpf_k in the n2u source, side stream after _route, TUNE option
  xpf = MB budget, 1 KB per row tile chunks): H100 ab_flag 3 rounds xpf 16 / 8 vs off: d16 +2.8/+2.6 sl, +2.6/+2.4 full;
  d32 +2.4/+1.2 sl, +3.0/+1.1 full. Timeline d32 sl xpf 16: xpf_k 7.3 us (16 MB, after the routing which ends ~with the
  pooler) and tk2f 95.5 -> 94.4 only. Prefetch efficiency in tk2f is ~10-25% of bytes/BW (oracle 20/40/80 MB -> -2/-3.6/-7
  us): tk2f's own evict_last re-read lines (multi-group experts at T=32) and its 250 MB stream evict most prefetched
  prefixes before their items run. So even the H200 window (router 6 us vs pooler 16-25) would give <= ~1 us: not added.

## s56 (2026-10-26 19:59Z-20:25Z, sandbox okc-night4, session 5): v92 = v91 + pr1 non-bnd x/x32 __ldcg
- No judge runs (layer 20/20 checks, 10/10 submits until 00:00Z).
- proj s47 SASS audit request: pr1_src.cu non-bnd path (the live one) read x32 (line 181) and x (line 316) with __ldg
  after griddepcontrol.wait: safe in today's SASS but fragile (.nc may be hoisted above the wait). Now __ldcg (as r-proj
  dev/pr1.cu since s13). Cache-op only. bench test 13/13, replay_test 10 x seeds 11,23 d1/d16 0 fails.
- H100 timeline d32 sliding v91 (tuner on): qkv nvjet 16.8 | nds 17.3-32.5 + combine -35.0 | o nvjet 6.7 | bnd 5.2 |
  g19 pooler 43.7-72.8 (|| router_split 19.1, routing 20.5, end 72.2) | rmsnorm 1.7 | tk2f 74.4-173.1 (98.7) | norm2
  -177.1 | g19 unpool -199.5 | bnd -203.6.
- Reviewed for a new layer-side lever, none found that isn't already measured dead: nds S 4/16 (attn s34), warm KV for
  nds (s16: -1.5 us at best, attention latency-bound), g19 m32 consumer bound (proj s12-s44), expert L2 prefetch beside
  the pooler (s55), qkv g19 at T=32 (tuner +2.6 H100). The remaining d32 gaps are tk2f (moe) and g19 m32 (proj).

## s57 (2026-10-26 20:20Z-, sandbox okc-night4, session 6): v93 = v92 + BND_SEP at T=8 (no judge runs: budget resets 00:00Z)
- Tail oracle (H100 ab_flag 6 rounds, same state; wrong outputs, timing bounds; /tmp/tail_oracle.sh recipe: monkeypatch
  `_tail` to return (h, stream1) when f is given = no second boundary, and proxy `_G10["lib"]` with n2u/nu8_launch no-ops):
  d1 sl/full 85.8/105.5 -> no bnd 84.3/104.0 -> no n2u+bnd 78.8/98.9; v1 143.9/155.1 -> 139.5/150.9 -> 132.8/143.8.
  So the whole post-tk2f tail is 7.0 us at d1 and 11.1 us at v1 (H100); the T=8 boundary alone 4.3 us (d1 1.5).
  A persistent experts->norm2->unpool->boundary kernel must still do the y32 L2 trip + norm + GEMV + 2 grid barriers
  (~0.7 us each, vs PDL release ~0.9): on paper <= ~1-2 us at d1; not built (N2U_BOUNDARY / BND_SPIN already lost, s8/s14).
- Oracle 2: the T=8 boundary reading only slab 0 of nu8's 8 FP32 slabs: v1 -2.0/-2.4 us. Cause: `_boundary` loaded the
  branch with the N=8192-wide masked layout, so only the ~2 warps covering cols < 1024 issued all 8 slab loads.
- v93: `_boundary` SEP constexpr (BND_SEP_TOKENS = (8,)): the sliced branch is its own [NS, NB] tile over all threads
  (tl.sum over slabs), first NB columns as separate tensors, cols >= NB = alpha*s (bitwise as before there); the row sums
  are now sum(rest) + sum(first NB) (FP32 order change only). H100 ab_flag 8 rounds: v1 143.70/155.12 -> 142.65/154.10
  (T=8 only 142.74/153.82), d1 85.55/105.25 -> 85.65/105.33 (tie: kept off at T=1, d1 bitwise = v92).
  SPL 1/2/4/8 at T=8 with SEP: noise (2 stays). Tests 13/13, scored verify x3 0 bad, replay verify 1/16 x seeds 11,23 0/80.
  files_eq vs v92: d1 sl identical; d1 full diff = the tuner (v92's copy picked g13o; v92 vs v92 identical when both pick pr1).
- Oracle 3 (H100 ab_flag 5 rounds, second boundary skipped, g19 unpool kept): d16 198.98/295.94 -> 195.65/292.64,
  d32 196.88/414.54 -> 192.38/410.88: the T=16/32 second boundary costs 3.4/4.3 us. It launched WITHOUT PDL there
  (PDL_TOKENS = (1, 8), from when the unpool was cuBLAS, which never triggers), and the g19 GEMM never triggered either.
- v94 G19_TRIG (g19_src.cu: npre bit 16 = griddepcontrol.launch_dependents at the gemm's start; `_g19(..., trig)`):
  at T=16/32 the g19 unpool triggers and the second boundary is its PDL secondary (PDL=True, launch_pdl); same for
  the g19 o GEMM -> first boundary (G19_TRIG_O_TOKENS; o is on g19 at T=16 sliding by TUNE_BASE). Safety: the boundary's
  pre-wait loads are stream/alpha/beta/gains only (never g19's out); it can launch only after every g19 CTA is resident
  (trigger = all CTAs), so g19's mode-2 epoch grid barrier cannot starve; stream1 was complete before tk2f started
  (rmsnorm triggers after its own wait). H100 ab_flag 6 rounds (unpool only): d16 198.19/294.58 -> 197.06/293.59,
  d32 193.96/412.39 -> 192.35/410.87 (second pair: -0.8/-1.1, -1.6/-1.4); + o edge, 8 rounds: d16 sl 197.01 -> 195.32,
  full tie (o not on g19 there). d1 unchanged (path not taken). TUNE[16], TUNE[32] += {"g19trig": False}; with trig on,
  the T=32 {"g19:o": True} option also gets the PDL boundary (the H200 tuner may now pick it).
- DEAD (oracle): n2u's redundant y32 reads (every block reads all 8 slots = 128 KB, ~16 MB of L2 per call). Variant with
  every slot reading slot 0's row (variants/s57_y0.py, TUNE = {}): H100 ab_files 6 rounds d1 86.26/106.20 vs 86.21/106.35:
  the n2u tail is its latency chain, not L2 bytes (so a K-split n2u would not pay at T=1).
- v94 tests: bench test 13/13, replay_test d16/d32 x seeds 11,23 0/80, scored_test d16/d32 x3 0 bad. H100 tuner: {g19trig: False} lost in every T=16/32 tuner line (e.g. T=16 sl 235.4 vs 232.3).

## s58 (2026-10-26 20:41Z-21:00Z, sandbox okc-night4, session 7): v95 = v94 + moe tk2h as T=16/32 TUNE option
- No judge runs (budget resets 00:00Z).
- Port: `git -C ~/r-moe diff 1f3b71b 3710aa8 -- muse_moe_w4a16/submission.py` applied to moe_src.py with `patch` (10/14
  hunks; by hand: TK2G_PAR define, norm3 condition in kernel_fn; DIAG hunks skipped). Layer-side edits in moe_src.py (not
  layer.py text patches): the h16 PF4 SA instantiation `tk2f_go<8,4,2,false,false,true,16,1,4>` built outside TK2_EXP
  (cfg 14W004, W = cfg/1000 % 10); `griddepcontrol.wait` before tk2h_consume (TK2FX/TK2FJ: x is the PDL predecessor's
  output); tk2h_consume x loads `__ldcg` (were __ldg: .nc after the wait); `griddepcontrol.launch_dependents` at the
  start of the CTA's gemm2 phase (= TK2F_LATE_TRIG 3 for the merged consumer, so norm3 is an early PDL secondary).
  python: `TK2H_M = {M: W + 1}` (SA only) in _tk2f_cfg; layer `_experts_moe` sets it from _opt "tk2h". moe has an
  equivalent tools/n6/patch_layer_h16.py (not used).
- Correctness: bench test 13/13 (forced tk2h 4 at T=16/32 and with the tuner), scored_test decode x3 0 bad, replay_test
  d16/d32 x seeds 11,23 0/160, tools/tk2h_stress.py (tk2f_stress with _moe_TK2H_M set; args: reps file W+1) 0/400 (W3) +
  0/240 (W0) bitwise vs the 3-launch moe_tk2 path. Layer outputs are not bitwise run-to-run anyway (g19 red.add order).
- Timing. H100 full grid (ab_flag 5 rounds, seed 99, tuner off): ties everywhere (d16 sl +3.7 with W2/W3): DRAM-bound
  locally, as moe warned. G88 proxy (variants/s58_g.py = tk2f grid from env MOE_GRID; MOE_GRID=88; ab_flag now takes
  SEEDS=7,11,13,17), base / tk2h 3 (W2) / tk2h 4 (W3), us:
  d32 sl s7 216.1/207.2/208.1, s11 224.7/224.6/222.7, s13 216.1/202.6/201.2, s17 202.7/189.5/189.6;
  d32 full s7 433.7/426.3/427.9, s11 433.0/433.7/432.5, s13 418.9/421.0/422.4, s17 419.3/406.1/405.5;
  d16 sl s7 168.4/170.1/169.8, s11 163.3/158.9/159.8, s13 168.5/170.7/170.3, s17 167.5/159.8/160.6;
  d16 full s7 281.1/283.8/283.3, s11 305.2/307.9/312.1, s13 279.0/282.9/292.6, s17 291.8/297.6/297.3.
  W0 (tk2h 1) lost at d32 sl on seed 99 (+7): dropped from TUNE.
- H100 tuner (synthetic skewed routing): T=32 sl default 225.5 vs {tk2h 4, lt:o} 220.2 -> picked; T=32 full -> {tk2h 4}
  (430.4 vs 432.4); T=16 sl/full tk2h +5 us -> not picked (synthetic T=16 has no >8-token expert).

## s59 (2026-10-26 20:56Z-22:10Z, sandbox okc-night4, session 8): v96 = v95 + NU8D at T=8 (no judge runs: budget resets 00:00Z)
- Board 21:00Z: we fell to #52 (1.995x, 11.34 ms); #1 2.380x, #10 2.208x. Per-load ratios vs SGLang (703d3a7e):
  d1 2.43, d16 1.88, d32 1.68, v1 2.19, v16 1.93, v32 2.00 (geomean of loads): d32/d16 are the weak loads.
- Both lead items were already done: T=8 norm3+unpool = nu8 (s26, H200 default), persistent tail bounded s57.
  Built instead NU8D = nu8 without its sum-of-squares exchange (s26: ideal-vs-X ~2 us H100). The post-expert norm is
  per SLOT (h = sum_i f_i rsqrt(ss_i/K+eps) y_i w1), so pull the per-slot scale out of the GEMV: nu8 D computes
  P_i = W . BF16(y_i w1) per local slot (slots as mma N, up to 8 n-tiles), red.adds P_i over the 8 k chunks into
  y32 rows 65..80 ([64 slots][1024]) and row-group-0 blocks red.add ss_i into y32[64][slot]; the pre-expert
  _rmsnorm_gain zeroes the local slots' rows + ss after its wait (ZL; ~4 KB/program); the T=8 boundary (SC, SEP path)
  forms sum_i f_i rsqrt(ss_i/K + eps) P_i over the token's local slots (masked [8, 1024] tile), then RB/BF16 as before.
  Numerics: per-slot BF16 rounding of y w1 instead of one rounding of the weighted sum (same precision class);
  scored v1 errors equal to 4 digits (5.841e-2 vs 5.839e-2 etc.).
- Boundary versions (micro /tmp/bnd_micro.py, no PDL, X slabs = 3.9 us): per-slot loop with tl.sum inside 6+ us
  (in-layer +3.3 us); [64, 1024] tile 128 regs / 78 spills; per-k-chunk P slabs = 3x X's bytes for the busiest
  token (boundary at T=8 is per-program L2-bytes bound); [8]-vector slot scales cost ~1 us of layout conversion:
  fixed by loading ids/tw (pre-wait) and ss as [8, NB] broadcast tiles. Final micro 4.9 us (its ids->load chain is
  pre-wait in the layer).
- H100 timeline v1 sl: tk2f end -> nu8 end 5.3 (X 5.6-6.3), nu8 end -> bnd end 3.5 (X 2.8-2.9).
  ab_flag 6 rounds ABBA, SEEDS 99,7,13, tuner off: v1 sl 142.67/116.46/135.03 (X) -> 142.05/115.78/134.62 (D),
  full 154.94/128.62/155.28 -> 154.64/128.15/154.87: -0.6..0.9 sl, -0.4..0.6 full (~+0.08% score).
- Tests: bench test 13/13 (tuner on), scored_test v1 x2 0 bad, replay_test v1 x seeds 11,23 0/80, nu8 D unit vs torch
  (/tmp/nu8d_unit.py: nl 0/5-7/30/64 local slots) ss and P rel err <= 5e-7. Default NU8_D True; TUNE[8] += {nu8d: False},
  {nu8d: False, nu8x: False} (the H200 decides).

## s60 (2026-10-26 ~21:15Z, sandbox okc-night4, session 9): measurements only, no code change (budget 0 checks/0 submits until 00:00Z)
- H100 tuner on v96 at T=8 picked {nu8d: False, nu8x: False} (sl 143.6 vs 145.4 default; full 155.0 vs 156.5) on SYNTHETIC routing.
  Real inputs (ab_flag 6 rounds, SEEDS 99,7,13, TUNE = {} variant): D / X / plain-nu8 (no X) / D:
  v1 sl s99 143.05/143.69/143.03/143.20, s7 117.10/117.89/116.52/117.11, s13 135.19/135.51/133.07/135.14;
  full s99 154.45/155.06/154.82/154.81, s7 128.59/129.01/128.57/128.96, s13 155.35/156.37/156.09/155.87.
  D beats X (~-0.5) as in s59; plain nu8 is -0.9 avg on sliding vs D (s13 -2.1), +0.2 on full. All three stay H200-tuned
  options (TUNE[8]); no default change: the 00:00Z check's 'layer tune T=8' lines decide.
- H100 timelines (TUNE = {}): d16 sl 201 us = qkv g19 17.6 | csl 12.2 + o g19 tail 7.6 | bnd 5.6 | pooler g19 28.0 beside
  router 18.5 + routing | rmsnorm 2 | tk2f 108.7 | norm2 3 | unpool g19 21 after norm2 | bnd 2.5.
  d32 sl 201 us = qkv cuBLAS 17.3 | nds 15.0 + combine | o 6.8 | bnd 5.4 | pooler g19 29.6 beside router 21.3 | rmsnorm 2.3 |
  tk2f 98.2 | norm2 3.6 | unpool g19 23.6 after norm2 | bnd 2.7.
- Pooler phase micro (/tmp/s60/pool_micro.py: _route + _g19 pooler from prepared state, cold L2): T=16 router alone 11.1,
  pooler 108 blocks 25.4, 132 blocks 25.3, concurrent (as now) 27.8 (grid 120: 27.7), serial router->pooler 35.4;
  T=32 11.9 / 26.9 / 26.9 / 31.8 / 37.2. => the router costs the pooler only +2.5 (T=16) / +4.9 (T=32) us; the pooler grid
  does not matter; the phase floor is g19 itself (48 MB in ~25 us = 1.9 TB/s on H100; proj diag: issue-bound).
  A fused pooler+router g19 can't use 12-bit for the router's lo rows (exponents ~8 below hi: escapes overflow -> whole
  matrix uncompressed), and would save <= ~2 us at T=32 only. Not pursued.
- DEAD again (s9/v32): pr3 router at T=32 beside the g19 pooler. Micro concurrent phase -1.2..1.6 us with cfgs
  (32,32,2,2,512,4)/(32,64,2,4,256,4)/(32,64,2,4,1024,4), but in the layer (ab_flag 5 rounds x 3 seeds) every pr3 cfg
  is +1.5..2.7 us on d32 sliding and +0.5..2 on full (base [0]/[4] 194.98/195.34, 203.54/203.79, 196.85/196.54 sl).
  The micro misses the routing/rmsnorm/tk2f PDL chain. pr3 at T=16: no micro gain either.
- Conclusion for the plan: at T=16/32 the three g19 calls (qkv at T=16, pooler, unpool) are ~65-75 us of ~200 us on H100
  at 1.6-2.2 TB/s; that, the experts (moe) and nds attention (15 us vs 5.6 us KV floor at d32) are where the time is.
  Layer glue is ~done; posted to proj under Requests.

## s61 (2026-10-26 21:23Z-21:45Z, sandbox okc-night4, session 10): v97 = v96 + moe n8 BAL + proj s51 TAG2 (no judge runs: budget resets 00:00Z)
- Integrated: moe n8 BAL (r-moe tools/n8/patch_layer_bal.py@de18925, run on muse_decode_layer/: tk2f per-segment
  gemm13->gemm2 counters + rounding rows of the two phases on different CTAs; cfg + 1000000, SA pf 4 only, NU8 pf 5 needs
  MOE_BAL_NU8 = not built) and proj s51 TAG2 (r-proj dev/s51/tag2_layer.patch, git apply -p1 clean: g19 consumers read
  ks/st/tr from a producer-written smem int; split-K red.add order noise only).
- BAL timing, H100 ab_flag 4 rounds ABBA, SEEDS 7,11,13,99, TUNE = {} (variants/s61_nt.py), us off/on/on/off:
  d16 sl s7 166.9/162.4/162.2/166.4, s11 160.1/157.6/157.9/160.2, s13 166.2/162.5/163.0/166.0, s99 195.5/195.6/195.8/195.4
  (mean -2.6); d16 full s7 279.6/276.1/276.4/279.8, s11 303.7/304.3/304.1/303.5, s13 274.0/274.2/283.2/274.3,
  s99 294.7/296.7/297.1/294.7 (mixed); d32 sl s99 194.4/198.2/198.1/194.3 (+3.8), other seeds 0..+0.7; d32 full 0..+3;
  d1 tie (BAL not used at T=1). Same picture as moe (d16 win, d32 loss).
- Defaults: TUNE_BASE (16, True) += bal True; TUNE[16] += {bal: True}, {bal: False}; TUNE[32] += {bal: True},
  {bal: True, lt:o}. H100 tuner (synthetic routing): T=16 sl keeps default (bal 223.2 / no-bal 224.3 us), T=16 full picks
  {bal: True} (309.5 vs 310.9), T=32 picks tk2h 4 (bal 226.1, loses).
- Tests: bench test 13/13 (tuner on); BAL forced: scored_test decode x2 0 bad, replay d16/d32 x seeds 11,23 0/80;
  main: replay_test all 12 shapes x seeds 11,23 0/12 each, scored_test all 0 bad; tools/tk2f_bal_stress.py (BAL at
  T=16/32 vs 3-launch path) 0/800 bitwise.
- DEAD / bounded this session (H100):
  * nds (instead of csl) at decode 16 sliding (CSL_MAX_BATCH 8, NDS_MIN_BATCH 16), S 8 / 16: +4.6 / +4.0 us (ab_flag 5
    rounds: 195.3 -> 199.9 / 199.4). Full unaffected. (S 12 fails: _combine_hb needs a power-of-2 S.)
  * pooler W L2 prefetch during sliding attention at T=16/32: g19 pooler is consumer-bound, a warm L2 does not speed it
    (/tmp/s61/pool_warm.py: T=32 cold 27.0 us, each further back-to-back call 26.0; T=16 25.6/25.0).
  * qkv at T=32/16 (QSS cuBLAS bmm 16.2/15.8 us cold) vs the best raw Triton read of the same 37.7 MB cold in a graph
    (15.7 us, 264 programs; /tmp/s61/rd_bw.py): at the H100 cold-stream floor. Nothing layer-side there.
  * T=1 pr1 SMEM-preload during csl/o (s45 idea) re-bounded: pr1 is 1024 thr x 64 regs (a full SM register file), so it
    cannot co-reside with o; pr1 already preloads ~6 MB pre-wait during the boundary, and the csl window is not free
    (G10_PFL2 +1.8 us). Realistic upper bound ~-3 us d1 for a multi-session rewrite (o+bnd+pr1 megakernel w/ grid
    barrier): not started.
- H100 timelines (main v96, tuner on): d1 sl 85.3 us = qkv g13 12.8 | csl 4.9 | o g13 -> 22.1 | bnd -> 25.2 | pr1 -> 44.1
  (58 MB at ~3.1 TB/s) | rmsnorm | tk2f 44.6-78.9 | n2u -> 83.4 | bnd -> 85.3. v1 sl 148.0 = g14 15.2 | vtk 6.6 | g12 o -> 29.0 |
  bnd -> 32.4 | pr8 -> 54.1 | tk2f 54.5-140.7 | nu8 -> 145.2 | bnd -> 148.0.

## s62 (2026-10-26 21:38Z-21:55Z, sandbox okc-night4, session 11): v98/v99 = CSL_PW (no judge runs: budget resets 00:00Z)
- Idea: the sliding attention's K/V gathers depend only on inputs (page table, cache, seq_lens), not on the qkv GEMM.
  csl (decode sliding, B<=16) gets template PW: page-table/cos-sin/K/V gathers first, then griddepcontrol.wait, then the
  qkv rows (q, new k/v, gate); launched with the PDL attribute; the g19 qkv at T=16 triggers at start (trig bit 16).
  SASS (PW instances): 62-64 LDG before ACQBULK, the qkv loads after. vtk (verify 1 sliding) got the same (VTK_QLOADS).
- H100 ab_flag (SEEDS 99,7,13, ABBA): d16 sliding 5-6 rounds: -0.7..-1.0 us every seed (195.4 -> 194.7, 162.5 -> 161.7,
  162.9 -> 162.0) -> default CSL_PW_TOKENS (16,), TUNE[16] {cslpw: False}. d1 sliding (after g13): +1.3..1.7 us (off,
  no TUNE). v1 sliding (vtk after g14): -0.3 / -0.2 / +0.0 -> TUNE[8] {cslpw: True}, default off.
- Timeline d16 sl: csl starts 16.5 vs 17.3 (only ~1 us early): the g19 qkv (180 KB smem/CTA, 132 CTAs) leaves no room under
  its default carveout. npre bit 17 (CSL_PW_CARVE / TUNE cslcarve) sets g19's PreferredSharedMemoryCarveout 100: then one
  csl CTA/SM co-resides from 2.8 us, but its gathers slow g19 17.3 -> 20.6 us and csl still ends ~8-9.7 us after qkv
  (only 132 of 512 CTAs early); qkv xb 1 (148 KB, 3 csl CTAs/SM) same picture. ab_flag: carve +0.3..0.6 vs no-carve PW.
  On the H200 (more HBM headroom, g19 issue-bound) the tuner decides.
- Tests: bench test 13/13; scored d16 x3 (forced) + x2 (default) 0 bad, scored d1/v1 sl x3 forced 0 bad; replay d16 0/160,
  v1 0/160 (vtk PW forced) and default.

## s63 (2026-10-26 21:55Z-22:10Z, sandbox okc-night4, session 12): CSL_PW safety audit (lead 21:51Z), no judge runs
- Lead's ask: list what csl/vtk PW read before griddepcontrol.wait, show nothing earlier in the replay writes it, stress
  with inputs that change between replays.
- Pre-wait reads (attn_src.py csl::decode / vtk::verify, PW=true): page_table rows, seq_lens[b] (OLD lengths, not
  cache_seqlens), cache_loc[b/tok], k_scale/v_scale, cos/sin rows (`_cos_sin`: built from rope_theta only, config-derived),
  K/V cache rows at token positions t = clamp(.., 0, seq_lens-1), i.e. only the context BEFORE this step. Post-wait:
  the qkv rows (q, new k/v, gate). Writers: the new token(s)' K/V rows at cache_loc (positions seq_lens..seq_lens+L-1) are
  written by csl/vtk itself and taken from smem/registers for its own attention; with a valid page table those slots
  never equal a gathered slot (same as without PW: CTA 15 writes while the others gather, unchanged by PW).
  No other kernel in the layer writes k/v cache, page table, seq_lens, cache_loc or scales.
- Ordering: PW's primary is the qkv GEMM (g19 at T=16, g14 at T=8), which is the FIRST kernel of kernel_fn (KV_PF is T=1
  only, not combined with PW). Its PDL attribute has no in-graph predecessor, so the graph launch fully orders it after
  the previous replay and the harness's input copies/flush. Hazard found in the control below: if a kernel that writes
  the cache and triggers early ever precedes the qkv GEMM in the same graph, PW races. Comment added at CSL_PW_TOKENS.
- Tests (H100, main v99 + comment = v100):
  * tools/pw_race.py (new): per input, step A = D moved back by L tokens (writes D's last L context rows), step B = D;
    fixed random qkv per step, so attention is bitwise deterministic. 40 inputs x {d16 csl, v1 vtk}: eager ref twice
    nondet 0; split (graph [csl A] then graph [qkv GEMM trig-at-start -> csl/vtk B PW], flush between) 0 diffs; split_nf
    (no flush) 0; joint (one graph A -> GEMM -> B PW) 0 (B can't get SMs until A is done); joint2 (one graph [A, B PW],
    B direct PDL secondary of A, which triggers at start) 39/40 and 40/40 DIFF = the test sees a real race.
  * tools/step_test.py (new): the whole layer, step A then step B in the same captured graph (caches not re-copied):
    d16 sl 0/12 + 0/12, v1/v16 sl (cslpw forced at T=8) 0/12 + 0/12. NB the reference tolerance does NOT catch one stale
    row (/tmp/s63_sens.py: stale B passes 6/6), so pw_race.py's bitwise check is the real evidence; the whole layer is
    not bitwise deterministic (g19 split-K red.add order: tools/step_bitwise.py nondet 6/6), so it can't be used there.
  * s62's replay_test (fresh inputs: new K/V cache, page table, seq_lens, positions every replay) d16/v1 0/160.
- Verdict: CSL_PW (T=16 default) and vtk PW (TUNE[8]) are safe to check at 00:00Z.
- Bound (tools/gather_bw.py, H100, graph replay + cold L2, same method as attn's csl numbers): raw Triton gather of
  random page-1 cache rows (K+V, both heads = 128 B/row each) B16 9.12 us, B32 12.0 us (64 B/row = one head: 8.35 /
  10.5). csl B16 = 9.14 us: AT the gather floor. d32 sliding (nds/csl 15.8-15.9 us standalone) has ~4 us headroom
  (= ~0.2% total at 0.05%/us d32 sl): a B32 kernel reading both heads' 128 B rows per CTA would be the way. Not started.

## s64 (2026-10-26 22:02Z-22:20Z, sandbox okc-night4, session 13): d32 sliding attention bound, csl NC 2 DEAD (no judge runs)
- Goal: s63's "~4 us headroom" of d32 sliding attention (nds + _combine_hb 15.9 us standalone vs a 12.0 us raw gather).
- Theory: csl at B32 = 1024 CTAs at 80 regs (6 CTAs/SM = 792/wave): two waves of latency-bound gathers. Built csl
  template NC 2 (each warp gathers 2 x 32-key chunks at once, CTA = 256 keys, cluster 8 per KV head, 512 CTAs; QSS slab
  loads in-kernel; 32-bit slot offsets; launch bounds (128, 4)). Cache writes bitwise = NC 1; output within FP8-P noise
  (2e-2 of max vs nds at B32, 9e-3 vs csl at B16). H100, real d32 inputs, graph replay, cold L2 (all us):
  raw csl B32 NC1 19.6 -> NC2 17.5 (128 regs, 8-24 B spill; minB 3 no spill: worse); through attn_kernel_fn with the
  layer's QSS slabs: nds+combine 17.4-17.9 vs NC2 18.1-18.9 (qss loads moved after the gathers: qss = bf16). DEAD.
  Patch + test: dev/csl_nc2_s64.patch, dev/csl_nc2_s64_test.py (needs the patch). Reverted (6 more instantiations = compile time).
- Why there is no 4 us: tools/gather_real.py (new) gathers the layer's REAL d32/d16 inputs (2.75M-row cache, page table
  trip included): B32 128 B rows 12.3, csl pattern (64 B head halves from separate programs) 12.9, one head 10.9;
  B16 9.5 / 9.8 / 8.7. So the half-row pattern costs only ~0.5 us. And nds+combine with EVERYTHING warm in L2 is still
  14.0 us (cold 16.7): nds is bound by its own per-CTA pipeline latency, not DRAM. Consequences:
  * K/V L2 prefetch beside the d32 cuBLAS qkv bmm (it can't trigger PDL) wins at most ~2 us (= 0.1%), minus the
    prefetch kernel's own ~2 us on the serial path and L2 eviction by the 38 MB weight stream: not built.
  * fusing _combine_hb into nds (last CTA per batch merges; the combine ends 2.3 us after nds in the layer) bounded at
    ~1-1.5 us d32 sl (~0.06%): not built.
  * On real inputs csl B16 is 12.9 us cold vs the 9.5 gather floor (s63's "B16 at floor" used gather_bw's synthetic
    600K-row cache); at d16 its tail is overlapped by the PDL o GEMM anyway.
- H100 timelines main v100 (tuner on): d1 sl 87.4 = g13 qkv 12.9 | csl 5.3 | g10 o -> 23.4 | bnd -> 28.1 | pr1 -> 45.6 |
  tk2f 46.1-81.0 | n2u -> 85.5 | bnd -> 87.4. d16 sl 200.9 (tk2f 108). d32 sl 198.6 = bmm qkv 16.4 | nds 16.6-31.0 |
  combine -> 33.3 | o nvjet 6.8 | bnd | g19 pooler 40.9-70.4 (router/routing beside) | tk2f 71.1-169.9 | norm2 | g19
  unpool -> 196.0 | bnd -> 198.6. v16 sl 304.9: glue router end 82.3 -> twm start 88.5 (_routing 3.8, moe_route 4.1,
  moe_gathern 5.9; RG option exists only in TUNE[256]); v1 sl 145.4.

## s65 (2026-10-26 22:19Z-, sandbox okc-night4): v101 = v100 + proj s53 gemv24 (thanks proj) as default at T=16/32
- No judge runs (budget resets 00:00Z). `git apply ~/r-proj/muse_bf16_proj/dev/s53/g24_layer.patch` (r-proj 3f24c15):
  applied clean on 71be355 (g24_src.cu, build.py "24", G24_CFG, _pack24/_g24_prepare (own load_inline, per-pick
  F.linear check; falls back to g19/cuBLAS if the build or a pick fails), _g19() takes g24:<key> first).
- H100 ab_flag (variants/s65_nt.py = TUNE {}), 4 rounds A,B,C,A x SEEDS 7,11,99, us:
  d16 sl  161.3/157.3/194.0 -> pool+unp 159.4/155.4/192.4 -> all4 158.4/154.0/191.5  (all4 -2.9)
  d16 full 280.0/303.9/294.8 -> 278.6/301.7/292.9 -> 276.9/300.3/291.3             (all4 -3.0)
  d32 sl  203.8/212.4/195.8 -> pool+unp 202.2/210.3/194.6 -> all4 204.8/212.6/196.0 (pool+unp -1.5)
  d32 full 422.9/422.6/414.4 -> 420.7/420.5/413.3 -> 420.2/420.6/412.4             (-1.8..-2.0)
  T=32 split on top of pool+unp: +o sl +2.3/+2.4/+1.2 full -0.6..-0.3; +qkv sl +1.8/-0.2/+0.2, full -1.6/-1.6/-1.8.
- Defaults (TUNE_BASE): T=16 sl+full all four GEMMs on g24 (_G24_ALL); T=32 pooler+unpool. TUNE[16] off-options
  (all / pool+unp / o / qkv), TUNE[32] off (pool+unp, pool, unp) + on (qkv, o). H100 tuner (cold run) keeps g24 defaults
  everywhere (off = +1.5..2 us).
- Tests: forced all4 at T=16/32: bench test 13/13, scored_test decode x3 18/18, replay d16/d32 x seeds 11,23 0/160.
  main: cold compile+prepare+test (fresh TORCH_EXTENSIONS_DIR/TRITON_CACHE_DIR) 5:05 total, 13/13; replay decode 0/240;
  route_judge decode 0/240.
- g24 PDL: W prologue before griddepcontrol.wait reads only prepared weights; x bulk-loaded after the wait; out zeroed
  in-kernel + epoch barrier (one counter per grid, as g19). cslpw at d16 sliding: csl is now the PDL secondary of the g24
  qkv (still the graph's first kernel, so s63's audit holds: csl's pre-wait loads read only inputs).
- FIRST CHECK AFTER 00:00Z = main v101: read 'layer tune T=16/32' g24 lines (H200 picks) besides the s64 list.

## s66 (2026-10-26 22:30Z-, sandbox okc-night4, session 15): v102 = v101 + proj s54 g24 stream-K (g24sk) at T=32 pooler+unpool
- No judge runs (budget resets 00:00Z). `git apply -p1 muse_bf16_proj/dev/s54/g24sk_layer.patch` (r-proj 0f2e581, thanks proj):
  clean. G24_SKCFG (RR | 256 template = stream-K over ring stages), state["g24sk"], `_g19()` takes g24sk:<key> before g24:<key>.
  TUNE_BASE (32, sl/full) += g24sk pooler+unpool; TUNE[16] on-options (pool+unp / qkv / all3), TUNE[32] off (both/pool/unp) + qkv on.
- H100 ab_flag variants/s66_nt.py (TUNE {}), 4 rounds x SEEDS 7,11,99, us: [0] v102 | [1] SK off (= v101) | [2] v102 + SK qkv
  d32 sl  203.9/211.2/192.6 | 205.3/212.1/194.7 | 204.6/212.1/193.1   (SK pool+unp -0.9..-2.1; +qkv loses 0.5..0.9)
  d32 full 420.2/419.1/411.8 | 422.1/420.5/413.8 | 420.9/419.6/412.0   (-1.3..-2.0)
- Tests: bench test 13/13; scored_test decode 32 x3 PASS; replay_test d32 seeds 11,23 0/160; route_judge d32 0/240.
- Cold compile+prepare+test (fresh TORCH_EXTENSIONS_DIR/TRITON_CACHE_DIR): 5:09 total (v101 5:05), 13/13 PASS, twice.
- H100 timelines (v102): d1 sl 86 us = g13 qkv 13.3 | csl 5.1 | o 8.1 | bnd 4.3 | pr1 22.5 (~bytes floor) | tk2f 34.2 | n2u+bnd tail 6.4.
  d32 sl 198 us = cuBLAS qkv 16.6 | nds+combine 17 | o 6.3 | bnd 4.5 | g24sk pooler 27.6 (|| router_split 20.3 + routing 21.0, ends
  just after the pooler) | rmsnorm 2 | tk2f 98.7 | norm2 +3.8 | g24sk unpool ends +21 after tk2f | bnd 2.7. The remaining lever at
  d32 is tk2f (moe) and the unpool weight stream after it (can't be prefetched: dead s14/s15/s55).
- Next session at/after 00:00Z: FIRST CHECK = main v102 (STATE "FIRST THING").

## s67 (2026-10-26 22:46Z-, sandbox okc-night4, session 16): pre-check stress of main v102 (no code change yet)
- Handoffs: nothing new since s66. RG at T=128 was already measured (s52: +1.8 us v16): not re-added.
- H100 v1 sliding timeline (v102): g14 qkv 0-14.8 | vtk 15.3-21.8 | o g12 -29.4 | bnd -34.0 | pr8 22.5-55.3 | rmsnorm
  -56.7 | tk2f 55.7-139.1 (83 us: moe lane) | nu8 -144.3 | bnd -147.0. The 8 us tail = known dead fusions (s8/s14/s15).
- Whole-replay stress over all 12 scored shapes, router seeds 5,17, 20 inputs each: tools/replay_test.py 0/480,
  tools/route_judge.py 0/480 (worst pick gap 0.034 at v32 sl seed 17 = the FP8 reference's own 0.034; limit 0.05).
- H100 tuner (shared GPU, noisy: T=32 sl default 222.5 vs 235.7 us across two runs) picks tk2h 3/4 (+lt:o) at T=32 sl
  in both runs (-5..-8 us): watch the H200 'layer tune T=32' line in the 00:00Z check.
- 00:01Z **check c973ab6f = main v102 (7f1c60b; submission.py of e8b5fee/233c0f1 identical): 2.0817x [2.050, 2.113], 10,866 us/step**
  (board best 703d3a7e 1.9954x). Per shape H200 us (ours / SGLang, ratio; vs 703d3a7e):
  d1 65.5/171.8 2.624 (-4.1) | 74.9/182.7 2.441 (-2.9); d16 132.3/258.2 1.952 (-11.0) | 215.1/436.3 2.029 (-12.6);
  d32 163.6/260.2 1.590 (-10.4) | 319.4/615.7 1.928 (-7.2); v1 108.9/260.2 2.390 (-8.8) | 115.9/263.3 2.272 (-12.3);
  v16 277.0/532.4 1.922 (+4.8) | 372.0/716.1 1.925 (+4.3); v32 361.1/738.4 2.045 (+4.7) | 535.8/1090.5 2.035 (-14.7).
  v16 and v32 sliding are a few us SLOWER than 703d3a7e (suspects: T=128/256 tune picks, NORM_EARLY / moe_hot changes; or noise).
  The local okc client hit my 600 s tool timeout; the job kept running (okc jobs / okc status <job> --json).
- H200 tuner picks (profiles/submission.c973ab6f398c42b9.log 'layer tune'; default vs pick, us):
  T=1 sl -> rf_trig False (71.5 vs 70.4) | full -> g13o (64.4 vs 64.2)
  T=16 sl -> {g19:qkv False, lt:qkv, lt:o} (166.8 vs 166.3) | full -> g24:o False (231.3 vs 230.4)
  T=32 sl -> {tk2h 4, lt:o} (168.7 vs 167.3) | full -> {tk2h 4, lt:o} (339.6 vs 337.0)
  T=8 sl -> g14alt:qkv 3 (105.5 vs 103.6) | full -> g14alt:qkv 2 (143.7 vs 143.4)
  T=128 sl -> twq 4 (275.1 vs 273.0) | full -> twf_split 1022 (388.5 vs 375.7)
  T=256 sl -> {rg, twf_split 1022} (342.6 vs 340.6) | full -> {rg, twf_split 1022} (653.1 vs 644.7)
  => T=32 tk2h 4 + lt:o (H100 tuner agreed) and T=128 full twf_split 1022 (-13 us) are candidates for TUNE_BASE defaults.
- v103 = v102 + proj s56 g24z (EZ/ZN zero-ahead chain qkv->o->pooler->unpool at T=16; thanks proj): `git apply` of r-proj
  dev/s56/g24z_layer.patch@a3c49ba, clean. Mixed tuner picks (e.g. the H200's lt:qkv/lt:o at T=16 sl) are safe: a g24
  zeroes the next out only if _g24_sel says the next key also runs on g24; else the consumer zeroes itself as before.
  H100 ab_flag 4 rounds x SEEDS 7,11,99 (tuner on), G24Z {16} vs off: d16 sl 157.3/150.8/188.8 vs 157.7/152.2/191.2
  (-0.4/-1.4/-2.4), full 270.0/297.9/291.3 vs 270.2/298.6/291.7 (-0.2..-0.7). Cold compile+prepare+test 310 s, 13/13;
  replay_test d16 seeds 11,23 0/80; route_judge d16 0/40. (A first warm-cache run printed "ninja: build stopped" yet
  passed 13/13 with g24 built: most likely a race with another agent's build in the shared default TORCH_EXTENSIONS_DIR;
  not reproducible cold.)
- v16/v32 sliding vs 703d3a7e (+4.7/+4.8 us in c973ab6f; the H200 tuner defaults also moved 269.6 -> 274.4 / 336.6 -> 342.3
  across runs): NOT NORM_EARLY: H100 ab_flag 4 rounds x SEEDS 7,11, NE on vs off: v16 sl 307.0/306.3 vs 308.2/307.3,
  v32 sl 346.3/337.8 vs 349.6/342.0 (NE wins 1..4 us); other sliding shapes tie. Most likely judge noise (1.8% per shape).
- 00:37Z check 0eb3484a = v103 (174a655) queued; submit 129ad449 (v102) queued.
- 00:27Z **submit 129ad449 = v102 (code of check c973ab6f): 2.0398x [2.009, 2.070], 11,089 us/step: NEW BOARD BEST** (was 703d3a7e
  1.9954x); layer board #39 at 00:52Z (#1 2.380x). d1 70.2/75.5, d16 132.6/216.6, d32 161.8/319.1, v1 116.4/129.3,
  v16 273.3/367.3, v32 359.6/527.8. vs its check: d1 sl +4.7, v1 +7.5/+13.4 (T=1/8 routing draw: local expert count), rest +-2.
- 00:37Z **check 0eb3484a = v103 (174a655): 2.065x [2.034, 2.096], 10,954 us**: d1 68.6/75.3, d16 127.7/214.3 (sl -4.6 vs c973ab6f),
  d32 177.1/333.5 (sl +13.5: T=32 code identical to v102; the H200 tuner's own T=32 default was 169.1 vs 168.8 us: run noise),
  v1 110.7/122.5, v16 279.8/368.2, v32 358.9/524.2. H200 tuner g24z off: T=16 sl 172.8 vs 170.6, full 238.2 vs 237.0 (g24z wins).
  H200 T=32 picks flipped between runs ({tk2h 4, lt:o} -1.4/-2.6 in c973ab6f, +3.2/+4.9 in 0eb3484a): keep it tuner-only.
- 01:01Z submit b9685e03 = v103 (check beat board 2.0398).
- 01:01Z **submit b9685e03 = v103: 2.0606x [2.030, 2.092], 10,977 us/step: NEW BOARD BEST, layer board #30** (01:27Z; #1 2.380x).
  d1 70.3/77.0, d16 133.4/216.1, d32 164.7/320.7, v1 114.0/122.9, v16 271.3/360.5, v32 352.9/520.7.
- Budget 10-27 used: 2 checks, 2 submits (18 / 8 left).

## s68 (2026-10-27 01:28Z-, sandbox okc-night4, session 17): cold prepare 314 -> 111 s (lead 00:48Z); v105 = + moe n14 TK2F_SQ
- Cold timing (/tmp/s68/prep_time.py, fresh TORCH_EXTENSIONS_DIR + TRITON_CACHE_DIR, H100, 128 cores) of v103: import 96.8 s
  (attention csl/nvs/nds load_inline at import, serial) + prepare 217.3 s (moe lib 84.4, g10 37.7, g24 34.8, lt 34.6, tuner 25.3).
- v104: every JIT build on one ThreadPoolExecutor(8) (`_BUILD_POOL`, layer.py top): attention's `_load_inline` is replaced in
  build.py by a submit that returns `_LazyBuild`; the layer's builds (_moe__lib, _g10_lib, and the new _lt_build/_g22_build/
  _g24_build split out of their prepare functions) are submitted at module end (`_PREBUILD`); prepare() first calls
  `_join_builds()` (waits, resolves _CSL/_NVS/_NDS to modules or None = Triton fallback, pops failed layer libs so they are
  retried serially: seen working when a build error hit the pool). Cold: import 1.2 s, prepare 111.5 s (join 87 = the moe
  build; tuner 24), wall 114 s. Kernels/sources unchanged (same extension hashes). test 13/13, replay 0/96, all 4 libs + csl/nvs/nds real modules.
  NOTE the judge's core count is unknown: with few cores the parallel builds still overlap (each is ~1 nvcc + 1 g++).
- moe n15's twm K-split clamp fix: NOT ported (our twf_split 1022 runs as the fixed 66 split, which moe found faster than real AUTO).
- v105 = v104 + moe n14 TK2F_SQ (r-moe 01326b4 recipe tools/n14/sq_layer.md, hand-ported to our older moe_src: tk2f_produce SQ
  template, PF | 512, cfg + 10000000 for SA 84204 and BAL (PF 36|512) only, `_sq_pack` in prepare, `_tk2f_w` helper at the moe
  kernel_fn launch; T=1 n2u site behind TK2F_SQ_T1 = False). Bitwise vs off at the expert level (/tmp/s68/sq_eq2.py, M 1/8/16/32,
  BAL on/off, 8 skewed routings): 0/192 (whole-layer outputs are NOT bitwise run-to-run anyway: split-K atomics).
  H100 ab_flag 4 rounds x SEEDS 7,11,99, SQ on vs off: d1 sl +0.7/+0.9/+0.7, full +0.9/+0.5/+0.6 (=> off at T=1);
  d16 +0.4/+0.1/+0.6 sl, +0.6/+0.2/-1.5 full; d32 +-0.5 (H100 DRAM-bound; moe's H200 diag: d32 -1.5, d16 -0.55).
  tk2f_stress 0/600, test 13/13, replay decode seeds 11,23 0/96, cold prepare 111.4 s.
- 02:00Z **check 18504372 = v105 (f9b702a): 2.0871x [2.056, 2.118], 10,838 us** (best check so far; board 2.0606x). H200 us:
  d1 67.6/76.6, d16 128.6/204.7, d32 163.1/312.0, v1 111.3/123.7, v16 272.5/355.1, v32 353.3/528.1 (vs check 0eb3484a v103:
  d16 full -9.6, d32 sl -14 / full -21.5 (0eb3484a's d32 was a slow draw), v16 full -13; d1/v1 within noise). Judge run 284 s
  start->finish (parallel builds fine on the judge; no build-failure lines in the log).
- 02:05Z submit acce7f04 = v105 (check beat the board).
- 02:25Z **submit acce7f04 = v105: 2.0525x [2.022, 2.083], 11,021 us** (below board best b9685e03 2.0606x; check->submit -1.7%,
  the usual spread): d1 69.0/77.5, d16 135.7/212.8, d32 171.2/317.4, v1 114.8/129.8, v16 268.4/364.6, v32 348.5/531.8.
  Board: layer #34 at 2.061x (#1 colinlikescode 2.380x). Budget 10-27 after s68: 17 checks / 7 submits left.

## s69 (2026-10-27 02:20Z-, session 18): v106 = v105 + proj s56 g24cfg + proj s57 g27 + moe n16 tk2h on SQ
- proj s56 dev/s56/g24cfg_h200_layer.patch (pooler T=16 g24 (4,3,12); H200 standalone 15.4 vs 18.3 us) and proj s57
  dev/s57/g27_layer.patch (g27 for the T=16 o; H200 5.25 vs g24 6.43 us): both `git apply` clean on v105. g27's load_inline
  moved out of _g27_prepare into `_g27_build` on _BUILD_POOL (_PREBUILD), _G27 added to _join_builds' retry list.
- moe n16 tools/n16/hsq_layer.md hand-ported: moe_tk2f_launch SQ branch dispatches tk2h (14W004) to the PF 4|512 instantiation
  (BAL flag stripped before, ignored by tk2h as on the 3-copy path); _tk2f_w maps 14W004 to +10000000 / w13sq,w2sq.
  Bitwise vs 3-copy (/tmp/s69/hsq_eq.py = sq_eq2 with TK2H_M {16,32}: W3 and W2): 0/96 + 0/96.
- TUNE_BASE (32, *) += tk2h 4 (moe n13 -1.9, n16 -1.7 us on H200, moe-only); TUNE[32] candidates {tk2h 0}, {tk2h 3},
  {tk2h 0, lt:o} so the tuner can revert; tk2h dropped from TUNE[16] (moe n13: loses +3..5 us on H200).
- H100 ab_files 4 rounds v105 vs v106 (tuner on): d16 sl 192.1 -> 191.3, full 292.8 -> 291.0; d32 sl 192.2 -> 195.8 (+3.6,
  H100 DRAM-bound tk2h; H200 decides via the tuner line), full 411.4 -> 409.1.
- Tests: bench test 13/13 (twice), replay decode seeds 11,23 0/160 + d32 0/20, cold prepare (fresh caches) 115.0 s (join 89.9, tune 24.4).
- 02:31Z **check 71f451bb = v106 (2c64a0a): 2.0726x [2.042, 2.104], 10,914 us** (board 2.0606, best check v105 2.0871):
  d1 67.4/76.1, d16 132.3/209.9, d32 171.5/316.5, v1 115.3/125.2, v16 285.3/361.9, v32 341.9/514.7 (v16/v32 code unchanged
  vs v105: -13..+13 us per shape = run noise). H200 tuner: T=32 sl default (tk2h 4) 172.4 vs {tk2h 0} 169.4, {tk2h 3} 180.4;
  full 348.3 vs 348.2 / 357.6 => tk2h W3 SQ does NOT win in the layer (moe-only -1.7): back to a tuner option (v107).
  T=16: {g27:o False} 177.1 vs default 177.1 sl, 236.6 vs 236.4 full (g27 ties in-layer); g24 all-off +1.7/+3.4 (g24 holds).
  {tk2fj False} looked -1 us on all 4 lines here but flips sign across the last 5 runs (noise).
- 02:47Z **check 7fda724c = v107 (tk2h back to tuner-only): 2.0424x [2.012, 2.073], 11,075 us**: d1 68.6/77.2, d16 128.1/213.3,
  d32 176.3/325.8, v1 114.5/121.4, v16 275.7/365.4, v32 363.7/539.0. Slow run overall (H200 tuner T=32 sl default 190.1 us vs
  172.4 in 71f451bb for near-identical code): run-to-run machine noise ~5-10% on T=32. Tuner: {tk2h 4} 200.7 vs 190.1 sl,
  357.8 vs 347.8 full (W3 loses in-layer in both checks), {tk2h 3} tie this time (+8..10 last time); g27 off tie again.
- 02:55Z submit of v106 (2c64a0a; its check 71f451bb 2.0726 beat the board 2.0606; v107 differs only by the T=32 tk2h default).
- 03:02Z **submit 08b0704a = v106 (2c64a0a): 2.0678x [2.037, 2.099], 10,939 us/step: NEW BOARD BEST, layer board #25** (#1 colinlikescode
  2.380x). d1 68.7/76.4, d16 124.1/210.7, d32 162.1/314.6, v1 111.0/121.9, v16 269.9/361.9, v32 353.1/538.0.
  Budget 10-27 after s69: 15 checks / 6 submits left. main = v107 (tk2h tuner-only); v106 vs v107 differ only in the T=32 default.

## s70 (2026-10-27 03:14Z-, session 19): v108 = v107 + proj s58 g27mc + moe n17 TK2G_PAR4
- `git apply -p1 r-proj/muse_bf16_proj/dev/s58/g27mc_inc_layer.patch` (clean on v107): g27_src.cu G27_MC 1, G27_CFG ("o", 32),
  TUNE[32] += {g27:o True}. moe n17 par4.diff hand-applied to moe_src.py (hunk 1 anchor differed; 3 edits, tk2g_bound only).
- H100 ab_files 4 rounds v107 vs v108 (tuner on): d1 85.6/104.6 -> 85.4/104.4, d16 190.3/293.0 -> 189.2/290.5,
  d32 193.7/409.9 -> 190.6/407.9.
- Tests: bench test 13/13; replay d16 seeds 11,23 0/80; d32 with g27:o + tk2h 4 forced (TUNE = {}) 0/80; cold prepare 137 s.
- 03:21Z **check 56d4f73b = v108 (85b5641): 2.1190x [2.087, 2.151], 10,675 us** (best check so far): d1 69.4/75.5, d16 127.1/210.3,
  d32 170.7/291.2, v1 110.9/120.7, v16 273.6/357.7, v32 341.5/521.2. H200 tuner T=32 picked {tk2h 4, lt:o} (confirm sl 170.8 vs
  167.3, full 330.7 vs 325.9; PAR4 makes tk2h competitive in-layer); T=16 {tk2fj False} (near tie); g27:o at T=32 tie (172.6 / 331.2).
- 03:34Z **submit 19ebde23 = v108: 2.0860x [2.055, 2.117], 10,844 us: NEW BOARD BEST (layer #34; #1 2.380x)**: d1 67.7/75.8,
  d16 127.9/208.9, d32 173.3/326.8, v1 106.8/121.8, v16 274.6/368.5, v32 335.7/500.0. Its tuner did NOT pick tk2h at T=32
  ({tk2h 4, lt:o} sl 176.1 vs 175.5, full 332.6 vs 332.6; {tk2h 4} alone +11/-0.3): the T=32 tuner picks are near-ties that flip
  between runs; d32 full 291 (check) vs 327 (submit) is run noise, not a pick (tuner lines differ <= 5 us).
- Budget 10-27 after s70: 14 checks / 5 submits left.

## s71 (2026-10-27 03:46Z-, session 20): v109 = v108 + proj s59 G24X + moe n18 TK2F_L2H mode 4
- `git apply -p1 r-proj/muse_bf16_proj/dev/s59/g24x_layer.patch` (clean on v108); moe n18 l2h4_layer.md hand-applied to
  moe_src.py (#define TK2F_L2H 4 + nfin = multi-group expert; cache policy only, bitwise).
- Tests: bench test 13/13; replay_test decode seeds 11,23 20 inputs: 0/240 (all 6 decode shapes).
- H100 ab_files 4 rounds v108 vs v109 (tuner on): d1 85.9/105.5 -> 85.7/104.5, d16 189.3/290.6 -> 187.1/284.9,
  d32 189.0/406.1 -> 189.4/406.2 (tie).
- 03:54Z **check c3bab7fd = v109 (003a510): 2.0822x [2.051, 2.113], 10,864 us** (board 2.0860 -> NOT submitted): d1 67.5/75.2,
  d16 128.8/210.7, d32 166.4/316.6, v1 111.5/127.5, v16 273.0/372.3, v32 354.1/533.0. H200 tuner T=16: {g24x False} +1.3 sl /
  +1.0 full (G24X holds on H200); T=32 picked {tk2h 4} sl, {tk2h 4, lt:o} full again. v16/v32 code unchanged vs v108: their
  -0..+33 us moves are run noise. Budget 10-27 after s71: 13 checks / 5 submits.

## s72 (2026-10-27 04:07Z-, session 21): v110 = v109 + QSS at verify sliding T=128 (TUNE option); probes
- No new Handoffs (moe n19 WBW dead; proj s61 diag only). Board: #34 at 2.0860x (#1 colinlikescode 2.380x, 9,505 us).
- H200 per-op rates from profile c3bab7fd (bytes / kernel time): d32 sl tk2f 283 MB in 71.7 us = 3.95 TB/s (at HBM);
  d1 g13 qkv 28 MB 10.8 us = 2.6 TB/s (first op), pr1 58 MB 17.5 = 3.3; v16 sl qkv lt (tst_48x64) 37.7 MB 16.9 = 2.2,
  unpool ~21 = 3.2, pooler 26.4 beside pr3 router. H100 F.linear standalone (cold): n4096 k8192 m16-128 26.8-27.2 us
  (2.5 TB/s, flat in m), n2304 k8192 m128 20.0, m256 22.4; bmm split-K 2 slabs m128 16.7 (+torch sum 22.1), S4 17.0;
  m256 S2 21.5, S4 19.6 (/tmp/s72/gemm.py, qkv.py).
- DEAD (oracle, H100 /tmp/s72/orc.py, 3 rounds x seeds 7,11): moe_norm2/3 replaced by a bare cnt.zero_() (wrong
  output, timing bound) at d16/d32 = SLOWER +1.5..3.0 us (d16 sl 154.1 -> 156.5, d32 sl 199.1 -> 201.3): norm2 itself
  is ~free, its value is the unpool's PDL pre-wait window. A norm2 -> unpool-prologue fusion has nothing to win.
- v110: QSS (2 FP32 split-K qkv slabs from one bmm, summed in FP32 + rounded to BF16 at load) extended to verify
  sliding nvs (attn_src `_NVS_SRC`: consumer/nvs templated on QT, qld8/qld4 helpers as nds; host `nvs(..., qss)`)
  + `_combine_hb` QSS=; layer `_kernel_fn` QSS condition adds length 8 / nvs (QSS_VERIFY). H100 timeline v16 sl:
  qkv bmm 18.4 vs lt 19.4 us, but nvs 12.8 vs 11.2 (FP32 slab query loads on its critical start): ab_flag 3 rounds x
  seeds 7,11: v16 sl -0.9/+0.0, v32 sl +2.0/+1.5 (full untouched). Default off; TUNE[128] += {qss: True} (H100 tuner
  349.5 vs 352.2 default, noise). Tests 13/13, scored verify x3 forced 0 bad, replay verify 16 x seeds 11,23 forced
  0/40. Cold prepare (fresh caches) 115.7 s (join 90.5, tune 24.5).
- Posted to proj (Requests): hybrid pooler/unpool at T=16/32 (part of the rows BF16 on some SMs = HBM-bound, the rest
  12-bit g24 = issue-bound), model -1..-3 us per GEMM on H200; needs the G72 proxy or an H200 diag.
- 04:37Z **check 0f12113f = v110 (9937f2e): 2.0766x [2.045, 2.108], 10,893 us** (board 2.0860 -> NOT submitted): d1 68.9/76.4,
  d16 131.2/215.0, d32 162.8/317.5, v1 114.6/124.3, v16 272.5/359.0, v32 355.8/539.3. H200 tuner T=128: {qss: True} sl
  306.4 vs default 306.0, full 368.6 vs 368.3 -> QSS at verify is DEAD on the H200 too; candidate removed (v111 = v110 minus
  the TUNE entry; code path kept, off). Budget 10-27 after s72: 12 checks / 5 submits.
- Layer-side idea list is exhausted on the measured terms (see STATE dead ends); the remaining levers are moe's experts and
  proj's T=16/32 g24 (consumer-bound at m32) and T=128/256 cuBLAS GEMMs.

## s73 (2026-10-27 04:51Z-05:10Z, night4 session 22): moe n20 S6 ported, H200-dead -> off; no judge runs
- Ported r-moe 4a87466 S6 (6-bit block-scale codes c = e4m3 byte - 80 in the SQ stage images, 17920 vs 18432 B/stage,
  s6dec = PRMT + LOP3) into moe_src.py: tk2f_produce/consume, tk2g_row, tk2h_consume, launch cfg + 20000000 (84204 (+BAL),
  tk2h GS16 PF4), `_s6_pack`, `_s6_unpack` + prepare()-time round-trip assert (lossless), falls back to 8-bit SQ if any
  scale byte is outside [80, 0x7E]. layer.py's NU8 anchor string updated (SB line now `(PF & 1024) ? 1120 : 1152`).
- Tests: bench test 13/13, tk2f_stress 0/600, tk2h_stress W3 0/400, tools/s6_eq.py (S6 vs SQ bitwise, variants/s6keep.py =
  submission with _moe_TK2F_S6_KEEPSQ = True) 0/480. Cold prepare (fresh caches) 123.9 s.
- H100 ab_flag 4 rounds x seeds 11,23 (S6 on/off): d16 sl -1.0/-2.1, full -1.3/-1.0; d32 sl -2.8/-1.9, full 0/-2.3 us;
  d1/v1/verify tie (T=8 runs the NU8 85204 path, not SQ; T=1 not SQ).
- moe's H200 diag 13b528a8 (moe-only, layer routings, 5 seeds, maxdiff 0): tk2f d16 65.7/65.4 (sq) vs 66.4/66.4 (s6),
  d32 86.1/86.0 vs 89.0/89.0, tk2h W3 d32 82.4 vs 84.0, d1 tie, v1 -0.8 (not used by the layer). => S6 LOSES on the H200
  (issue-bound: the extra unpack instructions cost more than 2.8% fewer bytes). main v113 = TK2F_S6 = 0 (= v111 behavior).
- Looked for other layer-side work: H200 profile 0f12113f d1 sl = tk2f 27.4 | pr1 18.5 | g13 qkv 10.8 | bnd 2x3.2 | o 6.2 |
  n2u 5.4 | csl 4.0; everything left is on the dead-end list except the T=1 o+bnd+pr1 megakernel (~-3 us bound).
- v114 = v113 + proj s63 G24P (pooler g24 EZ at T=32, persistent zero buffer re-zeroed by the unpool; zeroing is after
  g24's griddepcontrol.wait, so the rmsnorm reading pooled has completed) as TUNE[32] options; also moved `_SLIDING[0] = ...`
  before the G24X/G24P `_opt` reads (proj's find: they read the previous call's flag). Tests 13/13, replay d32 forced
  seeds 11,23 0/120, zero invariant eager+replay ok, cold prepare 120.2 s. H100 tuner rejects g24p (+1..2 us).
- CHECK c755dcc3 v114 (aa08453) 2.0578x (not submitted, < 2.0860): d1 67.9/76.1, d16 124.1/210.7, d32 161.2/318.5,
  v1 111.4/122.1, v16 268.2/359.6, v32 352.6/517.8. H200 tuner T=32: sl g24p 168.6 | +tk2h4 166.4 | +lt:o 166.8 vs tk2h4
  165.2 / +lt:o 165.1 (picked {tk2h 4, lt:o}); full g24p 336.0 x3 vs default 334.1, bal 333.1 (picked bal). G24P DEAD on
  H200 too -> v115 drops the g24p TUNE options (code kept, off).

## s74 (2026-10-27 05:19Z-05:50Z, night4 session 23): PR1_BND + BND_SPLIT as H200 tuner diag -> both dead; main v117 = v111 behavior
- Lead 05:07Z: per-load H200 us vs floor table now in STATE (gap x score weight is 4-6% sliding / 2-2.7% full on every load;
  experts are 29-53% of each load's time; _gd_full d16 87.5 / d32 167.6 us is at its ~4.0 TB/s gather floor per ~/attn STATE).
- Noted: v108's check 56d4f73b had d32 full 291 us vs 317-327 in every other run of v108-v116 (the judge's inputs/routing
  vary per run; synthetic tuner d32 full ~331 in all): single-shape noise up to ~10%.
- PR1_BND (s11, the first boundary fused into pr1 at T=1, never H200-timed) as TUNE[1] option pr1bnd (+ with g13o).
  H100 ab_flag real inputs 4 rounds x seeds 99,7,13: d1 sl +1.8, full +1.3 us; + pr1_npf 2/4/8: +4..8 us (pre-wait L2
  prefetch fights the o GEMV's stream). H100 timeline: separate path o end -> pr1 end 21.4 us (bnd 4.3 + pr1 17.1),
  fused 22.5 (the per-block prologue = br L2 hot-spot read by 132 blocks + 2 block reductions, not overlapped with W).
- BND_SPLIT (s33, H100-picked) as TUNE options bspl1 / bspl2 at T=1/8/16/32: H100 d1 1/2/4/8 all within +-0.4 us.
  Tests: forced bnd 13/13, forced SPL 8 13/13 (files_eq vs main: d1 sl bitwise, T>=8 the usual atomics noise, ids same),
  replay d1/d16 bnd 2 seeds x 40 0 fails; cold prepare 122.6 s.
- CHECK 4bf626fa v116 (a9d8cb2) 2.0841x (not submitted, < 2.0860): d1 67.2/76.3, d16 131.4/207.0, d32 166.3/318.7,
  v1 113.9/124.7, v16 278.5/368.6, v32 362.5/527.7. H200 tuner: pr1bnd +0.6/+0.6, with g13o +1.4/+0.5 us (DEAD on H200
  too); bspl all within -1.0..+1.2 except T=16 full bspl1 8 +9.4 (noise; T=8 full picked bspl2 4 at -0.3 confirm).
- v117: TUNE entries removed again (the `_opt` hooks stay: pr1bnd / bspl1 / bspl2), = v111 behavior. Tests 13/13.
- Remaining layer-own idea with a nonzero bound: T=1 o+bnd+pr1 megakernel (~-3 us d1). The boundary itself is at its
  latency floor (PDL release + 2 reductions; 4 programs already), so a cluster/DSMEM boundary would save little.

## s75 (2026-10-27 05:44Z-, night4 session 24): moe n21 twq K-split ported as TUNE[128] options (lead 05:38Z top priority)
- moe_src.py patched with r-moe e7e0123 muse_moe_w4a16/tools/n21/patch_layer_twq.py (clean apply on v117's moe_src; twq flags
  q | items << 8 | log2 parts << 16; partials allocated per call, tickets = st["tk"] shared with twf, self-resetting).
- layer.py TUNE[128] += {twq 73988} (4|33<<8|1<<16), {twq 73996} (12|...), {twq 73988, fr}, {twq 73988, lt:o, lt:unpool}.
  H100 tuner never picks them (twq loses on H100 in-layer, as for plain twq 4 since s32); the H200 decides.
- Correctness (forced via variants/v118_f73988.py / _f73996.py, prepare() override of state["tune"][(128, *)]): scored verify 16
  3 seeds both types 0 bad (73996: 2 seeds 0 bad), replay_test verify 16 seeds 11,23 x 20 = 0/80; bench test 13/13.
  NOTE: tools filters are substrings of "verify 16 sliding", NOT "v16" (a "v16" filter silently tests nothing: "bad 0").
- Cold prepare (fresh caches) 152.1 s (join 93.6, tune ~55; shared GPU), <= 200 OK.
- 05:54Z CHECK a0bcbbf4 v118 (a90cadb) 2.0866x [2.055, 2.118], 10,841 us (board 2.0860: +0.0006 = noise; behavior = v111 since
  the tuner picked no twq option -> NOT submitted, would be a lucky-run resubmit): d1 67.6/77.5, d16 129.0/212.1, d32 159.5/316.3,
  v1 103.8/114.1, v16 269.3/359.5, v32 351.8/522.5. H200 tuner T=128 sl: default 287.2 | twq4 292.2 | twq73988 298.2 | 73996 297.6 |
  +fr 296.2 | fr 281.6 (picked fr); full: twq73988 383.3 vs default 386.6, picked fr+twf_split 66 (379.0). The split LOSES in the
  synthetic-routing tuner (moe-only real routings said -10.6). Previous check 4bf626fa: twq4 278.1 vs default 284.9 (picked).
- v119: TUNE_BASE (128, True) = {twq 73988} (forced under every non-twq pick) as a real-input diag check.
- 06:10Z CHECK 2f7349ee v119 (476fe00) 2.0855x [2.054, 2.117], 10,847 us (not submitted): d1 67.0/77.0, d16 123.9/205.4,
  d32 157.3/315.7, v1 111.5/121.2, v16 **282.2**/368.1, v32 366.0/537.1. v16 sl ratio 1.888 vs 1.92-1.93 in the last 5 runs
  (SGLang 532.9 this run, +2.5%: slow run, but the ratio still drops ~2% = ~+6 us). Tuner T=128 sl (base = split): default 269.1,
  twq73988 270.6 (same thing), fr 268.7 picked; full: twq73988 381.2 vs default 365.2 (+16).
  => moe n21 twq split DEAD IN THE LAYER on the H200 (both judge runs; the moe-only -10.6 us with layer routings does not
  transfer: in-layer twq runs beside / after the side-stream kernels, unlike the moe-only proxy). Told moe (their HOTQ builds on it).
- v120: TUNE options + base removed (= v117 behavior); moe_src keeps the n21 code (flags without split bits = old twq). Tests 13/13.

## s76 (2026-10-27 06:33Z-, night4 session 25): v121 = moe s23 HOTQ as TUNE[128] options + proj s66 T=32 combos
- Port: `python ~/r-moe/muse_moe_w4a16/tools/s23/patch_layer_hotq.py moe_src.py` (clean on v120's moe_src). HOTQ = moe_hot<NH 64>
  (33..64-token experts, one N 64 pass) on `grid` CTAs launched before moe_twq, twq (flags bit 20 hx) skips those experts, no wait at
  entry, waits at its end. Layer value: twq flags | grid << 24. TUNE[128] += {twq 536953348} (4|66<<8|1<<16|32<<24 = n21 split 66 +
  hot 32; moe-only H200 v16 191.1 vs split 201.5 vs plain twq ~210), {twq 536870916} (hot 32, no split), {twq 536953348, fr}.
  g_hot is reset by the twm path (moe_set_hot(0/1)) each call, so T=128 HOTQ (sets 4) cannot leak into T=256.
- proj s66 tune32_stack_layer.patch applied: TUNE[32] += {tk2h 4, g24:qkv_gate}, {tk2h 4, lt:o, g24:qkv_gate}.
- Tests: forced (variants/v121_f536953348.py / _f536870916.py, prepare() override) scored verify 16 x3 0 bad each; replay verify 16
  seed 23 0/40 each (seed 11 lines cut by my tail); bench test 13/13; cold prepare 122.4 s (join 96.8, tune 24.9).
- H100 tuner (shared GPU) T=128 sl: 536953348 346.7 / 536870916 355.9 vs default ~353 (picked twf_split 66 at 336.8); full 447/445 vs 434.
- 07:11Z CHECK baae0e41 v121 (34a1bdc) 2.0772x [2.046, 2.108], 10,890 us (not submitted, < 2.0860): d1 67.2/76.9, d16 126.9/208.0,
  d32 167.8/321.4, v1 109.0/121.3, v16 277.3/377.4, v32 372.3/538.9. H200 tuner:
  T=128 sl default 268.7 | twq 4 271.2 | hot32 (536870916) 271.7 | split66+hot32 (536953348) 277.6 | +fr 276.9 -> picked fr 267.4;
  full 392.7 | 395.5 | 396.7 | 392.8 | 393.0 -> {lt:o, lt:unpool}. => HOTQ DEAD in-layer (hot alone = twq 4, split +6 as in s75).
  T=32 sl default 176.3 | g24:qkv_gate 175.7 | {tk2h 4, g24:qkv_gate} 174.9 | {tk2h 4, lt:o, g24:qkv_gate} 174.5 PICKED (confirm 176.1 vs
  175.0); full 340.1 | 339.0 | 337.3 | 337.0 PICKED (confirm 340.3 vs 336.3). proj s66 combos do stack on the H200 (-1.1 sl / -4 full tuner).
- moe 07:25Z diagnosis (COORD Requests): the layer's T=128 default is moe_twm; twq ~+5 us in-layer over 16 judge runs; n21's split code
  slowed every twq item. => v122: every twq option removed from TUNE[128] (HOTQ code stays in moe_src, unreachable).
- v122 also = + proj s69 g24sk2 (SK g24 (4,3,12) at T=32 pooler/unpool; r-proj dev/s69/g24sk2_layer.patch, clean on 34a1bdc) as TUNE[32]
  options, + one more combo {g24sk2 both, tk2h 4, lt:o, g24:qkv_gate}.
- v122 tests: forced (variants/v122_fsk2.py) scored decode 32 x3 0 bad, replay decode 32 seeds 11,23 0/80; bench test 13/13; cold prepare 114.5 s.
- 07:26Z CHECK 1b9a63fb v122 (5a986d9) 2.081x [2.050, 2.112], 10,870 us (not submitted, < 2.0860): d1 67.7/78.6, d16 128.2/203.6,
  d32 **160.5/308.0** (best d32 yet; board 173.3/326.8), v1 116.0/121.5 (v1 sl +7 = noise, code unchanged), v16 281.2/370.9,
  v32 361.4/518.3. H200 tuner T=32 sl: default 174.2 | {tk2h 4, g24:qkv_gate} 169.7 | +lt:o 169.6 | g24sk2 pool/unp/both 174.0/173.2/174.0 |
  sk2 both + combo 169.6 PICKED; full 330.9 | 325.6 | 325.3 PICKED | sk2 331.2/330.6/331.9 | sk2+combo 325.7. => g24sk2 tie (dead on
  its own), the s66 combo wins again (-4.5 / -5.3 us). T=128 picked fr both types.
- v123: TUNE_BASE[(32, *)] += {tk2h 4, g24:qkv_gate} (2 H200 tuners agreed, s35 policy); TUNE[32] += {tk2h 0, g24:qkv_gate False}
  (= old base). Tests: forced tune {} at T=32 (variants/v123_fbase.py = the new base) scored decode 32 x3 0 bad, replay seed 11 0/40;
  bench test 13/13. Not checked (behavior = what v122's tuner picked; check it with the next real win stacked).

## s77 (2026-10-27 07:42Z-, night4 session 26): v124 = TUNE_BASE stacking (proj s66 tune_stats over the last 10 H200 logs)
- The tuner keeps ONE option set per (T, type), so consistent H200 singles never stack. `python ~/r-proj/muse_bf16_proj/dev/s66/tune_stats.py
  profiles 10 1,8,16,32,128,256` (mean delta vs default < -0.4 us, >= 4 logs): T=1 sl g13o -0.49 (9/9), pr1_rb 32 -0.50 (8/9),
  full g13o -0.41 (8/9); T=8 sl g14alt:qkv_gate 3 -1.26 (10/10), cslpw -0.66 (10/10); T=16 sl tk2fj False -0.74 (10/10);
  T=32 = v123 base; T=128 fr sl -1.89 (9/10) full -1.26 (10/10), + lt:o/lt:unpool sl -0.88 (9/10); T=256 full rg -1.87 (8/9),
  rg+twf_split 66 -6.3 (6/9, bimodal).
- v124: TUNE_BASE (1, *) = {g13o, pr1_rb 32}; (8, sl) += g14alt:qkv_gate 3; (16, sl) += tk2fj False; (128, *) = {fr}. Every option now
  stacks on these (cslpw at T=8, lt:* at T=128, ...). Options added: T=1 {g13o False} (+rb20 / +kv_pf False), T=8 {cslpw, g14alt 3/2},
  T=128 {fr False}, {twf_split 66, lt:o, lt:unpool}, {gnorm False, lt:unpool}; T=256 rg+twf_split 66 + lt:o / lt:unpool / all lt.
- Tests (base forced, variants/v124_fbase.py = tune {} everywhere): scored_test all 12 shapes x 2 seeds 0 bad; replay_test seeds 11,23 x 20
  inputs 0/480; bench test 13/13. tools/opt_test.py (new: every TUNE option set forced on top of the base, eager, 1 seed per shape):
  0 bad of 300 (all 12 shapes). Cold prepare 120.5 s (join 96.6, tune 23.1). H100 tuner T=256 picked the new rg+twf_split stacks (sl 392.8 vs 403.6 confirm).
- 08:01Z CHECK 08d10234 v124 (701bf12) 2.0832x [2.052, 2.114], 10,858 us (not submitted, < 2.0860; tie = noise): d1 68.4/77.0,
  d16 132.0/207.9, d32 162.5/321.6, v1 110.7/123.4, v16 285.8/360.3 (SGLang 555 sl: slow run), v32 345.1/518.6.
  H200 tuner picks (on the new base): T=1 sl ~default, full kv_pf False; T=8 sl cslpw (-1.4), full cslpw+g14alt 3 (-0.4);
  T=16 sl default, full bal; T=32 sl tk2fj False, full bal+lt:o; T=128 sl twm_order 76 (-2.7), full {twf_split 66, lt:o, lt:unpool}
  (-8.3 confirm); T=256 sl rg, full rg+split1022+all lt (-12 confirm). Stacking makes the picks move but the total is flat:
  the tuner-level gains (1-3 us per layer type) stay below judge noise (+-1%).

## s78 (2026-10-27 08:23Z-09:05Z, night4 session 27): re-bound of the layer-own structural items; no build, no judge runs
- Score weight check: sum over the 12 scored shapes of %/us (STATE table) = 0.68 % per us saved on EVERY shape; d1 sl alone 0.177.
- Judge 08d10234 per-run spread (runs_ns, 60 inputs/shape): d1 sl 59.1..84.5 us (median 71.0; routing = 0..3 local experts),
  d32 sl 154..187, v1 sl 92..134. Per-run tk2f cost follows the routed expert count; nothing layer-side in the spread.
- H100 timelines now (main v124, seed 99): d1 sl qkv g13 0-12.8 | csl -19.8 | o g13 -23.7 | bnd -27.0 | pr1 -46.2 | tk2f 46.7-80.2 |
  n2u -84.8 | bnd -86.6. v1 sl g14 0-13.7 | vtk -20.6 | g12 o -27.9 | bnd -32.6 | pr8 -53.9 | tk2f 54.3-138.4 | nu8 -142.4 | bnd -145.2.
  d32 sl qkv g24 0-16.0 | nds+combine -33.4 | o g27 -42.7 | bnd -47.8 | pooler g24 -70.8 (|| router_split+routing s585 -72.4) |
  rmsnorm -74.7 | tk2f 73.7-171.5 | norm2 -175.3 | unpool g24 -196.0 | bnd -198.7.
- Re-bounded the lead's layer items with these + earlier measurements:
  * pooler+router+routing+expert prologue at T<=8: tk2f already starts 0.4-0.5 us after pr1/pr8 end (TK2FX): bound ~0.
  * persistent experts->norm->unpool->boundary at T<=8: tail after tk2f = 6.4 us d1 / 6.8 us v1 (H100); every partial fusion
    of it (N2U_BOUNDARY, n2u8, bndc, BND_SPIN, UPF/UPF2) measured +0..3 us; bound <= 1-2 us d1, ~2-3 v1 (<= ~0.5% total).
  * T=1 o+bnd+pr1 megakernel: its pooler-W preload during o needs spare HBM in the o/csl window, but pr1 npf 4/8 (H200 tuner)
    and G10_PFL2 (H100) both measured that window as NOT spare (o / csl slow by the same amount). What is left is one launch
    (bnd ~2 us) minus a grid barrier (~0.7-1 us): <= ~1 us d1 (~0.25%). Not worth a multi-session rewrite.
  * NEW idea checked on paper, dead: unpool(norm2(z)) = W (g*z) / rms(z) (rms is a per-token scalar) would let the unpool skip
    norm2 / start on partial z, but (a) s72's oracle shows norm2 is free (it hosts the unpool's PDL window) and (b) starting the
    unpool on rows finished by early tk2f CTAs = UPF3's window, where the extra stream slowed tk2f stragglers by +7.8 us (s15).
- Conclusion: the layer's own glue/fusion levers are exhausted at <= ~0.5% each; the gap to #1 (14%) is the expert kernels
  (tk2f at ~4.06 TB/s marginal vs ~4.4 achievable; twm at v16/v32 consumer-bound) and full attention (_gd_full/_gw_verify,
  frozen). Asked the lead (COORD Requests) to redirect the layer lane (proposal: layer takes tk2f at T<=32 off moe's plate
  while moe does twm; or attention full shapes inside the layer).

## s79 (2026-10-27 08:29Z-08:50Z, night4 session 28): no build, no judge runs
- No lead answer yet to the s78 lane question; no new Handoffs; no moe "DIAG for layer" bundle posted (4 of today's 5 layer
  checks stay reserved for moe). proj's GUEST carrier is banned by the lead (08:13Z): never ship it in layer runs either.
- Re-read moe's tk2f record for an unowned lever (moe is on twm since 07:42Z): routing prologue already hidden by TK2FX
  (s27); PSC/BAL (H200: CTAs balanced, barrier wait ~0.5 us); phase-2 "slow" = accounting (moe s28/s30); deeper rings,
  GS16/h16, NC16, S6, RFLAG all dead on the H200. tk2f's dq is already 11 instr / 8 weights with the scale folded into
  one bf16 FMA (bmul); a byte_perm LUT would save ~3 of 11 per 8 weights = moe's twm item (lead 07:42Z (a)): if moe
  lands it for twm, the same dq swap in tk2f_consume / tk2g_row is the layer's port (d16/d32 are consumer-ALU-bound).
- H200 d1 sl profile (08d10234): tk2f 27.6, pr1 18.0, qkv g13 10.8, bnd 2x 3.1, o g13 6.0, n2u 5.5, csl 4.0, rmsnorm 1.3 us.

## s80 (2026-10-27 08:47Z-, night4 session 29): v125 = v124 + moe s27 SCB
- Ported moe s27 SCB (cheaper e4m3 block-scale decode, bitwise per moe; r-moe 40de385) with
  `python ~/r-moe/muse_moe_w4a16/tools/s27m/patch_layer_scb.py muse_decode_layer/moe_src.py` (5 sites, clean on v124).
  moe H200 moe-only diag (5ad98078 vs b14b8be4) on the layer routings: d16 -0.8, d32 -1.3, v16 twm -5.4, v32 twm -4.8 us.
- Tests: bench test 13/13, scored_test all shapes x2 seeds 0 bad, tk2f_stress 300 0/1200, replay_test seeds 11,23 x20 0/480,
  cold prepare 120.3 s (join 96.7, tune 22.9).
- tools/files_eq.py is NOT usable to show bitwise equality with the tuner on: v124 vs v124 itself differs on 11 rows
  (tuner picks differ between prepares; decode 32 sl seed 0 even flips a routing tie). Use a forced-tune variant for that.
- H100 ab_files verify 3 rounds v124 -> v125: v1 142.8/155.7 -> 142.6/155.0, v16 297.0/444.1 -> 295.6/442.0, v32 376.5/614.1 -> 375.7/613.2.
- 08:54Z CHECK 19163546 v125 (2663c2c) 2.0731x [2.042, 2.104], 10,911 us (not submitted, < 2.0860; tie = noise): d1 68.6/77.5,
  d16 126.4/211.3, d32 161.4/**296.3** (best d32 full yet), v1 112.1/123.0, v16 268.0/355.2 (SGLang 506 sl, fast run), v32 354.2/520.8.
  H200 tuner defaults vs v124's check 08d10234: T=1 sl 78.2 -> 70.6 (SCB ~0 there: machine is faster), T=16 186.2/242.9 -> 167.0/230.0,
  T=32 175.4/340.4 -> 163.4/332.8, T=128 285.5/396.3 -> 282.1/362.1, T=256 351.3/564.1 -> 341.6/530.3: a different/faster H200, so
  cross-run tuner numbers cannot isolate SCB's ~1-5 us. SCB stays on main (bitwise per moe, H100 -0.2..-2 us, moe H200 diag -1..-5).
- okc check client was killed by my 25 min background timeout while the job was still queued; the job ran on. Poll `okc jobs` instead.

## s81 (2026-10-27 09:12Z-, night4 session 30): lane (1) per lead 08:55Z = tk2f is layer's kernel now; H200 joint-ceiling DIAG
- Lane start: re-read moe's tk2f record (NOTES-moe n9-n14, n22). In-layer H200 tk2f (profile 19163546: d1 sl 20.6 / full 27.0,
  d16 54.7/57.1, d32 (tk2h) 83.0/87.1, v1 57.1/61.6 us) is already AT moe's H200 stream-probe floor for 18 KB copies (n13
  diag9s sq18: d16 57.5, d32 73.0, v1 60.5; the in-layer kernel starts streaming before its PDL wait, so its duration overlaps
  pr1). sq32/sq64 (contiguous 32-64 KB copies) are 2-6 us lower; the probe's t = 3.8 us + bytes / 4.26 TB/s (sq64 fit over
  d1..v1): the lead's "4.6+ TB/s" is not reachable on these routings, the stream ceiling is ~4.2-4.3 TB/s.
- Paper bound of the lane: tk2f at sq32 speed everywhere = d1 ~2, d16 ~4.4, d32 ~5.6, v1 ~4.7 us per layer = ~2% total if
  fully reached. moe's per-slice fused gemm13->gemm2 idea (no phase barrier) was bounded by moe n11 at ~0.6 us (d1).
- Open H200 question (never measured): with NO consumer math, do 36 KB single-copy stages (KPS 4 SQ) stream faster than
  18 KB ones inside the real tk2f structure (hints on)? H100 says no (DRAM-bound); moe's H200 real-kernel sq402/403 (with
  math) were flat (consumer-limited?). If nm402 << nm204 on the H200, the lever = a consumer that keeps up with 36 KB stages.
- Built (r-layer 9783a61): TK2F_DIAG (python flag in moe_src, False on main; `_moe__lib` turns it into `#define TK2F_DIAG 1`):
  moe_tk2f SQ dispatch cfgs 184402/184403 (KPS 4 SQ NSTG 2/3 on _sq_pack nq 2 buffers st w13sq4/w2sq4), 184214/184412/184413
  (same + PF 256 = no consumer math, wrong output). Layer: TUNE option "tk2x" (1 nm204, 2 sq402, 3 sq403, 4 nm402, 5 nm403,
  6 plain SQ 204), options with "diag": True are timed and printed but NEVER selected (tuner min over non-diag only). In a
  DIAG build TUNE = {T=1/16/32: default + the 6 probes} (T=16 with bal off, T=32 with tk2h 0 + a {tk2h 0} reference).
  Tests: bench test 13/13 on main and on the DIAG copy (variants/diag81.py = sed _moe_TK2F_DIAG = True); cold ~2 min.
- 09:20Z CHECK 93a0990d (DIAG copy of 9783a61, NOT for submit) 2.0734x, 10,910 us (= v125 noise; tuner only had the probes, all
  picks default). H200 'layer tune' whole-layer us (synthetic skewed routing; ref = plain SQ 204 (code 6), T=32 ref = {tk2h 0}):
            default | SQ ref | nm204 | sq402 | sq403 | nm402 | nm403
    T=1  sl  69.4   | 69.8   | 68.3  | 70.6  | 71.9  | 68.4  | 68.0
    T=1  fu  74.3   | 74.6   | 73.0  | 74.2  | 75.2  | 73.2  | 72.3
    T=16 sl 160.8   | 162.3  | 158.0 | 162.7 | 161.8 | 158.2 | 157.8   (probes with bal off; default has bal)
    T=16 fu 247.0   | 248.0  | 246.3 | 249.3 | 248.0 | 245.7 | 244.4
    T=32 sl 163.9   | 166.0  | 163.6 | 179.0?| 166.6 | 163.0 | 160.6   (default = tk2h 4)
    T=32 fu 324.9   | 328.2  | 326.4 | 331.3 | 329.1 | 325.6 | 323.9
  READING: in-layer on the H200 the WHOLE tk2f consumer math costs only 1.5-4.3 us (ref - nm204), and with no math the
  36 KB stages add only 0-3 us more (nm403). Joint ceiling (free consumer + best stages) vs today's default: T=1 -1.4/-2.0,
  T=16 -3.0/-2.6, T=32 -3.3/-1.0 us = ~0.9% total score, unreachable in full (a free consumer does not exist). Real KPS 4
  kernels: tie/worse again (5th time). => tk2f at T<=32 is at its HBM stream rate; the lane is closed at <= ~0.3-0.5% realistic.
  Lead's "4.6+ TB/s" target: not available on this access pattern (moe sq64 fit ~4.26 TB/s + 3.8 us).

## s82 (2026-10-27 09:29Z-, night4 session 31): submit v125 under the lead's 09:28Z rule -> 2.1339x board best
- Rule check (lead 09:28Z): (a) v125 check 19163546 2.0731x >= 2.065x; (b) SCB targets v16/v32 twm: check v16 sl 268.0 us
  (best v16 sl of all our runs; board 274.6), tuner T=128 sl 285.5 -> 282.1 and T=256 351.3 -> 341.6 vs v124's check; v32 sl
  354.2 sits inside its 335.7-372.3 range over the last 15 runs (noise), no other shape out of range; (c) proj's tools/validate.py
  on v125 ALL OK (proj s75: 13/13, scored 0/24, opt 0/300, tk2f 0/1200, replay 0/480, route 0/240, cold 120.4 s).
- Submitted the exact checked file (`git show 2663c2c:muse_decode_layer/submission.py`, md5 3ada35dd...).
- 09:30Z SUBMIT 35dd53d0 v125 = **2.1339x** [2.102, 2.166], 10,600 us, board #24 (was #38 at 2.0860). Per shape: d1 67.4/75.0,
  d16 124.9/204.5, d32 164.3/308.0, v1 112.8/127.3, v16 273.0/352.7, v32 322.0/494.6 (SGLang fast-ish on v16 sl 556.9).
  Same code as check 19163546 (2.0731x): +2.9% between two runs of identical code = judge noise is ~+-1.5% as noted.
- Attention gap (for the lead, COORD Requests s82): layer leaders colinlikescode/pjmullin3 are #1/#2 on muse_attention_decode
  (3.03x, 1242 us vs our 1482 us = 16% less attention time). Attention = 7-27% of our layer loads; 16% faster attention on
  every shape = +2.5% layer score (10% = +1.5%). In-layer H200 attention (profile 19163546) = standalone (gd_full d32 166.6,
  gw_verify v32 190.7, v16 99.2 us). _red_fin/_new_tokens "GPU time" in the layer profile (124/25 us) is PDL wait overlap.
- H100 timelines (main, seed 99) for the attention question: d1 sl g13 qkv 0-12.6 | csl 13.1-19.8 | o g13 14.0-23.7 (PDL) | bnd
  22.8-28.0 | pr1 -46.7 | tk2f 47.1-81.6 | n2u -86.0 | bnd -87.9; v1 sl g14 0-14.0 | vtk 14.4-21.0 | o g12 15.3-28.2 | bnd -32.0 |
  pr8 -53.5 | tk2f 53.9-137.8 | nu8 -142.3 | bnd -145.1. Batch-1 sliding attention exposes <= ~2-3 us per layer (~0.4-0.5%).
- Next session: read the lead's answer to Requests s82 (reopen full attention in the layer's attn_src.py copy, or another lane).

## s83 (2026-10-27 09:41Z-09:55Z, night4 session 32): no lead answer; merged proj validate.py; no judge runs
- Lead has not answered Requests s81/s82 (lane after tk2f; reopen full attention?). Last Human decision = 09:28Z submit rule.
- Merged proj's tools/validate.py + SUB= in tk2f_stress.py (branch proj/validate f625289) into main. `python tools/validate.py --quick`
  on main v125: ALL OK (13/13, scored 0/12, opt 0/300, tk2f 0/300, replay 0/240, route 0/240 gap 0.0256, cold prepare 115.5 s), ~6 min.
  Use it before every check from now on (full mode for submits).
- Board context 09:45Z: layer #24 2.134x (#1 2.380). Sub-boards: moe ours 2.247x (#58) vs #1 2.620; proj ours 1.323x (#278) vs
  colinlikescode 1.637 (but the layer runs its own g13/g14/g19/g24/lt GEMMs, not proj's board file); attention 2.566x (#24) vs 3.03.
- Per-load ratios on board run 35dd53d0: d1 2.54/2.44, d16 2.13/2.14, **d32 sl 1.61**/2.03, v1 2.30/2.16, v16 1.94/1.98, v32 2.24/2.18.
  d32 sl is the outlier (SGLang d32 sl ~= d16 sl, ours +39 us: tk2h +28, g24 pool+unp +13, attn +5). tk2h at T=32 is at stream
  rate (s81 diag), so the bytes are the cost; no new lever found this session (paper only).
- Paper-bounded and dropped: KV L2 prefetch during the T=16/32 full-layer qkv (HBM spare ~1.8 TB/s x 13 us -> <= ~5 us d32 full
  = ~0.1%); expert weight L2 prefetch beside the pooler at T=128 (same shape as s55's dead XPF: needs real ids, crowds router SMs).

## s84 (2026-10-27 09:49Z-, night4 session 33): lane 2 (full attention in-layer): H200 ceilings DIAG + H200-tunable attention plan (v126)
- Lead 09:45Z LANE DECISION 2: layer owns full-layer attention (_gd_full d16/d32, _gw_verify v16/v32) in attn_src.py.
- Read ~/attn record: verify-full consumer mined by 126 sessions (5 native designs s71/s110/s111 lost; FA3 pipelining needs
  >= 112 regs = 1 CTA/SM; FP8 P@V blocked by the transpose + fixed-reference range; MUFU not the limit). exp/hdiag5 (H200
  gather floors) was never run.
- Built (a036793) DIAG probes: attn_src `ATT_DIAG` -> _gd_full DX (11 gathers only after tile 0, 12 all-L2 = same rows every
  tile) and _gw_verify GX (1 no V widen once the ring is warm, 2 all-L2, 3 no consumer math after the LZ_PK exact tiles,
  4 = 1+2); results stay finite. Layer: TUNE option "attx" (diag: timed, never picked), set per call (full layers only);
  `ATT_DIAG_BUILD = True` (sed) makes the DIAG TUNE. DIAG copy cold prepare 115 s (Gluon variants compile fast).
- 10:01Z CHECK cf5e1621 (DIAG copy variants/diag84.py, never submit) 2.0778x, 10,887 us. H200 whole-layer tuner us (synthetic
  ctx 80000, fixed lengths):
      T=16 full  real 229.8 | gathers-only 228.8 | all-L2 210.6
      T=32 full  real 334.0 | gathers-only 332.8 | all-L2 282.4
      T=128 full real 351.9 | all-L2 341.0 | no consumer math 373.3 (!)
      T=256 full real 509.5 | no widen 503.9 | all-L2 489.1 | no consumer math 501.1 | all-L2 + no widen 473.9
  H100 same probes (cold prepare run): T=16 303.2/293.5/265.9, T=32 422.3/429.9/311.9, T=128 426.5/423.9(allL2)/459.9(nomath),
  T=256 620.4/598.4/551.9/595.4/535.8.
  READING: decode full (_gd_full d16/d32) is purely gather-bound on the H200 (math removed: -1.0/-1.2 us). Verify full v32:
  free memory -20 us, free consumer -8 us, so its producer and consumer are balanced near ~180 us and the overlap loss is
  ~10-20 us; v16: free memory -11, free consumer +21 (free-running gathers are SLOWER: pacing helps DRAM). Ceiling of all
  consumer-side work on full attention: ~-1 (d16) -1 (d32) ~0 (v16) -8 (v32) us = ~0.15% total. The only real lever left is
  the random-128B gather rate itself (d32 full 655 MB in ~166 us = 3.94 TB/s vs moe's contiguous ~4.26: <= ~7% = d32 -12,
  d16 -6 us = <= ~0.4% total, unproven).
- v126 (83a4f42) = v125 + TUNE option "aplan" (attn_src ATT_PLAN plan overrides, full layers only; tuner skips "full" options
  on sliding layers): T=16/32 {stages 3}, {stages 3, dprogs 768}, {block_n 16, stages 4} (more gathers in flight: the gather
  lever above, chosen on the H200 itself); T=128/256 {lag 2}, {stages 2}. {dprogs 1536} hits an illegal address under graph
  replay in the tuner (eager runs pass): dropped (latent dbal limit > 1024 programs, not on any default path).
  H100 tuner: all aplan options tie/lose (T=16 +2..5, T=32 +3..10, T=128 -2.5..0 (confirm tie), T=256 +5..13).
  validate --quick ALL OK (13/13, scored 0/12, opt 0/320, tk2f 0/300, replay 0/240, route 0/240 gap 0.0256, cold 139.5 s).
- 10:14Z CHECK 3df61290 v126 (83a4f42) 2.0844x, 10,852 us (noise tie; the tuner picked NO aplan option, so the run = v125 code
  paths; not submitted). H200 tuner, whole layer: T=16 full default 236.1 vs stages 3 238.4, s3+768 241.8, BN16 s4 244.2; T=32 339.7
  vs 341.2 / 348.4 / 355.2; T=128 382.1 vs lag 2 395.4, stages 2 397.5; T=256 545.4 vs 580.3 / 582.9. More gathers in flight LOSES
  on the H200 (as the unpaced no-math probes said): the attention plans are at their H200 optimum. aplan options removed from TUNE
  (v127 = v126 minus the options; plumbing ATT_PLAN/ATT_DIAG stays, inert). Full-layer attention lane: CLOSED at ~0.15% ceiling
  (consumer) + an unproven gather-rate question (moe asked to run ~/attn exp/hdiag5 on the attention budget).

## s85 (2026-10-27 10:27Z-, night4 session 34): v128 = v127 + moe s31 verify-32 sliding cl nvs
- No new lead decision after 09:45Z (s84 lane question open). Only Handoff for layer: moe s31 v32 sliding nvs `cl` (agent2
  halfpfkr: the 2 splits of a (request, KV head) as a 2-CTA cluster merged over DSMEM, writes out, no _combine_hb; H200
  standalone v32 sl 15.51 -> 14.07/14.39 us). Skipped moe's csl/vtk __fdividef owner merge (H100 -0.1..0.2 us, precision change).
- git merge-file of moe's whole attention file onto attn_src mis-aligned (4 tangled conflicts): ported the cl source verbatim as a
  SEPARATE build `_NVSCL_SRC`/`_NVSCL` (r-moe 6c4fee3's _NVS_SRC), on the build pool, joined in _join_builds. _verify_nvs uses
  it when data["nvs_cl"] (TUNE option nvscl, layer NVS_CL = True), batch >= 32 (S 2), BF16 qkv, no QSS slabs, no O_PF (both off
  at T=256 by default); else the old nvs + _combine_hb. Tuner: new "sl" flag = sliding-only option (skipped on full layers).
- H100: tuner T=256 sl {nvscl False} 398.8 vs default 399.5 (one-shot); ab_flag 6 rounds verify 32 sliding 370.81 vs 371.90 us.
  validate --quick ALL OK (test 13/13, scored 0/12, opt 0/302, tk2f 0/300, replay 0/240, route 0/240 gap 0.0256, cold 140.5 s).
  proj's tools/attn_probe.py (copied from proj/attnprobe) --only=verify --passes=0: all 6 verify shapes PASS, worst 2.07e-3.
- 10:5xZ CHECK 940a4fca v128 (536adfd) 2.0943x, 10,801 us (last layer check of 10-27). Per shape: d1 66.1/76.0, d16 126.4/206.6,
  d32 161.7/320.5, v1 108.7/124.1, v16 260.5/350.3, v32 337.5/511.1 (v32 sl slower than board 322.0: SGLang 681.8 vs 698.1, judge
  noise). H200 tuner T=256 sl: default (nvscl on) 349.7 vs {nvscl False} 350.6 us (-0.9 us, same sign as H100 and moe's standalone;
  below noise); tuner picked {hot 24} on confirm. NOT submitted: 1.9% under the board 2.1339x (lead 09:28Z rule needs within 1%).
  main = v128. Value of the item ~0.03-0.05% total: housekeeping, not a lever.

## s86 (2026-10-27 11:18Z-, night4 session 35): lead 10:50Z lane (a) T=8 o+bnd+pr8 fusion built (OB8) -> DEAD on H100
- Lead 10:50Z: build the T=1 o+bnd+pr1 megakernel and its T=8 analogue (proj s80 H100 oracle: no boundary at T=8 = o->pr8 end -4.2 us).
- Built OB8 (ob8_src.cu, r-layer main, flag OB8 / OB8_LATE, off and not compiled unless OB8 = True): gemv12's m=8 body for o
  (k 1024, 128 blocks x 512 thr, one (token, column) per thread) + the first boundary in its epilogue: per block/token 4 FP32
  partials (sum b^2, (as)^2, (as)(bb), (bb)^2 -> sum s'^2 = A + 2 rb C + rb^2 D), part[block][tok][4], epoch-counter grid barrier
  (all blocks resident: pr8 launches only after every block triggered), fixed-order sum (deterministic), then each block writes
  stream1 / h / h32 for its columns; pr8 = its PDL secondary. bench test 13/13 PASS with OB8 forced on.
- H100 ab_flag (TUNE[8] = ({},), SEEDS 99,7,13, 6 + 5 rounds), us base / OB8 (early trigger) / OB8_LATE (trigger after W loop):
  v1 sl s99 140.96 / 143.42 / 143.17, s7 114.83 / 116.99 / 116.99, s13 134.38 / 136.58 / 136.02;
  v1 full s99 153.54 / 153.71 / 153.86, s7 127.10 / 127.74 / 127.82, s13 153.66 / 154.24 / 154.13.  => +2.2..2.5 sl, +0.2..0.6 full.
- Why (/tmp/s86/tlavg.py: median timeline of 25 cold replays, v1 sl seed 7): base g12 14.9-27.2 | bnd -31.0 | pr8 20.8-51.3;
  OB8 ob8 14.7-29.0 (barrier + epilogue +1.8 us over g12's end) | pr8 19.1-52.9 (23.9 us after ob8 end vs 20.3 after bnd).
  pr8 is 1024 threads x <= 64 regs (a whole SM's register file) + ~140 KB smem: it cannot share an SM with ANY o block. With the
  16-program boundary, pr8 gets ~116 SMs as g12 blocks exit (~27 us) and runs its pre-wait W ramp behind the boundary (the
  boundary's 3.8 us is effectively hidden). The grid barrier keeps all 128 SMs until the slowest o block + epilogue, so pr8's ramp
  is exposed. proj's -4.2 us oracle (no boundary at all) does not survive any form that needs h before pr8 (an o-side barrier or
  a pr8-side prologue, which needs 5x the x reads per block; PR1_BND s74 = the T=1 prologue form, dead on H100 + H200).
- T=1 o+bnd+pr1 megakernel: same mechanism (pr1 = 1024 thr x 64 regs, o g13 = 528 blocks at 4/SM; PR1_BND s74 showed the ramp:
  o end -> pr1 end 21.4 separate vs 22.5 fused). Not built: closed with this measurement. Lane (a) CLOSED on H100; no judge runs
  (budget 10-27: 0 checks left).
- Only remaining shape of the idea: make pr1/pr8 light enough to co-reside with o (<= 512 thr x 64 regs, so its W ramp runs
  during o's tail) = a proj kernel redesign of pr1/pr8, after which the boundary fusion could pay. Not attempted.
- main: OB8 code inert (layer.py allocates stream1/h/h32 before the o GEMM now: needed by the ob8 path); validate --quick: see below.
- LATENT BUG FOUND + FIXED (validate --quick opt: `verify 1 sliding {'pr1': False}` FAIL; old main v128 fails the same, 7/30
  seeds, /tmp/s86/pr1f.py): with pr1 off at T=8 the routing runs on the side stream, but the nu8d pre-expert _rmsnorm_gain reads
  local_ids (ZL zeroing of y32 rows) on the main stream BEFORE main joins the side stream -> stale ids. Probe: {pr1 False} 5/7 bad,
  + nu8 False 0/7, + nu8d False 0/7. Fix: nu8d only with pr1 or tk2fj (routing on/joined to the main stream). After: 0/30.
  Default paths unaffected (pr1 on at T=8; the H200 tuners measured {pr1: False} +1.7..2.6 us and never picked it), but a noisy
  tuner pick would have produced wrong outputs on the judge: the fix removes that risk.
- validate --quick on v129: ALL OK (test 13/13, scored 0/12, opt 0/302, tk2f 0/300, replay 0/240, route 0/240 gap 0.0256, cold prepare 118.1 s).

## s87 (2026-10-27 11:55Z-, night4 session 36): lead 11:50Z correctness item (T=8 {pr1: False} nu8d race)
- main v129 (6621403, s86) already carries an equivalent fix (nu8d only when pr1 or tk2fj keeps routing on / joined to the main
  stream; old main 7/30 bad -> 0/30). proj's fix (proj/nu8fix 357859d: join first, rpdl off) not merged on top (same effect).
- H200 tuner audit (lead asked): parsed the '-> <pick>' of every 'layer tune' line in all 22 saved layer judge logs
  (profiles/*.json|log, incl. board run 35dd53d0 = copied from /tmp/v125/profiles into profiles/, and the only later check 940a4fca).
  T=8 never picked any {pr1: False} candidate. 35dd53d0: T=8 sl -> {cslpw: True} (pr1 False 122.2 vs default 118.3 us),
  full -> {cslpw, g14alt:qkv_gate 3} (134.2 vs 128.4). The board run did not take the racy path: no withdraw needed.
- v130: all three {pr1: False} TUNE[8] options removed (never picked, +1.7..5.8 us on every H200 log). The s86 guard stays
  (pr1 can still be off if pr1's build is unavailable).
- v130 validate --quick: ALL OK (test 13/13, scored 0/12, opt 0/302, tk2f 0/300, replay 0/240, route 0/240 gap 0.0256,
  cold prepare 147.8 s; s86 was 118.1 s, GPU shared). proj's tools/nu8_race.py (GPU hog, run under the GPU lock) on v130,
  options forced via state['tune']: {} 0/60, {pr1 False} 0/60, {pr1 False, g19:pooler} 0/60.
- Request to proj (COORD s87): a lighter pr1/pr8 (<= 512 thr x 64 regs) as a TUNE option = the last open form of the o+bnd+pr
  fusion. 0 judge runs (layer checks 20/20 for 10-27; 4 submits left but nothing beats the board).

## s88 (2026-10-27 12:12Z-, night4 session 37): light pr8 (last open form of o+bnd+pr8) built -> DEAD on H100
- Took my own s87 request to proj (COORD claim). pr8_src.cu template generalized: fused<KCU, WG, PF, XP, XR0, NGT>: NGT row
  groups per block (4 = old 1024 thr; 2 = 512 thr, 256 blocks, router items of 8 rows, finisher waits for a.bk items), x staged
  as an XR-piece ring of XP pieces (cfg xp = XP | XR << 4; escape fixups then read x from global). Old config = <8,8,4,1,0,4>,
  same code path (ring/light branches compile out). Correct: opt_test 0/128 at T=8 with the alts forced (4 seeds).
- Pitfall: finish()'s static smem (~10.9 KB, inlined) made the first light config (XP4 XR2, ~117 KB) 1 block/SM = 2 waves (+6 us).
  Configs that really fit 2/SM: XP4 XR1 (~85 KB) and XP8 XR2. g12 o = 512 thr x 56 regs, 1.3 KB static: co-residence possible.
- H100 tlavg (/tmp/s88/tlavg.py = s86's, 15 cold replays, v1 sl seed 7), pr8 end - boundary end / layer end:
  base 20.5 / 114.0-115.0; light XP4R1 20.8 / 114.9; light XP8R2 21.0 / 115.5; heavy ring XP4R2 21.8 (ring + global escapes
  cost ~1.3); light PF2 30.9 (in-flight bound). ab_flag v1 full (first light cfg, 2 waves) +5.0..5.4.
- OB8 (s86) + light pr8 (the combination s86 lacked): pr8 end - ob8 end 23.5-23.8 vs 20.5 after the separate boundary; layer
  end 116.1-116.6 vs 115.0 (OB8 alone 116.3). Pre-loaded light pr8 blocks beside ob8 do not shorten pr8: pr8 ends at ~51-53 us
  whatever precedes it = the separate boundary is ALREADY hidden; pr8's own stream rate is the cost, not its ramp. CLOSED.
  (Test pitfall: OB8 must be set in the file text (sed), not appended: _g10_lib prebuilds at import with the stub ob8_gemm.)
- What is left there: pr8 STREAM RATE. H200 board profile 35dd53d0: pr8 21.3 us (kernel duration incl. PDL wait) vs pr1 17.5 at
  d1 for ~the same 56 MB (pooler 12-bit 48 MB + FP32 router 8 MB). H100 cannot judge it (2.7 TB/s there). Worth an H200 diag
  of pr8 variants (PF 6/8, WG, bulk prefetch depth) inside a layer check once budget is back.
- main v131: generalized pr8 template + PR8L_ALTS = [] (nothing extra compiled), no TUNE option. validate --quick ALL OK
  (13/13, scored 0/12, opt 0/296, tk2f 0/300, replay 0/240, route 0/240 gap 0.0256, cold prepare 116.8 s). 0 judge runs.

## s89 (2026-10-27 12:34Z-, night4 session 38): moe s33 handoff ported (v132); proj/attn31 stack = H100 tie, left out
- v132 = v131 + moe s33 `spec=True` on the verify-1 full attention plan (attn_src _plan, splits 127 / pf / pre 2 / fix 1): seq from
  the page-table width, redo if the real length differs. The layer's batch-1 table width is always longest + q_len = seq + 8
  (reference.py:259), so the redo never runs here (moe tested it standalone: tools/s33/spec_redo.py). H200 standalone (moe DIAG,
  4/4 pairs) -0.63/-0.70 us. H100 in-layer ab_files v1 full, 6 rounds: AB 155.41 -> 154.86, BA 155.81 -> 155.53 (mean ~-0.4).
  attn_probe verify 1 8 passes PASS; validate --quick ALL OK (13/13, scored 0/12, opt 0/296, tk2f 0/300, replay 0/240,
  route 0/240 gap 0.0256, cold prepare 118.9 s).
- proj/attn31 (c71dbb6: moe s31 csl/vtk owner-merge __fdividef, nds page-table L2 prefetch + K-early readyv) applied on v132:
  attn_probe 29/29 PASS; H100 ab_files sliding 4 rounds AB/BA: d1 +0.1/+0.1, d16 -1.0/+0.8, d32 -0.8/+1.5, v1 -0.7/0,
  v16 -0.3/+0.1, v32 -0.1/-0.4 (second slot faster in both orders = proj's position artefact) = TIE. Not on main (no evidence,
  changes nds sync); patch = `git diff fa3f107 origin/proj/attn31 -- muse_decode_layer/attn_src.py`.
- pr8 tuner variants for the next check: skipped (proj s36-s38/s80: no pr8 lever > ~1 us, H200 incl.).
- 0 judge runs (layer checks 20/20 for 10-27). First check after 00:00Z: main v132.

## s90 (2026-10-27 12:44Z-, night4 session 39): proj stream_fuzz merged; v132 fuzz-clean; 0 judge runs
- Merged proj/fuzz 19b5ba1 tools (tools/stream_fuzz.py + validate.py step 'fuzz') into main. Full fuzz on main v132
  (side,main,gside,gmain, 2 reps, every TUNE set per scored shape): 0/1462 bad (314 s). v132 = race-clean under injected delays.
- H100 timelines (seed 99) for orientation: d1 sl 86.3 us (g13 qkv 13.0, csl 6.6, o 9.8 ovl, bnd, pr1 21.4, tk2f 34.7, n2u 8.4,
  bnd -> end); v1 sl 145.5 (g14 14.8, vtk 6.9, pr8 32.4 span, tk2f 84.2, nu8, bnd); d16 sl 195.8 (g24 qkv 14.3, csl 13.1, g27 o,
  bnd, g24 pool 26.6 || router_split 19.4 + _routing, tk2f 111.6, norm2, g24 unpool 30.0 span, bnd). Experts at stream rate;
  pooler+router phase ~2.6 TB/s; nothing new found outside closed lanes.
- Considered + dropped: L2-prefetching "hot" experts predicted from router_bias during attention. The reference's skew comes
  from router rows aligned with the stream's mean direction (data), bias alone predicts weakly; s55 bounded expert L2
  prefetch at <= ~1 us anyway. Not built.
- Budget 10-27: layer checks 20/20 used, 4 submits left (nothing beats the board). First check after 00:00Z: main v132.

## s91 (2026-10-27 12:53Z-, night4 session 40): v133 = v132 + TUNE[8] pr8_npf options (H200 decides pr8 bulk-PF depth)
- No layer checks until 00:00Z (20/20 used). Every listed lane is closed (STATE). Cheapest open H200 question: pr8's stream rate
  (H200 21.3 us vs pr1 17.5 at d1 for ~the same bytes); H100 can't judge it. pr8 bulk L2 prefetch depth is a runtime arg
  (npf: 2xx = 'xx' atoms after the wait (main 212), 1xx = before the wait; W only, so pre-wait is safe), so no new compile.
- layer.py `_pr8_cfg(state, tokens)` reads TUNE `pr8_npf`; TUNE[8] += {cslpw, pr8_npf 206 / 220 / 112} (cslpw = the H200
  sliding pick on 35dd53d0). H100 tuner: npf 220 -0.9 sl / -0.6 full vs {cslpw} (noise level); 112 +0.9/+2.3; 206 +0.8/+0.4.
- opt_test T=8 3 seeds 0/108; validate --quick ALL OK (13/13, scored 0/12, opt 0/302, tk2f 0/300, replay 0/240, route 0/240
  gap 0.0256, fuzz 0/449, cold prepare 118.5 s). Readout for the 00:00Z check: 'layer tune T=8' pr8_npf entries vs {cslpw}.
- 0 judge runs. First check after 00:00Z: main v133.

## s92 (2026-10-27 13:02Z-, night4 session 41): verify-load gap re-audit (no build); 0 judge runs
- Budget 10-27: layer checks 20/20 used, 4 submits left (nothing beats the board). main = v133 unchanged.
- H100 timelines (main v133): v16 sl 297 us = qkv cuBLAS 18.6, nvs+combine 14, o 7, bnd 6.0, FR pooler+router 33.8, routing+
  route+gather ~8, twm 172 (88-260), norm tail 2.3, unpool 28.5, bnd 5.5. v32 full: qkv 22.1, gw_verify 241 (new_tokens/
  combine_bal overlap its tail, +4 us), o 10.3, bnd 9.5, FR 39.0, routegather 8.9, hot||twm 239 (335-574), unpool 33.4, bnd 9.2.
- Where the verify gap is (H200 940a4fca profile + tools/route_hist.py): local experts touched v16 17-18, v32 18-21 (~25.5 MB
  each) -> stream-rate (4.06 TB/s, moe fit) experts ~111 us v16 / ~125 us v32 vs twm 168 / ~208 us span: ~57 / ~80 us above
  stream rate = ~3.1% + ~3.5% weighted = the largest single slice of the gap to #1. moe measured it dq-bound (s18-s30: dq ALU
  serializes with RS wgmma, WS dq->smem->SS no overlap, scaled LUT/byte LUT no fewer ops, cvt e2m1x2 sm_100 only, FP8/FP16
  MMA not lossless on BF16 x). Lead closed twm 09:45Z. A cross-SM split (dq SMs -> BF16 in L2 -> SS wgmma SMs) needs ~4x the
  expert bytes through L2 twice (~3.6 GB at v16): dead on paper.
- cuBLAS GEMMs at T=128/256 on the H200: qkv 16.9 / 20.5 us (37.7 MB), FR pooler+router 25.8 / 33.6 (75 MB), unpool 21.2
  (67 MB) = 2.2-3.2 TB/s; to 4.0 TB/s would be ~1-1.5% total, but proj's own m128/256 GEMMs (g22, g23, DeepGEMM bf16_gemm_nt,
  lt algo sweeps) all lost (proj STATE s16/s25/s28). Not retried.
- Boundary at T=256: ~9.5 us H100 each vs ~6-7 us for its ~24 MB (FP32 stream in/out + BF16 branch/h): <= ~0.15%; BND_SPLIT
  already noise (s74). Not built.
- Next: first check after 00:00Z = main v133 (read 'layer tune T=8' pr8_npf lines). No open >= 1% lever outside twm dq.

## s93 (2026-10-27 13:08Z-, night4 session 42): plateau check, no build; 0 judge runs
- Board: #27 at 2.134x (35dd53d0); #1 pjmullin3 2.399x, #20 2.172x. Budget 10-27: layer checks 20/20, nothing to check until 00:00Z.
- No new Handoffs since s89 (moe s35 still running _gd_full d1-full DIAGs; proj s84 running full validate of v133).
- Largest non-expert gap left (940a4fca profile): g24 at d32 = pooler + unpool 49.6 us for 2 x 48 MB 12-bit (~2.0 TB/s; d16 35.5 us
  = 2.7 TB/s), qkv g24 m32 13.2 us. vs 4 TB/s that's ~24 us d32 + ~11 us d16 = ~2.5% weighted (sl+full). proj already worked it
  in depth (STATE-proj s12/s50/s55/s60/s61: consumer/latency-bound at 30% issue; wgmma consumers g20/WG mode, BF16 wgmma
  g23, cuBLASLt (in TUNE[32]) all lose). Re-opening it needs a new consumer design, not a tweak; flagged here, not started.
- Expert weights incompressible (moe NOTES: e2m1 codes uniform, 4.0 bits). main = v133 unchanged; first check after 00:00Z = v133.

## s94 (2026-10-27 13:26Z-, night4 session 43): main v134 = v133 + proj/qf35 bundle; 0 judge runs
- Budget 10-27: layer checks 20/20 (none until 00:00Z), 4 submits left. Board #27 2.134x; #1 pjmullin3 2.399x.
- Merged origin/proj/qf35 (ac340f7, merge 566828e) = v133 + (1) moe s35 qf=True on the batch-1 decode-full _gd_full plan
  (query rows loaded before the first K/V gathers; moe H200 attention DIAGs 95f8158e+1259baad -0.28 us 5/5 pairs, H100 tie)
  + (2) proj s68 LT_ALT (pooler,128) x2 + TUNE[128] {ltalt:pooler 1}, {ltalt:pooler 2}, {ltalt:pooler 1 + twf_split 66 +
  lt:o + lt:unpool} (H200 standalone pooler m128 24.06 vs 26.75 us; H100 tuner picks the combo at T=128 sl 333.8 vs 336.3).
  proj s85 validated the merged code: full validate (qf) ALL OK, --quick on the combination ALL OK, cold prepare 116.5 s.
  Merge clean (no notes reverted); build.py regenerates submission.py identically.
- First check after 00:00Z = main v134: read 'layer tune T=128' ltalt:pooler lines and 'T=8' pr8_npf lines; d1 full per-shape.
- Not built: re-read moe's twm microbenchmarks (NOTES-moe s18): dq ALU and wgmma don't overlap in one WG, WS/SS variants worse:
  nothing new to try there from the layer side.

## s95 (2026-10-27 13:43Z-, night4 session 44): plateau re-check, no build; 0 judge runs
- Board: #28 at 2.134x (35dd53d0); #1 pjmullin3 2.399x, #20 2.172x. Layer checks 20/20 used until 00:00Z; main = v134 unchanged.
- No new Handoffs/Requests for layer since s94. moe and proj both report no open >= 0.1% attention/expert item (COORD s36/s37/s84).
- Re-read 940a4fca d32 sl profile (weakest ratio, 1.61x): tk2f 88.1, g24 pool+unp 49.6, g24 qkv 13.2, norm2 12.3 (includes its
  PDL wait; s72 oracle: norm2 is free), nds 10.3, bnd 2x 4. Re-checked the g24 m32 consumer against proj s61 (ncu 30% issue,
  2.75 active warps/sched, latency-bound; XSP/DF/EZ dead on H200; G24P = the pooler EZ chain already a TUNE[32] option here).
  Only untried form I see: split M=32 across 2 co-scheduled CTAs (M16 each) sharing one W tile via cluster TMA multicast, to
  double the consumer warps per W byte. proj's cluster grids lost before (NOTES-proj s2/s58: max active clusters, MC dead for
  PD), so I'm noting it as a proj idea and not building it here.
- First check after 00:00Z = main v134 (unchanged plan).

## s96 (2026-10-27 14:00Z-, night4 session 45): s95's g24 m32 M-split idea measured DEAD (consumer oracle); 0 judge runs
- Layer checks 20/20 used until 00:00Z; main = v134 unchanged; no new Handoffs/Requests for layer.
- s95 idea (split M=32 into 2 x M16 consumers sharing one W tile: cluster multicast or warp pairs on one ring slot) would
  DUPLICATE the 12-bit decode (dec8 is per W byte, independent of M), so it only wins if doubling consumer warps nearly
  halves the m16 consumer time. Oracle with proj's harness (run in place, read-only: ~/r-proj/muse_bf16_proj/dev/s61/g24.py,
  SH=pool, H100, us over a 3 us pred), G24_NOW (consumer only, no W HBM):
  m16 (1024,4,3,12,2) 12.03 | (1024,4,2,12,2) 12.26 | (512,4,2,24 warps,2) 11.74 ; m32 (512,4,3,12,2) 15.42 | (512,4,2,16,2) 14.82.
  Full (HBM): m16 18.59 / 20.35 / 19.07 ; m32 22.43 / 22.82.
  24 warps at m16 = -2.4% vs 12 warps: the consumer is per-SM throughput-bound, not warp-latency-bound. A pair split would
  cost ~2 x 11.7 = ~23 us of consumer time vs 15.4 for today's m32: DEAD (both cluster and intra-CTA forms). Not sent to proj.
- Nothing else open on the layer side (STATE CLOSED list). First check after 00:00Z = main v134 (unchanged plan).

## s97 (2026-10-27 14:19Z-, night4 session 46): plateau, no build; 0 judge runs
- Board #29 at 2.134x (35dd53d0); #1 pjmullin3 2.399x, #20 2.172x. Layer checks 20/20 used (resets 00:00Z), 4 submits left (unusable without a check).
- No new Handoffs/Requests for layer; moe s36-s38 and proj s86 report no open item (proj g28 ping-pong m32 consumer DEAD).
- Re-derived the e2m1 -> s00eem00 e4m3 byte -> cvt.rn.f16x2.e4m3x2 dequant for twm (exact incl. subnormals, factor 2^-6):
  already DEAD on paper in NOTES-moe (line ~919: >= today's instr count, needs FP16 x = not lossless on BF16 x). Not retried.
- First check after 00:00Z = main v134 (unchanged plan).

## s98 (2026-10-27 14:36Z-, night4 session 47): T=128 split-K reduction bound measured (for moe's s41 xr); 0 judge runs
- Budget 10-27: layer checks 20/20 (reset 00:00Z). main = v134 unchanged. moe claimed lead 13:30Z item (2) (T=128 GEMM,
  "xr" = resident x K-slice per CTA, W streamed, FP32 red.add out[N][M]); I did not build a second GEMM, I bounded it.
- H100 v16 sliding timeline (main v134, seed 99): qkv lt 18.9 | nvs+combine 15 | o 7.0 | bnd 5.8 | FR pooler+router 32.5 |
  routing/route/gathern ~8 | twm 172 | unpool 29.0 | bnd 5.5 us. GEMMs = 87 us at 2.0-2.4 TB/s.
- moe's xr @14:37 (copied to /tmp, run read-only): pool 4096x8192 44.2 us, unp 8192x4096 43.9 vs cuBLAS 29.2/28.9;
  with XR_RED=0 (no reds, diag): 27.5 / 27.4. The reds cost ~16.5 us and do not overlap the W stream.
- dev/s98_redbench.py: red.global.add(.v4).f32 at xr's volume and S-way overlap = ~2.1 TB/s FP32 on H100 (v4 = scalar):
  FR T128 (G8 S16) 9.4M floats 17.3 us, pool 16.1, unp (G16 S8) 15.9, FR T256 (S32) 37.7M floats 55 us.
- dev/s98_dsmbench.py: cluster (4/8 CTAs along K) DSMEM reduce per 64x128 FP32 tile before one red: ~2.2/2.6 us per tile
  marginal (~1.9 TB/s aggregate) + ~7 us fixed: FR T128 26/31 us, slower than plain reds. No cheap reduction path.
- Conclusion (posted to moe, COORD Requests s98): split-K at T=128 pays ~16 us of partial reduction that has to hide under
  a ~21 us W stream; red-free xr is already only 2.4 TB/s. Dead unless the red volume drops ~4x. T=256 split-K worse (55 us).
  Also posted the layer interface xr would need (token-major out[T][N], zero_next chain, N=2304, T=256).
- No layer plumbing built (nothing to plug in). First check after 00:00Z = main v134 (unchanged plan).

## s99 (2026-10-27 14:47Z-, night4 session 48): lead 13:30Z item (2) BUILT: g30 = 12-bit-weight wgmma GEMM for T=128 (H100 standalone); 0 judge runs
- Triton first (dev/s99_g12t.py, simple sm-byte + nibble layout, tl.dot): decode inside Triton is far too slow (best pool
  57.9 / unp 41.9 us vs cuBLAS 29.3 / 28.8; even with x read once 48 / 38 us). Same Triton GEMM on plain BF16 W: unp 27.2 (beats
  cuBLAS 29.0), pool 32.4. Triton is dead for 12-bit at T=128.
- dev/s99_g30.cu + dev/s99_g30.py (CUDA): swap-AB wgmma m64n128k16 with A = W decoded straight into REGISTER A-fragments by
  proj's DEC3 dec8 (1 prmt + 1 lop3 per BF16 pair; dec8 output = the m16n8k16 / wgmma A fragment), B = x [128 tok x 64 k] SW128
  TMA box. CTA = 64 W rows x K/S: producer warp (6 KB 1-D bulk W stage packed per thread fragment + 16 KB x box per stage,
  NS ring) + 2 consumer WGs on alternate stages, summed through smem; epilogue staged token-major in smem, 16 B stores.
  pack30() in s99_g30.py (uses layer _compress/_dec3: lossless; ESCAPES NOT HANDLED yet). Lesson: A[par][..] with runtime par
  put the fragments in local memory (60 us); 2x manual unroll with A0/A1 fixed it (22 us).
- H100 cold-L2 graph us (cuBLAS same run): unp 8192x4096 S1 21.9 (best NS10 21.0) vs 25.8 = -4.8 us; pool 4096x8192 S2 (FP32
  slabs) 25.7 / NS10 24.6 vs 26.4; S4 28.4; pool S1 (64 CTAs, MMA-bound on 64 SMs) 34.1; unp S2 28.0 (FP32 slab out +6 us).
  Ablations (unp S1): no MMA 20.6, no x loads 21.1, no W loads 18.1. => not decode/MMA-bound: L2-bound: every CTA re-reads all
  of x (128 CTAs x 1 MB = 128 MB of L2 reads + 50 MB W = ~8 TB/s of L2). MMA alone ~9 us at 128 CTAs.
- NEXT (biggest lever): TMA multicast of the x box across a cluster of 2-4 CTAs along N (x L2 reads /2../4; CUTLASS pattern:
  each CTA loads 1/C of the box with .multicast::cluster, empty barriers cluster-wide). Expect unp ~16-17 us (W stream bound at
  ~3 TB/s), pool with S2 + cluster. Then: escape fixups (per-row headers in the epilogue, x from global, s == 0 slab only), T=256
  (MMA ~17 us H100: n256 or two passes), qkv shape (QKV_SIZE x 8192), integration as TUNE[128/256] g30:<key> options. On the H200
  (HBM 4.8 TB/s, L2 not faster) the x L2 traffic matters even more: multicast before any H200 test.
- s99 continued (same session, all H100 standalone cold-L2, us; cuBLAS 25.5-25.9 unp / 26.3-26.8 pool in the same runs):
  * Cluster multicast of the x box (G30_CL, dev/s99_g30.cu): first form used mbarrier.arrive.RELEASE.CLUSTER for the remote
    slot release: 40 us at CL=2 (pathological; cluster launch alone with local release = 22.9). With CUTLASS's plain
    mbarrier.arrive.shared::cluster: CL=2 23.1 (tie with CL=1 21.9), CL=4 39.8, CL=8 ~77. Multicast DEAD: halving x L2 reads
    buys nothing, so the s99 "L2-bound by x" reading was wrong.
  * Split rings (own producer warps; G30_NS = x stages of 16 KB, G30_NW = W stages of 6 KB): x depth matters most (NS4/NW16
    35.6, NS4/NW24 37.9, NS6/NW16 25.8, NS8/NW8 21.8, NS10/NW8 22.2, NS10/NW10 **21.6** unp / **25.4** pool S2, NS12/NW4 35.0):
    each CTA's x box from L2 is latency-bound and needs ~128-160 KB in flight next to W. Smem caps the depth (NS12/NW6 no fit).
  * K-rotation per CTA (G30_ROT, CTA g starts at stage g*KS/grid): WORSE (unp 26.4, pool S2 28.6): lockstep x reads help.
  * R2 (CTA = 128 W rows, WG w on row group 2g+w over all k, x shared; G30_R2): unp S1 (64 CTAs) 32.4, unp S2 25.7-26.4,
    pool S2 33.4, pool S4 26.6. Halving x per W byte gains ~2 us but the FP32 slab output (S>1) costs ~6 us. Not better.
  * Best now: R1 S1 NS10/NW10: unp 8192x4096 21.6 vs cuBLAS 25.7 (-4.1 us, -16%); pool 4096x8192 S2 25.4 vs 26.5 (-1.1, slabs
    summed by the consumer). qkv (2304 x 8192 = 36 row groups) fits badly (S4 + slabs). At T=128 this is ~-5 us per v16 layer
    on H100 = ~0.3% total if it held on the H200, which is unmeasured: the kernel is not HBM-bound on the H100 (x latency /
    SM ingress), so the faster H200 HBM helps cuBLAS more than g30. Gate on an H200 number before building more.
  * To ship it as a TUNE[128] option: escape fixups (headers; per-row x[t][col] * (v - 2^-15) in the epilogue, s == 0 slab
    only), the pack into _g12_prepare (pack30 in dev/s99_g30.py), PDL, and replay/route tests. T=256 = 2x MMA (n256 or
    two passes), not tried.

## s100 (2026-10-27 15:39Z-, night4 session 49): g30 integrated: T=128 unpool on the 12-bit wgmma GEMM (main v135); 0 judge runs
- Budget 10-27: layer checks 20/20 used (reset 00:00Z), 4 submits left (unusable without a check).
- g30_src.cu = s99 dev kernel (split W/x rings NS10/NW10, S=1 only) + proj s91's in-fragment escape patch (r-proj
  dev/s91/g30e.cu @15:43Z: esc_patch/esc_pop: the thread owning an escape's (row, k) A-fragment half overwrites it with
  v = true/scale, exact in BF16; prepare drops g30 if any v isn't BF16-exact/normal) + PDL (W bulk loads before
  griddepcontrol.wait, x after; launch_dependents at start). layer.py: _g30_build (on _BUILD_POOL), _pack30, _g30_prepare
  (in _g19_prepare; F.linear check), _linear T=128 path, G30_PDL / G30_TRIG (second boundary = g30's PDL secondary).
- First escape form (epilogue acc += x[t][col] * (v - 2^-15), x gathers on the tail): 28.1 us vs 21.6 without escapes:
  DEAD. In-fragment: 22.4 vs cuBLAS 25.6 us (H100 cold L2, reference unpooler, ~3.2k escapes).
- Exactness: dev/s100_g30exact.py (x = one-hot rows over every escape column + 256 random: out == W bitwise): 0 mismatches
  on seeds 0/1/7 (~20M weights each). dev/s100_g30t.py: rel err = cuBLAS's (1.66e-3).
- In-layer H100 ab_flag (4 rounds x seeds 99/7/13, v16 = T=128): g30 unpool sl -4.1/-4.5/-5.2, full -5.3/-4.7/-5.5 us.
  G30_TRIG: -0.6..-1.5 more (4 rounds x 2 seeds); G30_PDL (g30 as secondary) ~0, kept on. H100 tuner (bench prepare) picked
  g30 at T=128 full; sliding picked the twf_split/lt stack (g30 was not combinable then; now g30 is in TUNE_BASE[128] so
  every option set includes it; escape hatches {g30:unpool False} and {g30 False + twf_split 66 + lt:o + lt:unpool}).
- T=256 form (dev/s100_g30_tt2.cu: WG w = token half w over all stages, x stage 32 KB, no reduce): exact, but 38.9-40.7 vs
  cuBLAS 27.6 us for every ring depth (NS2 4-6 / NW 2-16): each CTA must ingest 2 MB of x (per-SM ingest ~50 GB/s,
  proj s16/s28): DEAD. g30 is a T=128-only kernel.
- validate --quick (with g30:o option) ALL OK: test 13/13, scored 0/12, opt 0/316, tk2f 0/300, replay 0/240, route 0/240 (gap 0.0256), fuzz 0/464, cold prepare 121.8 s. o on g30 standalone 10.2 vs cuBLAS 10.5 us (tie): TUNE[128] option only. qkv (36 row groups) / FR pooler (S2, -1 us standalone) not built.
- First check after 00:00Z = main v135; read the H200 'layer tune T=128' lines ({g30:unpool False} should lose by ~4-5 us).

## s101 (2026-10-27 16:04Z-, night4 session 50): qkv on g30 at T=128 built + measured DEAD; main unchanged (v135); 0 judge runs
- Budget 10-27: layer checks 20/20 (reset 00:00Z), 4 submits left. Board #30 at 2.1339x (#1 pjmullin3 2.399x).
- H100 timeline v16 sl (main v135): qkv cuBLAS 20.5 us (37.7 MB = 1.84 TB/s, the slowest GEMM per byte at T=128), FR pooler+router
  torch.mm 32.4 (75 MB, 2.3 TB/s), o 7.0, g30 unpool 24.6, twm 175.
- Standalone qkv 2304x8192 m128 (cold L2): cuBLAS 19.8-20.5 us also standalone. s99 g30 (no escapes, FP32 slab out, no reduce): S1 35.9,
  S2 21.0, S4 24.1. dev/s101_g30.cu = uneven K split (CTA s takes stages [s*KT/S, (s+1)*KT/S)): **S3 (108 CTAs) 16.4 us**, S6 20.5, S7 20.1.
- dev/s101_g30split.cu = g30_src.cu + split-K: escapes filtered per K range, FP32 slabs [S][128][N], __threadfence + self-resetting
  ticket per row group, the last CTA sums the slabs IN SLAB ORDER (deterministic) and stores BF16; other slabs fetched by cp.async
  all at once. Bit-exact (dev/s100_g30exact.py one-hot test, qkv S3 seeds 0/1/2/7, unpool S1/S5: 0 mismatches).
  Real weights (dev/s101_g30real.py): S3 21.4 vs cuBLAS 20.0 us. Ablations: escapes 0.5, fence+ticket 1.4, reduce 3.5 (first form,
  serial __ldcg loop, was 6.5): the reduce chain after the slowest split is ~5 us, and that kills it.
- In-layer ab_flag v16 (4 rounds x seeds 99/7): g30 qkv sl +4.1/+3.8 (first reduce +5.1/+5.5), full +0.1/+2.5 us: DEAD. Its H200
  odds are worse still (g30 is SM-ingest-bound, not HBM-bound). The split-capable kernel also costs unpool +0.7 us at S1 (ABAB 22.5/22.7
  vs 23.3/23.3), so main keeps the S1-only g30_src.cu; the split form lives only in dev/.
- Only open route for qkv at T=128: no BF16 reduce at all, i.e. attention sums 3 FP32 slabs at its loads (QSS-style; nvs had +1.6 us for
  2 slabs) for sliding only, and full layers would still need the BF16 path. Estimated <= -2 us at v16 sl (~0.08%): not built.
- tools/timeline.py: STMT env (exec'd in the module before capture) to time a flag/TUNE_BASE override.

## s102 (2026-10-27 16:22Z-, night4 session 51): T=128 pooler on split-K g30 slabs built + measured DEAD; main unchanged (v135); 0 judge runs
- Budget 10-27: layer checks 20/20 (reset 00:00Z), 4 submits left. Last piece of lead 13:30Z item (2): the FR pooler at T=128.
- Built (dev/s102_g30sp.patch, on 1c49030): g30_kernel<SPL> template (SPL: uneven K range per blockIdx.y, escapes filtered to the
  range, FP32 tile -> slab y of [S][128][N], no ticket/reduce; S1 path unchanged: unpool 24.4 us as before) + host g30_slab;
  layer TUNE option g30sp at T=128 (pooler on g30_slab S2, slabs summed by _rmsnorm_gain NS=2 ROUND, router hi/lo as its own
  cuBLAS torch.mm [T,512] FP32, _routing ST=512; 2 = router first, 3 = pooler first). bench test 13/13 with it forced.
- H100 ab_flag verify 16, 4 rounds x seeds 99/7, TUNE {}: base sl 300.6/295.7, full 434.8/421.8; g30sp 2 sl 308.9/303.0, full
  442.4/436.3; g30sp 3 sl 308.9/302.9, full 444.6/438.5 (+7..17 us).
- Timeline v16 sl: FR torch.mm 32.8 us (75 MB) vs g30 slab pooler 25.2 (50 MB 12-bit; standalone was 23.5) + router cuBLAS splitK
  7.6 + reduce 1.9 + routing 3.1 + rmsnorm 2.4; experts start 3.8 us later, twm +2.8. Even a free-standing perfect router (8 MB
  ~3 us) bounds it at ~-3 us H100 (~0.1%), and on the H200 g30 is per-SM x-ingest-bound (moe s43): DEAD. A g30 that also carries
  the router's hi/lo rows raw (BF16 stages) would need 136+ CTAs (2 waves) or a rebalanced split: est. <= -4 us H100, not built.
- Lead 13:30Z item (2) is closed: unpool T=128 g30 (v135, first check after 00:00Z), qkv (s101), pooler (s102), T=256 (s100) dead.

## s103 (2026-10-27 16:29Z-16:50Z, night4 session 52): no build, no judge runs (layer checks 20/20 used on 10-27)
- Board 16:30Z: layer #30 at 2.134x (35dd53d0); #1 pjmullin3 2.399x, #10 2.240x, #20 2.172x. No new Handoffs for layer; lead
  has not answered s102's lane request (last Human decision 13:30Z).
- Re-checked the one item flagged "untested" (s45): T=1 pooler W preloaded into SMEM by early-launched pr1 CTAs during
  csl/bnd (H200 ~8.5 us of idle HBM). Dead on paper by measured mechanism: every concurrent stream issued during csl slowed csl
  by about what it saved, whether L2 (G10_PFL2 +1.8 H100, pr1 npf 4/8 H200 tuner) or register prefetch: csl is load-latency-bound,
  and HBM queueing raises its latency whatever the destination. SMEM residency changes only eviction, not that contention. On top
  of that it needs a cross-stream spin flag (BND_SPIN was +2 us and racy) and co-residency with o/csl (g13/pr8 fill whole SMs,
  s86). Bound <= ~3 us d1 before those costs. Not built.
- Plan unchanged: 00:00Z check = main v135 (read 'layer tune T=128': g30:unpool False must lose), submit per the 09:28Z rule.

## s104 (2026-10-27 16:47Z-, night4 session 53): main = v136 = v135 + proj g30_sort + G30_PRE (merge of proj/g30pre 88ba088); 0 judge runs
- Lead 16:45Z LANE DECISION 5 for layer: (a) g30 for T=128 qkv and FR pooler: both already built + measured DEAD in s101 (qkv split-K
  reduce chain, in-layer +0.1..4.1 us) and s102 (pooler slabs + own-path router, +7..17 us); nothing new to build there. Only unbuilt
  form left: ONE g30 launch carrying pooler rows + the router's hi/lo rows (no split-K, router out FP32 [T,512]), s102 bound <= -4 us
  H100 (~0.15%), needs 136+ CTAs or a rebalanced grid, and the H200 may be x-ingest-bound (moe s43). Queued, not started.
  (b) G30_PRE merged: g30_src.cu now = proj s92 sorted escape candidates (pop = register shift) + s94 consumers read/sort the
  escape header before griddepcontrol.wait (header = prepared weight, never written in a replay; out is written only after
  x arrives, which waits behind warp 9's griddepcontrol.wait; with G30_ESC 0 consumers skip the wait but still order behind x).
- validate --quick ALL OK: test 13/13, scored 0/12, opt 0/316, tk2f 0/300, replay 0/240, route 0/240 (gap 0.0256), fuzz 0/464,
  cold prepare 117.8 s.
- H100 ab_files "verify 16" 6 rounds (tuner on, as shipped): v135 sl 289.28 / full 436.18 vs v136 289.09 / 436.19 us = tie
  (proj's tuner-off ab: sl 0..-1.6, full -0.3..-2.2). Kept: bitwise equal, no cost. ab_files filter is the full shape name
  ("verify 16"), "v16" matches nothing and silently times no shape.
- 00:00Z check = main v136 (read 'layer tune T=128': g30:unpool False must lose).
- One-launch pooler+router g30 feasibility (judge weights, layer._compress): pooler [4096x8192] ok (<= 7 esc/row), router hi
  [256x8192] ok (3 esc/row, emax 124 vs pooler 123), router lo (FP32 - BF16 hi residual) NOT compressible: 18 esc/row > G13_ESC 8
  even in its own window (emax 116); concat [pooler; hi; lo] -> 1000 esc/row. So the fused form needs raw BF16 stages for the 256
  lo rows (new producer/consumer path + FP32 epilogue) or 24+ escape slots (register cost in the consumers). Bound <= -4 us H100
  v16 (~0.15%), H200 maybe x-ingest-bound: not started; next session's candidate if nothing better arrives.

## s105 (2026-10-27 17:01Z-17:05Z, night4 session 54): one-launch pooler+router g30 at T=128 measured DEAD (H100 standalone); 0 judge runs
- Layer checks 20/20 used on 10-27; sandbox moves ~18:00Z. Measured the last queued g30 form (s104) before building it.
- dev/s105_frbound.py (main g30_gemm, S=1) and dev/s105_slabbound.py (s102 g30_slab: old g30_src @1d405c8 + the g30_src.cu hunks
  of dev/s102_g30sp.patch, path as argv[1]); extra rows = pooler copies (12-bit, cheapest case: raw-BF16 lo rows would cost more).
  H100 T=128 cold L2, cuBLAS FR torch.mm [4608 x 8192] FP32 out = 29.3 us (pooler alone 26.4):
  | g30 form | N=4096 | +256 | +512 | +640 | +768 |
  | S1 g30_gemm | 36.3 | 36.8 | 37.0 | 37.1 | 37.1 |
  | S2 g30_slab | 25.6 | 43.7 | 43.8 | 43.7 | 44.0 |
  | S3 g30_slab | 33.1 | 33.5 | 33.8 | 33.8 | 34.0 |
  S2 at N=4096 is exactly 128 CTAs (one wave on 132 SMs); any router rows make it 136+ CTAs = 2 waves. S3 and S1 are already slower
  than cuBLAS FR before escapes, FP32 router out and the slab sum. Uneven splits leave unsplit CTAs doing 2x K (~S1 time), and
  128-row CTAs were dead in s99. So the fused pooler+router g30 is DEAD whatever the router-lo format. g30 at T=128 = unpool only (v136).
- main unchanged = v136; first check after 00:00Z = v136 (read 'layer tune T=128': g30:unpool False must lose).

## s106 (2026-10-27 17:19Z-17:30Z, night4 session 55): no build, 0 judge runs
- Checks 20/20 used for 10-27 (4 submits left, unusable without a check win); sandbox moves ~18:00Z (STOP ~17:55Z).
- proj s96 claimed the g30 T=16/32 port into the layer (branch proj/g30n; proj STATE s95: d32 qkv -1.1 pool -2.6 unp -2.9 o -1.3,
  d16 qkv -0.5 pool -0.4 unp -1.5 us per call on H100 vs g24). Not duplicated here (same files).
- 00:00Z plan: check main v136 first; then v136 + proj/g30n after proj's validate ALL OK and cold prepare <= 200 s.

## s107 (2026-10-27 17:35Z-, night4 session 56): main = v137 = v136 + proj/g30n (453790c); 0 judge runs
- Checks 20/20 used for 10-27; sandbox moves ~18:00Z. Merged origin/proj/g30n (proj s95/s96/s97: g30 at T=16/32 with G30_MTOK + G30_S
  cluster split-K; wired as TUNE[32] option {"g30n:unpool": True}, off by default, G30N_DEFAULT_ON empty). proj: H100 TUNE-off d32
  unpool -1.2..-2.1 us; same-process A/B synth full tie, sl -1.4, real d32 sl -2.1 (s97: the s96 "+9 us at full" was a tuner artefact).
  Pooler g30n dead in-layer (router SM contention), T=16 tie. proj validate --quick ALL OK, cold 122.3 s.
- build.py regenerates submission.py identically to proj's commit (git status clean after build).
- 00:00Z: check 1 = v137 (covers v136 + g30n; read 'layer tune T=32' g30n lines; if the H200 picks it at full too, consider TUNE_BASE).
- My re-validate of v137 (validate --quick, 17:35-17:43Z): ALL OK: bench test 13/13, scored 0/12, opt_test 0/318, tk2f_stress 0/300, replay 0/240, route_judge 0/240 (worst gap 0.0256), stream_fuzz 0/467, cold prepare 125.3 s (<= 200 s rule). Ready for the 00:00Z check.

## night5 s1 (2026-10-27 18:57Z-, sandbox okc-night5 session 1): v137 re-validated on night5; v138 = v137 + g30n T=16/32 tuner options; 0 judge runs
- Layer checks 20/20 used for 10-27 (4 submits left, unusable). Board #30 at 2.1339x (35dd53d0); #1 pjmullin3 2.399x, #10 2.240x, #20 2.172x.
- night4's profiles/ were not in git (gitignored), so the board run's H200 profile was lost. Recovery: `okc status <FULL 16-hex job id> --json`
  (short ids give "no such job") writes profiles/<job>.{txt,json,log} into the cwd. Restored 35dd53d0d33246a1 into profiles/.
  Board run per kernel (H200 us): d1 sl tk2f 20.2, pr1 17.5, qkv g13 10.9, bnd 2x3.1, o g13 6.1, n2u 5.0, csl 3.9; d32 sl tk2f 102.0,
  g24 pooler+unpool 47.0, qkv g24 13.5, nds 10.3, norm2 10.1, bnd 8.0, router_split 6.4, o nvjet 6.2.
- main v137 on night5 H100: validate --quick ALL OK (test 13/13, scored 0/12, opt 0/318, tk2f 0/300, replay 0/240, route 0/240 gap
  0.0256, fuzz 0/467), cold prepare 137.7 s (night4: 125.3; fresh machine, still <= 200 s).
- v138 (variants/v138.py, layer.py TUNE only): TUNE[16] += {g30n:unpool}, {g30n:qkv_gate}, {both}; TUNE[32] += {g30n:qkv_gate},
  {g30n:unpool + g30n:qkv_gate}. All builds already in G30N_CFG (no new compile). proj s95/s96 H100: T=16 unpool/qkv tie in-layer,
  standalone d32 qkv S3 -1.1, d16 qkv -0.5, unp -1.5 us per call; the H200 is more consumer-bound than the H100 (g24 at ~2.0 TB/s
  at d32), so let the H200 tuner decide on the one 00:00Z check that is happening anyway.
- v138 validate --quick ALL OK: test 13/13, scored 0/12, opt 0/328, tk2f 0/300, replay 0/240, route 0/240 (gap 0.0256), fuzz 0/487,
  cold prepare 144.3 s. H100 tuner lines (prep step): T=16 sl default 219.7, g30n unp 219.5 / qkv 219.4 / both 219.5 (tie; picked g24:o False);
  T=32 sl default 222.0, unp 220.1, qkv 221.5, unp+qkv 220.0 -> picked unp+qkv; T=32 full default 440.7, g30n 439.7-440.5 (tie).
- **main = v138.** 00:00Z check 1 = v138 (covers v136 + g30n): grep 'layer tune T=16' / 'T=32' g30n entries in the check log.

## night5 s2 (2026-10-27 19:31Z-, okc-night5 session 2): v139 = v138 + ZR (G24X zero moved into the pre-expert rmsnorm); 0 judge runs
- Layer checks 20/20 used for 10-27 (resets 00:00Z). Board #31 at 2.1339x; #1 pjmullin3 2.399x, #10 2.240x, #20 2.172x.
- Per-load speedups of the board run (load = 46 sl + 16 full): d1 2.53, d16 2.08, **d32 1.78** (weakest), v1 2.32, v16 2.02, v32 2.16.
- H100 timeline d32 sl (v138, tuner picks g30n:unpool): the layer ENDED with a 1.2 us `vectorized_elementwise` = kernel_fn's trailing
  `_G24X.pop().zero_()` (G24X invariant: qkv's persistent FP32 out must be zero between calls; g24 unpool zeroes it as its ZN, g30n
  unpool does not, so the fallback zero_ ran on the critical path after the last boundary). This also biased the tuner against g30n:unpool by ~1.3 us.
- v139 ZR (ZR_ON, layer.py): when unpool will run g30n, `_rmsnorm_gain` (pre-expert, grid = T rows) zeroes row `row` of the G24X buffer
  after its gdc_wait (qkv's out is dead after o; the wait is transitive over the earlier kernels). Trailing kernel gone (timeline).
  H100 ab_flag "decode 32" 6 rounds SEEDS 99,7, ZR off vs on: sl 194.58 -> 193.24 (s99), 199.75 -> 198.52 (s7); full 419.96 -> 419.03, 428.57 -> 429.17 (noise).
- Also: v138's TUNE {g30n:qkv_gate} options were NO-OPS (G30N_CFG qkv S=3 needs K % 384 == 0; 8192 % 384 != 0, so _g30n_prepare skips it
  silently): removed from TUNE[16]/TUNE[32] (less tuner time; qkv at T=16/32 is still g24).
- Observation to re-check on an idle GPU (validate was running beside it): H100 d16 FULL timeline showed router_split+routing (153-163 us)
  BEFORE the pooler g24 (163-186), i.e. serialized, while d16 sliding overlaps them (pooler first, router beside it). If real, ~10 us at d16 full.
- v139 validate --quick ALL OK: test 13/13, scored 0/12, opt 0/320, tk2f 0/300, replay 0/240, route 0/240 (gap 0.0256), fuzz 0/471,
  cold prepare 166.5 s (proj's prep_time + builds ran at the same time: load 17).
- **main = v139.** 00:00Z check 1 = v139 (covers v136 g30 + v137/138 g30n unpool + ZR): read 'layer tune T=16/T=32' g30n:unpool lines.
- **STREAM-POOL COLLISION (found n5s2, fixed in v140).** The d16 serialization above is NOT an option effect: the FIRST graph capture
  after prepare() (any shape) runs the side branch in line (profiler: 1 stream; router_split -> routing -> pooler g24 serial), later
  captures overlap (2 streams). Cause: torch.cuda.Stream() hands out 32 pooled streams per priority round-robin; state["side_stream"]
  was a pool stream, and bench.capture's `torch.cuda.Stream()` right after prepare() got EXACTLY that stream (tools: /tmp/cap_var4.py
  printed [True, False x31, True, ...] for the next 40 pool streams == side). Capture stream == side stream => side_stream.wait_stream
  is a no-op and the "parallel" router/routing/tk2f/... are captured in line. Cost H100 d16 sl: first capture 201.6 vs 192.2 us
  (no profiler, time_graph). Any harness that captures on a pool stream (torch.cuda.Stream() or torch.cuda.graph()'s default
  capture stream, both from the same pool) hits it on every 32nd capture: on the judge some shape(s) per run, depending on how many
  pool streams the harness made before; our own tuner hit it on every 32nd candidate (layer.py makes one pool stream per
  candidate), i.e. random ~10 us penalties on single options at side-stream shapes (T=16/32/128/256).
- v140 fix: `_own_stream()` = cudaStreamCreateWithPriority(NonBlocking) via ctypes libcudart.so.13 (the runtime torch already
  loaded) wrapped in torch.cuda.ExternalStream, for side_stream and hi_stream (fallback: pool stream + print). After: next 40 pool
  streams never equal side; first capture d16 sl 191.9 us (= later captures 191.3-191.9).
- v140 validate --quick ALL OK: test 13/13, scored 0/12, opt 0/320, tk2f 0/300, replay 0/240, route 0/240 (gap 0.0256), fuzz 0/466, cold prepare 151.4 s.
- **main = v140.** 00:00Z check 1 = v140. What to read: per-shape d16/d32/v16/v32 vs the board run (a judge shape that used to land on
  a colliding capture should drop ~5%); tuner lines should no longer show single ~+10 us outliers at T=16/32.

## night5 s3 (2026-10-27 20:55Z-, okc-night5 session 3): v141 = v140 + proj/g30k139; 0 judge runs
- Layer checks 20/20 for 10-27 (resets 00:00Z). Merged origin/proj/g30k139 deb4a62 (proj s99/s100/n5s3: g30k stream-K pooler on 104/120
  CTAs with FP32 slabs summed in the rmsnorm; g30n qkv_gate built with uneven S3 K split (my v139 'K % 384' removal was wrong: proj's
  g30 splits K unevenly); moe n5s1 pr32 = T=16/32 pooler+router+routing in one launch, embedded via build.py, built on _BUILD_POOL)
  onto v140 (own streams). Clean auto-merge; build.py regenerates submission.py identically.
- v141 validate --quick ALL OK: test 13/13, scored 0/12, opt 0/344, tk2f 0/300, replay 0/240, route 0/240 (gap 0.0256), fuzz 0/508,
  cold prepare 153.6 s (<= 190 s lead rule; no option drop needed).
- H100 tuner (validate prep): T=32 sl default 220.3 | g30n unp 217.1 | pr32 219.8 | pr32+unp 216.7 | unp+qkv 216.3 | g30k104 218.9 |
  g30k120 231.4 | g30k104+unp 216.0 | g30k104+unp+qkv 215.0 (picked). T=32 full default 439.8 | g30k104+unp 435.1 | +qkv 435.3 | pr32+unp 436.4.
  T=16 sl default 219.5, g30k104 220.4 (ties); T=16 full default 315.8, g30k104 313.1.
- **main = v141.** 00:00Z check 1 = v141; read the H200 'layer tune T=16/T=32' lines (g30k/g30n/pr32) and copy them to NOTES + moe/proj.
- H100 timeline d32 sl v141 (tuner picks): qkv g30 0-16.0 | nds/combine 16.7-35.0 | o nvjet 35.5-42.0 | bnd 42.4-45.7 | g30k pooler 43.4-71.2
  beside router_split 46.2-67.8 + routing -72.0 (balanced: g30k ends 0.8 us before routing) | rmsnorm 72.3-75.0 | tk2f 73.5-171.3 | norm2 -175.2 |
  g30 unpool 164.8-193.2 (18 us after norm2 = ~2 TB/s on 36 MB, still the biggest non-expert post-dependency slice) | bnd -195.7.
- Git gotcha: `git merge --no-ff` + `git pull --rebase` LINEARIZES the merge and DROPS the merge commit's own extra changes (my notes vanished;
  code tree was identical). Restored by `git checkout <merge> -- NOTES STATE`. Next time: commit notes separately, or `git pull --rebase=merges`.

## night5 s4 (2026-10-27 21:07Z-, okc-night5 session 4): v142 = v141 + g30n o at T=16/32; 0 judge runs
- Layer checks 20/20 for 10-27 (resets 00:00Z). Board #33 at 2.1339x; #1 pjmullin3 2.399x, #10 2.249x, #20 2.172x.
- Routing census (reference router, 5 seeds, sliding): local routes / distinct local experts: d1 2-5 / 2-5, d16 24-30 / 7-10,
  d32 48-57 / 8-10, v1 10-16 / 6-8, v16 163-181 / 15-19, v32 320-355 / 18-23 (x 28.3 MB NVFP4+scales per expert).
- DEAD (standalone H100, /tmp/lay_s4/qk_sk.py): qkv at T=16/32 as g30k stream-K on more CTAs (proj's "qkv on 132 CTAs" item).
  qkv_gate 2304x8192, 12-bit, graph cold L2, minus empty graph: m16 g30n S3 (108 CTAs) 15.6 | g30k G72 17.2, G96 15.7, G108 14.8-15.1,
  G120 15.7, G132 15.8; m32 g30n 16.35 | G108 15.6-15.9, G132 16.4. CTA count is not qkv's limit (ramp/latency is); FP32-red
  variant not worth building.
- NEW g30n o (layer.py G30N_CFG ("o", 16/32): (1, 16, 16), existing builds): o = 8192x1024, K 1024, so each of the 128 CTAs'
  W share (~98 KB 12-bit) is about its 16-stage ring: as a PDL secondary the W loads before griddepcontrol.wait. o_pf (attention's
  L2 prefetch of o's weight, O_PF[32] 0.5) now points at the g30 pack when g30n runs o.
  H100 ab_flag TUNE {} (variants/n5s4_off.py), 4 rounds A B A, seeds 99/7, us:
    d32 sl 194.35/203.06 -> 192.62/201.32, full 422.27/431.52 -> 419.78/429.81; d16 sl (vs g27) 190.49/154.77 -> 191.42/155.38 (+0.8),
    d16 full 299.94/281.37 -> 297.53/278.94 (-2.4). Repeat d32 sl: 195.56/204.13 -> 193.70/202.36.
  Full layers gain more: _red_fin already triggers dependents early (rf_trig), so o's W ring fills during the merge.
- _combine_hb early trigger (attn_src CB_HB_TRIG, TRIG constexpr on the nds/nvs/gluon launch sites): d32 sl with g30n o -0.3/-0.4 us
  = noise; left False (code kept, inert).
- v142: TUNE_BASE (32, sl/full) and (16, full) += {"g30n:o": True}; TUNE[32] += {g30n:o False}, {g30n:o False, lt:o True};
  TUNE[16] += {g30n:o True} (sl), {g30n:o False}, {g30n:o False, g27:o False}. Note: g30n is checked before g27/g24/lt in _g19,
  so with g30n:o on, options that only set lt:o/g27:o/g24:o/g19:o are no-ops.
- v142 validate --quick ALL OK: test 13/13, scored 0/12, opt 0/354, tk2f 0/300, replay 0/240, route 0/240 (gap 0.0256), fuzz 0/528,
  cold prepare 174.2 s (v141 153.6; +5 tuner candidates and other agents' load; still <= 190 s). H100 tuner agrees: T=32 sl default
  (g30n o) 217.4 vs {g30n:o False} 219.2, full 436.6 vs 438.0; T=16 full 312.1 vs 313.9; picks T=32 g30k104+unp+qkv as in v141.
- **main = v142.** 00:00Z check 1 = v142 (covers v140 streams + v141 g30k/g30n/pr32 + g30n o). Read 'layer tune T=16/32' lines:
  {g30n:o False} must lose at T=32 sl/full and T=16 full; if it wins on the H200, move g30n:o back to an option.
- v143 = v142 + moe n5s3 PR_SA pr32 (pr32_src.cu = r-moe 11d49b7 muse_moe_w4a16/tools/n5s1/pr32_src.cu, the committed handoff
  version; moe's later uncommitted PR_HL edits NOT taken). pr32 stays a TUNE option. validate --quick ALL OK: test 13/13, scored
  0/12, opt 0/354, tk2f 0/300, replay 0/240, route 0/240 (gap 0.0256), fuzz 0/528, cold prepare 155.6 s.
- **main = v143.** 00:00Z check 1 = v143. Idea not built: g30 at MTOK 8 for the T=8 o (v1; o = g12 BF16 8.1 us H200, OB8 path).

## night5 s5 (2026-10-27 22:01Z-, okc-night5 session 5): v144 = v143 + g30n G30_MTOK 8 (T=8 o tuner option); 0 judge runs
- Layer checks 20/20 for 10-27 (resets 00:00Z; 4 submits left, unusable without a check). Board #33 at 2.1339x; #1 pjmullin3 2.399x.
- g30_src.cu: G30_MTOK 8 = wgmma m64n8k16 (4 FP32 acc/thread), final store loop generalized (MTOK 8 = 64 uint4, threads 0..63;
  MTOK >= 16 unchanged: same loop count, guard constant-true). x stage = 1 KB (8 tok x 64 k, SW128 = one atom).
- Standalone H100 (/tmp/lay_s5/o8.py, o = [8192, 1024], m 8, cold L2, minus empty graph): g30n S1 NW16/NS16 8.0 us, NW24 8.15,
  NS32 8.15; cuBLAS 8.9. rel err 7e-5 vs F.linear (FP32 order; not bitwise, like the T=16/32 g30n).
- In-layer H100 ab_flag (TUNE {}, seeds 99/7, 4 rounds A B A), v1 = verify 1:
  NW16/NS16: sl 143.63/116.58 -> 144.15/117.42 (+0.5..0.8), full 157.60/129.61 -> 158.34/130.72 (+0.7..1.1): loses.
  NW8/NS8:   sl 143.41/116.66 -> 142.62/116.39 (-0.3..-0.8), full 156.62/129.89 -> 158.20/130.87 (+1.0..1.6): weighted wash.
- Why (timelines /tmp/lay_s5/tl_*.txt, v1 sl): o ends ~29 us either way; g12 o ends 7.4 us after vtk, g30 NW16 8.8, NW8 6.2;
  pr8 (pooler+router, PDL secondary of the boundary, ~34 us H100) is the long pole and ends at ~55 us whatever o does
  (= s88: "pr8 ends at the same time whatever precedes it"). A faster T=8 o does not shorten v1; only pr8's own stream does.
- v144: G30N_CFG ("o", 8): (1, 8, 8); TUNE[8] += {g30n:o True} (H200 tuner decides per window; default off). ob8 is now also off
  when g30n:o runs at T=8 (OB8 is False anyway).
- Not tried (dead on paper): qkv at T=8 on g30n S3 (s4: m16 g30n S3 15.6 us standalone vs g14 m8 14.6 in-layer; cold-stream ramp-bound).
- v144 validate --quick ALL OK: test 13/13, scored 0/12, opt 0/356 (+2 = g30n:o forced at T=8 sl/full), tk2f 0/300, replay 0/240,
  route 0/240 (gap 0.0256), fuzz 0/530, cold prepare 155.6 s. H100 tuner T=8 sl: default 145.9 | {g30n:o} 146.8 (picked nu8d/nu8x off);
  T=8 full: default 159.2 | {g30n:o} 158.9 (picked cslpw + pr8_npf 220). H200 decides at 00:00Z.
- **main = v144.** 00:00Z check 1 = v144 (= v143 + this option; nothing else changed). Read 'layer tune T=8' {g30n:o} lines too.

## night5 s6 (2026-10-27 22:28Z-, okc-night5 session 6): T=8 pooler on g30k MTOK 8 (no pr8) measured DEAD; v144 check at 00:00Z
- Layer checks 20/20 for 10-27 at session start. Board #33 at 2.1339x; #1 pjmullin3 2.399x, #10 2.249x, #20 2.180x.
- DEAD (H100 in-layer, TUNE {}, variants/n5s6_*.py, seeds 99/7, 4 rounds): v1 = {pr1: False} + T=8 pooler on g30k (stream-K,
  G30K_CFG ("pooler", 8) G 88/104/120, NW16/NS16, MTOK 8 builds and passes the prepare check) beside the side-stream r3 router.
  v1 sl s99: pr8 144.4 | pr1 False (g14 pooler) 149.3 | g30k G104 149.8 | G88 150.1 | G120 149.8; full s99 155.1 | 160.0 | 161.9 | 162.9 | 162.1.
  Timelines (v1 sl): pr8 ends 23.5 us after the boundary; g30k pooler also ~24 us after it (its 104 CTAs share HBM/SMs with the
  router, which stretches 6.6 -> 27 us beside it); routing ends 6 us after pr8 would. Alone the r3 router is 6.6 us and the g14
  pooler 23 us: serial = no better either. pr8's fusion (router items inside the pooler blocks) is worth ~5 us at T=8;
  a g30 pooler would need the router fused like pr32 (FP32 router at T=8: hi/lo x hi/lo products, a new port) to have a chance,
  and even standalone it is ~17 us vs pr8's 23.5 for 56 MB: <= ~3 us H100, ~0.3% v1. Not queued. Change reverted (prepare time).
- **00:00Z: check 1 = v144 REJECTED at upload (422: "submission is larger than 1000000 bytes"; no check spent).** submission.py had
  been 1.050-1.051 MB since ~v141 (pr32 + g30 sources); no night5 check had run to notice. Fix (v145, d372b7e): build.py
  strip_comments() drops Python COMMENT tokens + the blank lines left (rows inside multi-line strings untouched; CUDA sources keep
  their comments), asserts the AST is unchanged and the file < 1,000,000 B: 963.7 KB. Headroom only ~36 KB: drop dead embedded
  sources (ob8 9 KB, g22 22 KB) before adding big new ones. v145 validate --quick ALL OK (opt 0/356, fuzz 0/530), cold 155.6 s.
- **check 4937830f (v145 = v144 code) 2.0907x** (board 2.1339x = 35dd53d0 v125). Per shape ours/SGLang us: d1 69.7/173.3, 76.5/181.3;
  d16 130.4/269.2, 210.6/433.6; d32 155.8/258.6, 312.8/612.9; v1 115.3/267.4, 125.2/278.5; v16 274.5/533.9, 369.8/715.8;
  v32 348.0/726.0, 532.4/1099.9. vs board run: d32 sl -8.5 us (g30k/g30n: the targeted shape, best ever), v1 full -2.1,
  d1 +2.3/+1.5, d16 +5.5/+6.1, v16 full +17, v32 +26/+38 (H200 tuner T=256 defaults 344.6/541.4 vs 336.1/521.4 in the board
  run's log, every candidate +8..15: the T=256 default has ranged 336-351 over past checks = node noise). Not submitted (< 1% rule).
  H200 picks: T=1 full {g10alt:o 1}; T=8 sl {nu8d/nu8x False}, full {cslpw, pr8_npf 220} (g30n:o not picked); T=16 sl {tk2fj False},
  full {bal}; T=32 sl {g30k:pooler 104, g30n:unpool} (169.2 -> 162.7 confirm), full + g30n:qkv_gate (329.6 -> 323.2);
  T=128 {lt:o}; T=256 sl {lt:unpool}, full {rg}. pr32 never picked (moe), g30k104 at T=16 full not picked on H200.
- v146 (2f105a0) = v145 + proj G30_PL (r-layer proj/g30pl 19d3f19: g30_src.cu only). validate --quick ALL OK; cold prepare 189.7 s
  during other agents' compiles (load avg ~30), 168.3 s on rerun (proj measured 156.2 s for the same tree).
- **check 133068b8 (v146) 2.1097x**: d1 69.1/171.5, 75.8/181.2; d16 128.4/265.3, 208.4/435.4; d32 152.2/253.7, 295.5/614.8;
  v1 116.3/268.8, 122.1/275.7; v16 276.3/544.3, 357.5/707.4; v32 333.1/697.8, 523.3/1074.7. vs board run 35dd53d0: d32 sl -12.1,
  full -12.5 (2nd check in a row: real, g30k/g30n at T=32), v1 full -5.2, v16 full +4.8, v32 +11/+29 (noise), rest +-2.
  1.13% below board: misses the 09:28Z rule's 1% (>= 2.1126x) by 0.003: NOT submitted.
  H200 picks: T=32 sl AND full {g30k:pooler 104, g30n:unpool, g30n:qkv_gate} (-5.4/-5.5 confirm); T=8 sl {cslpw, pr8_npf 220}, full
  {g14alt:qkv_gate 1}; T=16 sl default, full {bal}; T=128 sl {lt:o, lt:unpool}, full {twf_split 1022}; T=256 sl {rg}, full default.
  pr32 / g30n:o (T=8/16) never picked in 2 checks.
- v147 = v146 + build.py DEAD_SRCS ("22", "OB8") not embedded (G22_CFG {} and OB8 False: never built): 933.6 KB (66 KB headroom).
  validate --quick ALL OK (opt 0/356, fuzz 0/530), cold 154.1 s. Same kernels as v146 (no check needed for it alone).
- Next: TUNE_BASE[(32, *)] could take {g30k:pooler 104, g30n:unpool, g30n:qkv_gate} (picked in both checks; saves nothing on the
  judge since the tuner already finds it, but removes the pick risk). The d32 win is in; gap to the board's lucky draw is noise
  on v16/v32: keep stacking real per-shape wins and submit once a check lands >= 1% rule or > 2.1339x.

## night5 s7 (2026-10-28 00:57Z-, okc-night5 session 7): v148 = T=32 g30 picks in TUNE_BASE; check f824b925 2.0944x (not submitted)
- v148 (c0413d8) = v147 + TUNE_BASE[(32, sl/full)] += {g30k:pooler 104, g30n:unpool, g30n:qkv_gate} (H200 picked that set at T=32 in
  4937830f and 133068b8). TUNE[32]: the 4 options that set it removed; escapes {g30n:qkv_gate False}, {g30n:unpool False},
  {g30k:pooler 0}, all three off, {pr32 + g30k 0} added (g30k:pooler 120 kept).
- validate --quick ALL OK: test 13/13, scored 0/12, opt 0/350, tk2f 0/300, replay 0/240, route 0/240 (gap 0.0256), fuzz 0/520;
  cold prepare 199.8 s inside validate (other agents' compiles running), rerun cold (fresh caches) 154.4 s = v147's figure.
- **check f824b925 (v148) 2.0944x** (CI 2.063-2.126). ours/SGLang us: d1 67.5/170.9, 76.9/180.8; d16 121.3/251.9, 211.5/442.5;
  d32 154.2/263.6, 307.9/612.9; v1 108.1/258.0, 126.8/281.5; v16 274.9/531.2, 362.4/699.0; v32 378.5/740.1, 536.7/1094.6.
  H200 tune T=32 sl: default 168.9 vs every escape 169.5-172.4 (g30k 120 182.7); full 326.8 vs 327.4-329.5: the base is right.
  vs 133068b8 (v146, same kernels): d32 full +12.4 (295.5 -> 307.9), v32 sl +45, d16 sl -7, v1 sl -8: node noise both ways.
  Not submitted (< 2.1126x, the 09:28Z rule's 1%).

## night5 s8 (2026-10-28 01:18Z-, okc-night5 session 8): tuner picks vs REAL routings -> T=256 base {rg, twf_split 66}, T=128/256 options pruned
- Why: f824b925's v32 sl ran 378.5 us (ratio 1.956) vs 333.1 (2.095) in 133068b8 on the same kernels; its H200 tuner picked
  T=256 sl {hot: 24} (133068b8 picked {rg}). The tuner times ONE synthetic routing (SYNTH_P, seed 1234); the judge's inputs
  route very differently per input (verify 32: 2nd expert 38..88 of 256 tokens over reference seeds 1-5/7/13/99), and the
  judge picks the router per run (reference.weights_fields: router_seed hitting a target load).
- H100 ab_flag on REAL inputs (variants/n5s8_notune.py = main with TUNE = {}; TUNE_BASE swapped per stmt; seeds 99/7/13,
  3 rounds; /tmp/n5s8_ab256.txt, /tmp/n5s8_ab128.txt). us vs base, sl seeds | full seeds:
  * T=256 (verify 32): hot 24 +8/+23/+21 | +1/+18/-8; hot 0 +14/+6/+16 | +8/0/-8; twq 4/12 +14..30; gnorm off +6..10;
    fr off / bkt off / twm_hot +1..6; rg +10/-2/-6 | -5/-4/-15; **rg + twf_split 66 -1/-9/-10 | -13/-14/-5** (1022 same);
    lt:* / twm_order / nvscl: +-1 of each other (ab_flag shows a ~+3 us sl bias on every non-first stmt).
  * T=128 (verify 16): twf_split 66/1022 -9/+11/-1 | +1/-6/+5 (flips sign per routing: a tuner coin flip); g30:unpool off
    +6..8; bkt off +2..4 sl; gnorm off +4..5 sl; lt:qkv_gate off +2..3; fr off +1 sl / -3..-6 full; lt:o/unpool/ltalt ties.
- Tried first and reverted: tuner timing every candidate on 3 synthetic routing draws (SYNTH_P + the shares of reference
  seeds 1 and 4): on the H100 it still saw hot 24 as a tie (402.7 vs 404.3 us), i.e. the synthetic stream misses whatever makes
  hot 24 lose on real inputs (token correlation within verify blocks? not pinned down). Pruning on real-input A/Bs instead.
- v149: TUNE_BASE[(256, *)] = {hot 32, rg, twf_split 66}; TUNE[256] 36 -> 10 options (no rg/twf-off escapes: the synthetic
  tuner must not undo a real-routing win; twf 1022, lt:*,
  twm_order 76, twm_tail 32, hot 40, nvscl, lt:qkv_gate off); T=16 full base += bal; TUNE[128] 22 -> 10 (twf_split, bkt/gnorm/g30:unpool off, o_pf 0, lt:qkv_gate off dropped).
- Decode sweep (same method, stmts applied to T=16 AND T=32 bases; /tmp/n5s8_abdec.txt): d16 full bal True -2.5/-4.0/-1.5 us (all
  seeds; H200 picked {bal} at T=16 full in 4937830f + 133068b8) -> in TUNE_BASE[(16, False)]. d16 sl bal off +4, tk2h 4 +2..6 at T=16;
  d32: bal / tk2fj / tk2h 0 / tk2h 4 all within +-1.5 (mixed signs): unchanged.
- v149 (bffbb79) validate --quick ALL OK (test 13/13, scored 0/12, opt 0/280, tk2f 0/300, replay 0/240, route 0/240 gap 0.0256,
  fuzz 0/444), cold prepare 146.7 s (_tune 24.7 s, was ~30).
- **check 73c50eea (v149) 2.1048x** (CI 2.073-2.136). ours/SGLang us: d1 67.4/172.0, 75.4/182.1; d16 125.8/259.5, 209.2/432.6;
  d32 152.3/253.7, 307.5/611.1; v1 111.9/261.9, 123.1/273.7; v16 268.7/517.0, 362.8/702.7; v32 343.3/719.0, 513.9/1073.6.
  Ratios v32 sl/full 2.095/2.089 (last 3 checks 2.095/2.054, 2.086/2.066, 1.956/2.039: full best of 4, sl = the good draws;
  board run 2.168/2.132 was a lucky draw). v16 1.924/1.937 (in band). Not submitted (< 2.1126x).
  H200 picks: T=256 sl {lt:unpool}, full {hot 40} (+0.6 us real-routing tie at full, +2..9 sl: dropped in v150); T=128 sl default,
  full {lt:o}; T=16 sl {g30n:unpool}, full {g30n:unpool, g30n:qkv_gate} (by 0.6-0.9 us: ties).
- v150 = v149 - TUNE[256] {hot 40} (tuner list only; bench test 13/13). Not checked.
- Lesson: the tuner's synthetic routing does not represent the judge's inputs for routing-sensitive expert knobs (hot, twf_split,
  twq, bkt); decide those on real-input H100 A/Bs over >= 3 seeds and pin them in TUNE_BASE; keep tuner options to
  routing-insensitive GEMM/attention picks. Still open: why synthetic sees hot 24 as a tie (not pinned down).

## night5 s9 (2026-10-28 02:08Z-, okc-night5 session 9): v151 = T=8 base pin {nu8d/nu8x off, cslpw, pr8_npf 220} on real routings
- Board: #42 at 2.1339x (#1 pjmullin3 2.399x, #10 2.254x, #20 2.198x). 10-28 budget at start: 16 checks / 10 submits.
- H100 v1 sl timeline (main v150, seed 99): g14 qkv 0-14.0, vtk 12.0-19.1, g12 o 13.7-28.6, bnd 14.8-31.4, pr8 18.6-53.5,
  norm 31.5-55.6, tk2f 54.0-137.9 (84 us = 57% of the layer: moe's LANE 6 bs8 item), nu8 132.4-143.1, bnd 137.7-146.7.
  bnd ends 2.8 us after o on the H100 (moe audit said 4.0); o->bnd->pr8 not attacked this session.
- Real-routing A/B (variants/n5s9_notune.py = main with TUNE = {}; tools/ab_flag.py, TUNE_BASE swapped per stmt; /tmp/n5s9_ab1.txt,
  /tmp/n5s9_ab8.txt). us vs base:
  * T=1 + T=8 tk2fx off: d1 sl +1.5/+0.8/+0.9, full +0.7/+1.1/+0.7; v1 sl +1.7/+0.7/+1.6, full +1.4/+0.5/+1.3: base right; option
    dropped from TUNE[1] (and TUNE[8]).
  * T=1 kv_pf off: -0.2..+0.2 (tie; H100 tuner picked it at full this run): unchanged.
  * T=8 (seeds 99/7/13/5, 4 rounds): nu8d/nu8x off sl +0.1/-0.8/-1.1/+0.7, full -0.1/-1.2/-0.2/-1.6; cslpw+pr8_npf 220 sl -0.2/-0.7/
    -1.2/-0.3, full -0.2/-0.3/-0.3/-0.5; **both: sl -0.4/-1.8/-2.7/-0.4, full -0.3/-1.5/-0.8/-1.8** (every seed) -> TUNE_BASE[(8, *)]
    += _T8N. TUNE[8] rebuilt around the new base (escapes: nu8d/nu8x on, nu8x on, cslpw off (+npf 212), npf 206/212/112, g14alt 0-3,
    nu8 off, g19 qkv, g30n:o).
- v151 validate --quick ALL OK: test 0/13, scored 0/12, opt 0/264, tk2f 0/300, replay 0/240, route 0/240 (gap 0.0256), fuzz 0/428,
  cold prepare 162.5 s.
- **check bb8ea548 (v151) 2.1168x** (CI 2.085-2.149; within 1% of board 2.1339x). ours/SGLang us: d1 66.8/170.2, 77.4/181.3;
  d16 129.1/264.4, 201.7/425.2; d32 154.4/259.5, 311.6/613.0; v1 114.7/270.5, 126.6/283.4; v16 266.0/530.8, 360.4/717.3;
  v32 332.4/701.4, 510.0/1053.2. v1 ratios 2.358/2.237 (last 4 checks sl 2.311-2.387, full 2.220-2.258): no clear v1 win ->
  NOT submitted (09:28Z rule (b) unmet). d16 full 201.7 = best ever (bal base, n5s8).
  H200 tuner T=8 (synthetic): sl default 111.5 | nu8d+nu8x ON 110.4 | nu8x on 110.6 | cslpw off 112.5 | npf 206/212/112 112.1-112.4 |
  g30n:o 110.2 (picked); full 137.0 | nu8d/x ON 135.4 (picked) | cslpw off 137.0 | npf 112 140.6. So on the H200 the nu8d/nu8x
  half loses (~+1.1/+1.6 us) while cslpw+npf 220 holds: H100 and H200 disagree on nu8d (not routing: hardware).
- v152 = v151 with _T8N = {cslpw, pr8_npf 220} only (nu8d/nu8x off -> TUNE[8] options again). validate test/scored/opt/fuzz ALL OK.
  Not checked alone (a tuner-option tie: lead 10:50Z); goes with the next real change.
- H100 d32 sliding timeline (v152, seed 99, 195.1 us): g30 qkv 0-14.9, nds 15.6-32.1, combine 16.6-34.2, g30 o 34.3-41.3, bnd 35.5-44.4,
  g30k pooler 41.2-69.3 || router_split 45.9-66.4 + routing 49.1-71.1, rmsnorm 71.4-74.0, tk2f 72.4-169.6 (97 us), norm2 157.6-174.0,
  g30 unpool 163.5-192.7, bnd 174.0-195.1. No idle gap; the only big non-expert item left is unpool's 23 us after tk2f (48 MB at
  H100 ~2.1 TB/s effective; earlier unpool-W L2 prefetch at tk2f end (UPF/UPF2/UPF3) all dead).
- Next session: T=1/8 options all decided; open items unchanged (o+bnd+pr1 megakernel ~0.5% d1; moe/proj Handoffs: proj G30_GX
  branch proj/g30gx when posted -> TUNE option at T=16/32 unpool/o, bundle into one check with v152).

## night5 s10 (2026-10-28 02:39Z-, okc-night5 session 10): v153 = v152 + proj G30_GX (branch proj/g30gx afc0c8b)
- Board at start: #42 at 2.1339x (#1 pjmullin3 2.399x, #10 2.254x). 10-28 budget: 15 checks / 10 submits.
- Cherry-picked proj/g30gx afc0c8b (based on v150) onto main v152: clean merge (g30_src.cu GX path, layer.py G30GX_CFG /
  G30KGX_CFG builds + TUNE options: T=16 {g30n:o + g30gx:o}; T=32 g30gx unpool/o, g30kgx pooler (NS 8) and combos). proj also
  moved the T=32 escape options (g30n:* off, g30k 0/120, pr32) into a comment: the H200 tuner never picked them.
- validate --quick: test 0/13, scored 0/12, opt 0/252, tk2f 0/300, replay 0/240, route 0/240 (gap 0.0256), fuzz 0/406;
  cold prepare **220.1 s FAIL** (proj's prep_time ran concurrently). Dropped pr32 builds (PR32_TOKENS = (); moe n5s10 asked:
  never picked on H200) -> cold **161.5 s** OK. Committed 3ef3268 (v153).
- **check 60f02748 (v153) 2.1373x** (CI 2.105-2.169; board 2.1339x). ours/SGLang us: d1 66.3/172.5, 75.0/183.0; d16 129.0/264.3,
  206.2/433.1; d32 152.8/261.7, 310.4/621.0; v1 108.8/259.6, 113.1/260.5; v16 265.6/517.4, 359.8/710.8; v32 333.8/714.7, 508.1/1071.1.
  v1 full 113.1 = best ever (was 126.6 in bb8ea548; T=8 full picked g14alt 3).
  H200 tuner GX: T=32 sl default 155.0 | g30kgx:pooler 154.7 (picked, confirm 154.9 vs 154.3) | g30gx unpool 155.2 / o 154.8 /
  both 155.5 | all three 155.2; T=32 full default 319.0 | g30kgx 318.0 | gx o/unp 318.8 (picked tk2h 3 at 317.9); T=16 sl/full
  g30n:o+g30gx:o 167.2/231.4 vs default 166.3/231.5 (tie). So GX in-layer on the H200 = g30kgx pooler -0.3..1.0 us, rest tie.
- **submit 3e3b9fd3 (v153, 3ef3268) 2.1580x = new board best** (CI 2.126-2.190; was 2.1339x 35dd53d0), #35 (#1 2.399x). us:
  d1 66.3/75.0, d16 129.9/205.9, d32 155.7/308.2, v1 110.6/122.6, v16 266.6/356.1, v32 335.7/506.4 (v16 SGLang drew slow: 541/725).
- v154 (51acdef) = v153 + T=8 sliding TUNE_BASE g30n:o (picked by the H200 tuner in both bb8ea548 and 60f02748); T=8 options
  {g30n:o True} (full) + {g30n:o False} (sliding escape). test 0/13, scored 0/12, opt 0/252. Not checked (tuner-pick pin: a tie).
- Next: nothing new queued from moe/proj. Open item = T=1 o+bnd+pr1 megakernel (bound ~0.5% d1). T=32 base could take
  g30kgx:pooler if the next H200 tuner picks it again (one pick so far, -0.6 us).
- Lead 03:25Z: judge prepare_s cap 220 s; log prepare_s per run. Judge prepare_s: 73c50eea 183.1, bb8ea548 202.5, 60f02748 183.5,
  3e3b9fd3 189.0 s (local cold ~0.85x of the judge's). Get them with `okc status <FULL 16-hex job id> --json` (short ids fail);
  tuner picks: python /tmp/picks.py-style regex on json.dumps (arrow is > in the raw JSON).
  H200 picks (bb8ea548 / 60f02748 / 3e3b9fd3): T=1 sl pr1_rb 32 x3 (= base), full g10alt:o 1 / pr1_rb 32 / g10alt:o 1;
  T=16 sl g30n:unpool x2, bal (= base); full g30n:unpool+qkv x2, g30n:qkv; T=32 sl g19:o, g30kgx x2; full g24sk2 combo, tk2h 3,
  g30kgx; T=8 sl g30n:o x3; full nu8d+x (= base since v152), g14alt 3 x2; T=128 default/lt:o/ltalt:pooler 2; T=256 lt:o x2 + lt all.
- v155 = v154 + TUNE lists cut to those picks (+ escapes {g30kgx:pooler False}, {g30n:o False}, {g14alt 1}); base pins T=32
  g30kgx:pooler (sl+full), T=8 full g14alt 3. Old lists: git show 51acdef:muse_decode_layer/layer.py. test/scored/opt 0/42/fuzz OK,
  cold prepare **139.4 s** (v153 161.5 s). Builds that now serve no option (g19 qkv/o at T=32, g24sk2, lt alts ...) not yet pruned.
- check cd6b3ba4 (v155) 2.1074x (CI 2.076-2.139), **judge prepare_s 176.7 s** (v153 189.0). us: d1 68.4/76.4, d16 123.1/205.9,
  d32 147.2/302.1 (d16/d32 sl best ever), v1 109.0/121.2, v16 257.3/353.6, v32 343.8/519.9. d1 sl +2 us / v32 +8..13 vs 3e3b9fd3 =
  run noise: the H200 tuner timed the same T=1 config (default 69.4 here vs 70.2 in 60f02748) and T=256 options tie within 0.5 us.
  Picks: mostly default (T=16 g30n unpool(+qkv), T=128 sl ltalt:pooler 2, T=256 sl lt all). Not submitted (< 2.1580x). main = v155.

## night5 s11 (2026-10-28 03:27Z-, okc-night5 session 11): v156 = v155 + cold-prepare cuts (split-compile, pool 16, dead g30 builds)
- Board at start: #35 at 2.1580x (3e3b9fd3); #1 pjmullin3 2.399x, #10 2.254x, #20 2.207x. 10-28 budget: 13 checks / 9 submits.
- Re-audit (no new lever found): H200 cd6b3ba4 profiles + H100 timelines (main v155, seed 99). Boundary after its producer is a
  stable ~2.8 us at T=32 (3 replays sl/full; a single 4.9 us sample was noise). H100 vs H200 tk2f per load is not comparable
  (local seed-99 routings differ: d16 10 experts here vs ~6 in the judge's). Expert scale bytes: entropy 3.71 bits (32 distinct
  E4M3 codes 95..126), FP4 codes 4.00 bits: the only compressible part is the scales (= moe's TK2F_S6, already dead on H200).
- Cold prepare anatomy (fresh caches, per-build start/end stamps via a timed copy of _PREBUILD): _moe__lib alone = 117.5 s
  serial (99.9 in a 2nd run) and was the critical path (122.5 s in the 8-thread pool); 13 other builds ~41 s each in 2-3 waves.
  * nvcc --split-compile=0 on _moe__lib: 117.5 -> 65-67 s alone. SASS: 115 kernels, same set; 63 identical up to shared-var
    offsets, others +-16..48 instrs (scheduling/shared address codegen). H100 ab_files (TUNE = {} both, A B B A x 4 rounds, all 12
    shapes): tie (A/B avg d1 85.9/85.9, d16 192.0/191.6 | 295.4/294.8, d32 189.1/188.3 | 415.8/415.2, v1 142.4/142.3 | 157.7/156.9,
    v16 302.6/304.1 | 451.2/449.1, v32 377.3/376.8 | 631.0/630.6). files_eq A vs B = the A vs A control (d16/d32 1-ulp nondeterminism).
    Also on _g10_lib (NVCC_SPLIT, layer.py).
  * dead builds dropped: g30n pooler T=32 (1,20,8) / T=16, all G30_GX g30n builds (G30GX_CFG = set(): no TUNE option since v155),
    g30k pooler G 120 and T=16. 16 -> 12 layer builds (+ 4 attention builds on the same pool).
  * _BUILD_POOL 8 -> 16 threads (19 builds incl. attention): builds done 122.9 -> 106 s (24 threads: 98 s; kept 16 for judge RAM).
  * cold prepare (prep_time.py): v155 142.2 s (builds 125.6 + tune 15.7) -> v156 131.5 s (builds 103.8 + tune 25.2: GPU shared).
- v156 validate --quick (prep skipped, measured above): ALL OK: test 13/13, scored 0/12, opt 0/42, tk2f 0/300, replay 0/240, route 0/240 (gap 0.0256), fuzz 0/60.
- check 0e4ba3c2 (v156, 50355f2) 2.1421x (CI 2.110-2.174), tests 13/13, **judge prepare_s 182.0 s** (v155 cd6b3ba4 176.7, v153 189.0;
  earlier 183.1/202.5/183.5): the local -11 s (builds -22 s) does NOT show on the judge, whose build host is likely core-limited
  (split-compile and 16 threads only help with spare cores). Neutral, kept (no build failed). us ours/SGLang: d1 67.0/170.5,
  75.7/180.5; d16 130.6/268.2, 205.8/432.9; d32 153.6/261.8, 292.5/616.0 (d32 full best ever); v1 118.9/278.2, 123.2/279.0;
  v16 261.5/526.5, 356.1/703.5; v32 338.7/737.1, 519.1/1098.2. Tuner picks: T=1 full g10alt:o 1, T=16 g30n:unpool (+qkv full),
  T=32 default, T=8 default, T=128 sl ltalt:pooler 2, T=256 sl lt:o. Not submitted (< 2.1580x; no per-shape change intended).
- Next: no layer-own lever left with a size (audits s45-n5s10 + this one); the gap to #1 is the experts (moe board lane:
  pjmullin3 moe 2.623x vs ours 2.247x) and attention. Port moe/proj Handoffs as they land; bundle into one check.

## night5 s12 (2026-10-28 04:25Z-05:15Z, okc-night5 session 12): v157 = judge prepare_s 182 -> 92 s (TSHIM + MOE_PRUNE)
- moe n5s17 request (loop64k.py over the layer's libs): no kernel the layer launches has an MMA loop across a 64 KB boundary.
  Crossings only in dead instantiations (plain GS16 tk2f <8,4,2,0,0,0/1,16,0,0>, moe_tw<2,4,8,1,8>).
- Kernel census (torch profiler over prepare() incl. every TUNE option + all TESTS/SCORED shapes, /tmp/layer_kn.py): 302 kernel
  names; the moe lib's 115 kernels -> 12 used (tk2f SA PF4, SQ PF4, SQ BAL, SQ tk2h; twm<8,1>, hot<8,1,128>, route/gather/norm*).
- MOE_PRUNE (layer.py `_moe_prune`): unused moe launchers -> TORCH_CHECK stubs (gemm13/gemm2/tb/tw/twf/twq/tk2fn/sk),
  moe_tk2f_launch cut to the used cfgs (+ cfg 100001 and moe_tk2 kept for tools/tk2f_stress.py). Used-kernel SASS identical.
  Alone it saved only ~50 of 1154 CPU-s: device code is NOT the compile cost.
- WHY: nvcc --time on a 1-kernel g30 lib: ptxas 0.09 s; cudafe++ 11 s + cicc 12 s + gcc 11 s on cuda.cu, main.cpp 12 s:
  ~50 CPU-s per lib is torch-header parsing (torch/extension.h). 17 torch-extension libs = ~750 of ~1150 CPU-s.
- TSHIM (layer.py `_TSHIM_H`, `_tdesc`, `_G30Raw`, `_LtRaw`, `_g30_load`): load_inline(no_implicit_headers=True,
  with_pytorch_error_handling=False); cuda.cu includes a stand-in (torch::Tensor = 9-int64 descriptor ptr/ndim/sizes[4]/numel/
  dtype/contig, TORCH_CHECK throws std::runtime_error, at::cuda::getCurrentCUDAStream = stream passed from Python), main.cpp
  pybind11 only; a `*_raw(std::vector<int64_t> descs, ..., stream)` entry wraps the unchanged launcher. Python adapters keep the
  call signatures. Applied to the 8 g30-family libs (names *_ts) + lt. 1-kernel lib 50 -> ~6 CPU-s. SASS of all g30 kernels identical.
  Host cost per call is higher (descriptor list) but only during capture: graph replay unaffected.
- Local cold prepare: v156 CPU 1154 s / wall 129 s -> v157 556 s / 65-67 s. H100 ab_files (TUNE = {}, A B B A x 3, 12 shapes): tie
  (all within ~1 us). validate --quick ALL OK.
- check af87d466 (v157, 83bc412) 2.1460x (CI 2.114-2.178), **judge prepare_s 92.2 s** (v156 182.0): the judge host is CPU-bound,
  total compile CPU is what counts. us: d1 66.9/74.7, d16 126.1/200.8, d32 156.3/303.2, v1 110.0/125.6, v16 265.2/352.7,
  v32 336.8/510.2 = noise vs board. Not submitted (< 2.1580x).
- Remaining torch-header libs (each ~45 CPU-s of header parse): g10 (one lib, many srcs), g24, g27 (use torch::empty inside),
  moe (30 launchers), attention csl/nds/nvs/nvscl. Not needed now (prepare_s 92 << 220 cap); convert if builds grow.

## night5 s13 (2026-10-28 05:22Z-05:50Z, okc-night5 session 13): no build; lead's v157 scored run logged; lane request
- Lead 05:20Z submitted v157 (job 35af50319e604b7b, r-layer 6b44d53 = check af87d466 code): **2.1420x** (CI 2.110-2.174),
  passed, judge prepare_s 100.6 s. Board best stays 3e3b9fd3 2.1580x (#36 now). Tuner picks there: T=16 g30n:unpool (+qkv full),
  T=32 full g24sk2 combo, T=256 lt all, rest default (ties within ~1 us). Don't resubmit v157.
- 10-28 budget after it: 11 checks / 8 submits. No Handoffs or r-layer branches newer than v157 (moe 21deb8a = moe board
  running the layer's bs256 config; moe HOT_NX exact-N probe is wrong for 105..128-token experts, probe only).
- Re-checked candidate levers against the record (all already measured/bounded, nothing built):
  * d1 full KV L2 prefetch during qkv (like KV_PF at sliding): moe s32-s35 H200 DIAG 'all-L2' only -1.2 us on 12.2; dead.
  * hot-expert L2 prefetch predicted from weights: s55/s90 (skew comes from data, bound <= 1 us).
  * router FP32 lossless compression: random FP32 mantissas, ~13% of 8 MB = ~0.25 us at T=1: not worth it.
  * 11-bit qkv/pooler: a 3-bit exponent window leaves ~1.6% of Gaussian weights (~130/row) as escapes: net ~5% bytes, heavy
    escape path; dead in this format.
  * T=1 o+bnd+pr1 megakernel: every last-block/ticket epilogue form lost +1.4..2.8 us (s8, s14, s74, moe n5s13 OB8).
- H200 af87d466 d32 sl (weakest shape, 1.687x): tk2f 91.7, g30 x3 43.6 + g30k pooler 21.2 (incl. PDL pre-wait time), nds 10.2,
  norm2 10.2, router_split 7.2 + routing 4.1 beside the pooler. GEMM excess vs 4 TB/s there is ~20-30 us but is proj's
  closed g19/g24/g30 consumer-bound line (and pr32 = router fused into the pooler lost on H200).
- 0 judge runs. Asked the lead for a lane (COORD Requests).

## night5 s14 (2026-10-28 05:28Z-05:55Z, okc-night5 session 14): no build; 0 judge runs
- No lead answer to the n5s13 lane request yet. Board: #36 at 2.1580x (#1 pjmullin3 2.399x, #10 2.261x). 10-28 budget: 11 checks / 8 submits.
- Option (a) of the request (d32 pooler||router phase) re-read against the record: expert-weight L2 prefetch beside the pooler
  = s55 DEAD (prefetch blocks starve router_split, routing ends 12-16 us later; tk2f gains only 10-25% of prefetched bytes);
  pooler grid / router SM split / pr32 / router3 at T=32 all measured (s9, s10, s60, n5s10). Nothing new to build there.
- New idea checked: start csl (d1 sliding attention) inside qkv g13's drain (q rows first in every qkv CTA, csl spins on
  per-head flags instead of griddepcontrol.wait). H100 timeline v157 d1 sl: qkv 0-13.5, csl 14.0-19.3, o g13 15.0-23.3 (PDL W
  prologue), bnd 22.1-27.5, pr1 23.0-45.9, rms 31.8-47.0, tk2f 46.4-81.1, n2u 77.6-85.9, bnd 81.7-87.7. Bound <= csl's H200 3.9 us
  minus ~1 us of current-token work = ~0.4-0.5% d1-weighted, but s62's CSL_PW (csl CTAs resident during g13) already lost
  +1.3..1.7 us at d1 to SM crowding + lost KV_PF warmth; flag-spin forms lost in s14 (BND_SPIN). Not built.
- moe n5s9 audit (H100, v144) already sums the d1/v1 chain sub-peak time (d1 22, v1 39 us) and bounds what a chain kernel can
  remove at ~1-3 us d1: no persistent megakernel item >= 1% on layer's side.
- Gap to #1 stays in the expert kernels (moe board #1 2.685x vs ours 2.25x; proj bs128 / moe bs256 lanes) and attention
  (#1 3.03x vs 2.60x; moe s40 bounds the four full kernels at ~-4% of the attention geomean).

## night5 s15 (2026-10-28 05:46Z-06:35Z, okc-night5 session 15): v158 = moe n5s23 routegathern port; 1 check
- Ported moe n5s23 (r-moe d3fc872 tools/n5s23/patch_layer_rg.py) into layer.py moe_routegathern: own 8 ids + x row loaded
  before the id scan, scan batched 8 ids/thread, repeat fixup by shfl. Applied cleanly. test 13/13 PASS; tools/files_eq.py
  v157 vs v158 'verify' 3 seeds: v16/v32 bitwise identical (v1 diffs = tuner picks, RG is T>32 only).
- H100 ab_files (v157, v158, v157, v158; 6 rounds, seed 99): v32 sl 379.6/378.9 -> 376.9/377.1, full 631.0/630.3 -> 628.7/629.8,
  v16 sl 303.7/303.5 -> 302.4/302.8, full tie. rg in the T=128 base (ab_flag, TUNE {}, seeds 99/7/13, 4 rounds): tie (-0.9..+0.2): not taken.
- H100 v32 sl timeline (v158): qkv 25.0, nvs 17.2, o 10.4, bnd 9.0, FR pooler+router 39.7, routing 3.7, routegathern 8.1 (overlaps
  routing under PDL), hot || twm 107.6-342.7 (twm 232.5), moe_norm ends 344.9 (2 us after twm), unpool 34.3, bnd 9.4 = 388.5 us.
- check 1d0568ec (v158 = 76f4632) **2.1100x** [2.078, 2.142], judge prepare_s 89.4. H200 routegathern 5.2 / 5.4 us (v32 sl/full;
  af87d466 had 7.4): the -2 us holds on the H200. Per shape: d1 65.9/75.3, d16 125.9/209.6, d32 149.1/301.6, v1 112.2/123.7,
  v16 269.1/357.7, v32 345.6/514.4 (d32 sl 1.702x: SGLang 253.8 that run). Whole-run noise; < 2.1364x (1% rule): not submitted.
- 10-28 budget after it: 10 checks / 8 submits.

## night5 s16 (2026-10-28 05:55Z-, okc-night5 session 16): v159 moe n5s24 norm port; v160 NORME_TRIG (T=128 unpool under twm's drain)
- v159 (92c12db) = v158 + moe n5s24 (r-moe 725f39a) in the layer's moe_src.py: moe_norm (ids + tw to smem before the
  NORM_EARLY spin / after the wait) and moe_norm2 (all 8 ids + tw in one batch), layer copy keeps __ldcg. test 13/13.
  files_eq (TUNE = {}): v16/v32 bitwise = v158; d16/d32/v1 differ, but v158 vs v158 differs the same way (atomics). H100
  ab_files verify (4 rounds): tie (-0.7..+0.8 us).
- FOUND: moe_norm (T > 32, NORM_EARLY path) never triggered its PDL dependents, so the T=128 g30 unpool (PDL secondary, W ring
  loaded before griddepcontrol.wait: G30_PDL) launched only after moe_norm exited: its W prologue never overlapped twm's ~20 us
  drain (which is why s100 measured G30_PDL ~0). v160 NORME_TRIG (layer.py text patch): launch_dependents at the start of
  moe_norm's rdy path. x is still read after g30's wait (= moe_norm complete): no ordering change. T=256's unpool is lt (no PDL
  prologue): unaffected; g30 at T=256 is dead (s100).
- H100 timeline v16 sl (TUNE = {}): g30 starts 249.8 us (during the drain, twm ends 278.8) vs 278.7 without; unpool ends 22.1 us
  after moe_norm vs 24.5. ab_files (TUNE = {}, A B A B), off -> on: seed 99 (6 rounds) v16 sl 305.8 -> 301.2 (-4.4), full 450.7
  -> 448.5 (-2.1); seed 7 sl -1.9, full -2.2; seed 13 sl -1.9, full -2.3; v32 / v1 tie on all seeds.
- check aba95f63 (v160 = 9fa5164) **2.1033x** [2.072, 2.135], judge prepare_s 89.6. us: d1 68.4/77.3, d16 128.8/207.5, d32 153.6/308.7,
  v1 117.2/126.9, v16 **260.7**/357.4, v32 330.5/505.4. v16 sl = best H200 number so far (v158 269.1, af87d466 265.2, board 266.6);
  v16 full tie; d1/v1 drew slow (paths untouched by v159/v160) = run noise. H200 profile v16 sl: g30 GPU time 43.9 us (1d0568ec 22.4):
  it now launches ~21 us earlier, under twm's drain, as on the H100. Not submitted (< 2.1364x, the 09:28Z 1% rule). 10-28: 9 checks / 8 submits.
- Next: v160 rides in every later check. Same pattern to look for: any PDL secondary with a pre-wait W prologue whose
  primary never triggers early (grep launch_dependents per chain). At T=256 the unpool is lt (no prologue); a PDL-capable
  m256 unpool would get the same ~20 us drain window (g30 MTOK 256 dead in s100: x ingest).

## night5 s17 (2026-10-28 06:11Z-, okc-night5 session 17): v161 RF_TRIG_ALL + CB_TRIG, v162 T=8 full g30n:o base; 1 check
- Audit (lead/STATE item 1b: PDL secondaries whose primary never triggers early). tools/timeline_all.py <file> all (new: all 12
  scored shapes in one process, STMT='TUNE={}'). H100 v160: the only serial PDL hand-offs left are (a) batched _red_fin -> g30 o
  at d16/d32 full (o starts exactly at red_fin's end: 140.3 / 261.2 us; attn_src gated RF_TRIG on batch == 1) and (b) _combine ->
  o at v1 full (_combine never triggered; base o is g12, not a PDL secondary). Everything else overlaps (T<=32 chains, twm drain
  since v160). T=128/256 qkv/o/FR pooler/unpool are cuBLAS (no PDL); the T=256 boundaries are at their FP32-stream byte floor
  (~32 MB, 9 us H100); H200 v32 qkv 20.9 / FR 32.6 us are ~compute-bound at m256 (9.7 / 19.2 GFLOP).
- attn_src: RF_TRIG_ALL (TRIG at every batch, still right after the wait) and CB_TRIG (_combine: launch_dependents right after
  gdc_wait, both _combine launch sites of verify/decode-gluon + the one in _decode). No ordering change: dependents read x after
  their own wait. H100 ab_flag (variants/n5s17.py = TUNE {}, 4 rounds, seeds 99/7), us:
  d16 full -0.6/-0.2, d32 full -0.6/-0.75; v1 full base 156.15/129.57, g30n:o untriggered 156.67/129.56, g30n:o + CB_TRIG
  156.17/128.79. d1 full noise.
- v161 (17b925a) validate --quick ALL OK, cold 67.8 s. check 8f5427dc **2.1169x** [2.085, 2.149], judge prepare_s 88.8: d1 67.9/72.9,
  d16 126.5/205.5, d32 154.0/304.7, v1 112.8/124.9, v16 262.6/**349.3**, v32 334.8/509.9. v16 sl 262.6 = 2nd H200 sample of v160's
  NORME_TRIG win (260.7; pre-v160 265-269); v16 full 349.3 best so far (357 before). H200 tuner T=8 full picked {g30n:o} (128.4
  -> 127.6, confirm 128.5 vs 127.5). Not submitted (< 2.1364x, 1% rule).
- v162: T=8 full base += g30n:o; TUNE[8] {g30n:o True} dropped (no-op now).

## night5 s18 (2026-10-28 06:26Z-, okc-night5 session 18): check of main v162
- check 3e39a051 (v162 = b18c9a6 submission.py, unchanged) **2.1143x** [2.083, 2.146], judge prepare_s 88.9. us (ours / SGLang):
  d1 66.9/170.7 76.7/181.2, d16 124.0/258.2 205.7/435.9, d32 **149.7**/254.4 **301.2**/606.9, v1 109.7/259.6 125.5/279.8,
  v16 264.7/508.4 352.3/679.7, v32 **327.8**/683.4 510.6/1062.7. In absolute us v162 beats the board run (v153 3e3b9fd3) on
  9/12 shapes (d16 sl -5.9, d32 sl -6.0, d32 full -7.0, v32 sl -7.9, v16 -1.9/-3.8); the ratio is lower because SGLang drew
  fast this run (d32 sl 254 vs 262 there, v32 sl 683 vs ~700). Not submitted (< 2.1580x; the 2 '1% rule' submits of 10-28 are
  used: 3e3b9fd3 + lead's 35af5031). Tuner picks: T=16 sl g30n:unpool, full g30n:unpool+qkv_gate, T=32 full tk2h 3, T=256 full
  lt:o, rest default. 10-28 budget after it: 7 checks / 8 submits.
- Re-checked (no build): T=256 _boundary has H32 off (h32 only at T <= ROUTER_FP32_MAX_TOKENS), ~24 MB/call vs 8.1 us H200 = near
  its byte floor; T=256 unpool (26 us H200, 17 GFLOP) and FR pooler (32.5 us, 18 GFLOP) are compute-bound cuBLAS (s100-s105: g30
  variants there dead). Layer-side lever list unchanged (STATE NEXT); experts (moe/proj) remain the gap.

## night5 s19 (2026-10-28 06:35Z-07:05Z, okc-night5 session 19): no build; 0 judge runs
- Board: #36 at 2.1580x (3e3b9fd3); #1 pjmullin3 2.399x, #10 2.269x, #20 2.221x. 10-28 budget: 7 checks / 8 submits (09:28Z-rule
  submits of 10-28 used). Board JSON has no per-shape times for other teams (only total value), so #1's per-shape split is unknown.
- Candidate checked for a port: proj's twm KQ K-split (r-proj 9029938, TWM_KQ_M). Its "-2.8 v16 / -2.2 v32 us" is vs NO split;
  vs the TWF_SPLIT 66 option the layer already tunes it is v16 176.6 vs 176.9 (o66) = -0.3 us (H200 e6416b6f): not worth a build.
- Re-read moe n5s15/n5s20/n5s21, proj s50 (BF16 wgmma m16/m32 dead), ~/attn NOTES 711/4787 (scored page tables 0% consecutive):
  no unmeasured lever >= 0.3% on the layer's side. Unchanged conclusion: the gap to #1 is in the expert kernels and attention.
- Reminder for 10-29 00:00Z: v162 beats the board run v153 in absolute us on 9/12 shapes (3e39a051); it can go in under the
  09:28Z rule only if a check of it lands within 1% of the board (>= 2.1364x); never resubmit unchanged code for luck.

## night5 s20 (2026-10-28 06:54Z-, okc-night5 session 20): H200 (Modal, tools/h200.py) real-routing A/B of the pinned expert knobs
- Method: /tmp/n5s20/{a,b,c} = main v162 submission.py + `TUNE = {}` + `_ab(T, drop, **kw)` (resets TUNE_BASE, applies kw at T,
  sl + full; copy in variants/n5s20_notune.py); `bash run.sh` = tools/ab_flag.py, 2 rounds, SEEDS 99/7/13, base first AND last
  (stmt-order bias check). Modal H200 numbers: compare within a run only, never vs the judge. Run 1 = 3.8 GPU-min.
- T=256 (v32), us [base | hot 24 | hot 0 | -twf_split | -rg | base]: sl s99 335.1|340.1|346.0|342.2|341.7|345.0, s7 313.8|335.9|
  322.5|323.2|318.4|316.5, s13 311.9|335.3|324.8|322.7|318.9|312.0; full s99 516.6|518.0|529.2|526.6|523.7|516.5, s7 490.3|519.5|
  499.9|501.8|498.7|491.1, s13 523.9|537.9|539.3|536.7|535.7|524.6. => n5s8's H100 pin {hot 32, rg, twf_split 66} confirmed on H200.
- T=128 (v16) [base | twf_split 66 | rg | twf66+rg | base]: sl s99 251.3|248.2|249.5|246.1|252.5, s7 258.5|259.6|255.9|258.5|260.3,
  s13 259.3|259.9|258.9|256.4|261.0; full s99 357.0|365.8|354.6|362.8|358.1, s7 354.8|361.7|351.4|354.7|355.1, s13 320.4|325.7|
  318.1|323.7|320.4. **rg alone: sl -2.4/-3.5/-1.3, full -2.9/-3.6/-2.3 us vs mean base (6/6 inputs)**; H100 n5s15 said tie.
  twf_split flips sign (as on H100).
- T=32 (d32) [base | tk2h 0 | tk2h 3 | bal | base]: tk2h 0 +1.7..2.9, tk2h 3 +0.5..1.5, bal +-0.3: base right.
  T=16 (d16) [base | bal off | tk2h 4 | base]: bal off +0.2..2.1, tk2h 4 +1..3: base right.
- T=8 sweep crashed: `bal` at T=8 = tk2f cfg 184204 not built (MOE_PRUNE): that option is gone; rerun without it.
- Run 2 (2.9 GPU-min; a first attempt was killed locally with exit 144 before output, ledger charged its 11-min reservation).
  T=128 seeds 5/1/4/99 [base | rg | rg+hot 32 | hot 32 | base]: sl 250.3|248.8|250.7|251.6|250.7, 246.3|246.1|248.4|248.5|247.8,
  260.9|260.0|262.6|263.5|261.4, 251.1|248.8|250.5|251.4|251.5; full 346.9|343.3|345.4|346.6|346.3, 344.9|342.5|343.9|345.0|345.3,
  336.9|334.4|336.0|336.9|337.5, 353.9|351.7|353.5|354.6|354.7. **rg: sl -1.7/-0.9/-1.1/-2.4, full -3.4/-2.6/-2.8/-2.6 vs mean base**
  (14/14 inputs over both runs); hot 32 loses.
  T=8 [base | cslpw off | nu8d+nu8x on | pr8_npf 212 | base]: cslpw off sl +2.7..2.8; others +0.1..+1.0: base right.
  T=1 [base | kv_pf off | g13o off | base]: +0.2..+1.1: base right.
- v163 = v162 + rg in TUNE_BASE[(128, *)] (est. v16 sl -1.9, full -2.9 us = ~+0.12% total).
- Lesson: tools/h200.py real-routing ab_flag is ~3-4 GPU-min for 3 shapes x 3-4 seeds x 5 stmts; its base-first/base-last spread
  is <= ~1 us except T=256 sl s99 (10 us drift): always repeat the base. Shared ledger hit 40/60 by 07:45Z (all lanes).
- check c9f263d6 (v163 = d02f7a6) **2.1476x** [2.115, 2.180], judge prepare_s 88.6. us ours/SGLang: d1 68.2/171.8 76.5/181.9,
  d16 122.3/252.5 206.2/431.0, d32 154.8/261.4 **283.2**/621.4, v1 112.1/264.5 119.3/269.7, v16 271.5/552.0 362.9/713.7,
  v32 341.1/741.2 507.1/1076.8. d32 full 283.2 = best ever (prev 301); v16 sl 271.5 slow = node (the tuner's synthetic T=128 sl
  default 268.8 here vs 257.3 in 8f5427dc, 273.0 in 3e39a051: +-8 us per node, hides rg's ~2 us). Not submitted (< 2.1580x;
  09:28Z-rule submits of 10-28 used). 10-28 after it: 6 checks / 8 submits.
- Next: the H200 pre-screen makes <1% per-shape changes measurable (noise <= ~1 us within a run): candidates the H100 called ties
  (NOTES "tie"/"noise" items: g30 T=16 options, BND_SPLIT, pr1 npf, kv_pf...) can be re-decided there in batches, 3-4 GPU-min each;
  the shared cap is 60 GPU-min/day across lanes.

## night5 s21 (2026-10-28 07:13Z-08:20Z, okc-night5 session 21): H200 (Modal) ranking of the T=128 g30 GEMM forms; no build; 0 judge runs
- Question left open by s99/s101/s102 + moe n4s43 ("only an H200 run can rank g30 forms": maybe R2 = 128 W rows/CTA, or the
  split forms, win on the H200 where the W stream is 1.45x faster). Answered with tools/h200.py --cmd (8.8 GPU-min total;
  runners dev/n5s21_h200_s99.sh / _s101.sh; overlay = submission.py + the dev/s99 / dev/s101 prototypes in a non-dev dir,
  since h200.py skips dev/). Standalone cold-L2 graph us, T=128, Modal H200 (compare within the run only):
  | shape | g30 form | g30 | cuBLAS (same run) |
  | unpool 8192x4096 | R1 S1 (main's form, no escapes) | 21.7 | 20.5-20.6 |
  | unpool | R1 S2 FP32 slabs | 24.9-25.2 | 20.6 |
  | unpool | R2 (NS10 NW5) S1 / S2 | 34.9 / 21.9 | 20.5-20.7 |
  | pooler 4096x8192 | R1 S2 FP32 slabs (no escapes, no reduce) | 22.1-22.3 | 26.1-26.9 |
  | pooler | R1 S4 / S3 slabs | 24.8-25.0 / 30.1 | 26.5-27.3 |
  | pooler | R2 S2 / S4 | 35.5 / 22.4 | 26.6-26.7 |
  | pooler | s101 split real weights (escapes + ticket reduce, BF16 out) S2 / S3 | 26.4 / 36.1 | 28.1 |
  | qkv 2304x8192 | s101 split real S1 / S2 / S3 | 37.2 / 25.6 / 21.0-21.2 | 17.3-17.5 |
  | qkv | s101 uneven slabs (no escapes/reduce) S3 / S2 | 16.2 / 21.3 | 17.3-17.8 |
  H100 same day for reference: unpool R1 S1 22.2 vs 26.6; pool S2 25.6 vs 27.6; R2 S1 34.6-35.4, S2 25.8-26.1, pool R2 S4 26.9-27.6;
  qkv real S3 21.8 vs 20.6, slabs S3 16.7 vs 20.2.
- Read: (1) R2 buys nothing on the H200 either (= R1 at best): CLOSED. (2) The H200 favours cuBLAS on the unpool (g30 +1.1 us
  standalone; main keeps g30:unpool because in-layer its W prologue runs under twm's drain, H100 real-input A/B n5s8 +6..8 us
  when off, never H200-A/B'd on real inputs: candidate for a Modal run when minutes allow). (3) qkv split-K: the reduce kills it
  on the H200 too (+3.7 us); the slab form is -1.6 us before any consumer-side slab sum (QSS-style, s72: +1.6 in nvs): CLOSED.
  (4) Only real H200 lever: the pooler slab form -4.6 us vs cuBLAS pooler-only, but the layer's T=128 FR GEMM carries the router
  (FR = pooler + 3.1 us on the H100, dev/n5s21_rt.py: FR torch.mm 30.7, pooler-only 27.6) and the stand-alone T=128 router is
  slow: pr3 router + routing alone 15.6 us H100 (cuBLAS hi/lo 7.6 + reduce 1.9 in s102). So it pays only with the router's
  512 hi/lo rows INSIDE the slab kernel (hi = 12-bit ok, lo = raw BF16 rows: needs a 2nd decode path; 4.2 MB lo + 3.1 MB hi)
  and the work split kept at <= 132 CTAs (s105: 136 CTAs = 2 waves). Paper bound: ~22.8 (incl. escapes) + ~3.3 (router bytes)
  = ~26 vs FR ~29.5 on Modal = ~-3.4 us per v16 layer (sl + full) = ~+0.19% total, plus rg summing 2 slabs. Below the lead's
  1% bar; 2-3 sessions. Queued in STATE, not started.
- H100 real-input ab_flag (variants/n5s21_notune.py = v163 + TUNE {} + _ab; seeds 99/7/13, 2 rounds) T=128 fr False vs mean of
  base first/last: sl +1.0/-0.5/-0.8, full -0.5/-1.3/-0.3 us: tie (n5s8's "full -3..-6" was before rg). Not sent to the H200.
- Shared Modal ledger: 52.2 of 60 GPU-min used on 10-28 after this session (mine today: n5s20 ~18 incl. a lost reservation, n5s21 8.8).

## night5 s22 (2026-10-28 07:38Z-08:00Z, okc-night5 session 22): H200 (Modal) real-input A/B at T=128 of g30:unpool and fr; no build; 0 judge runs
- STATE NEXT 4. /tmp/n5s22/a = variants/n5s21_notune.py (= main v163 submission.py + TUNE {} + _ab); runner copied to
  dev/n5s22_h200_run.sh (`python tools/h200.py muse_decode_layer a=<dir> --cmd 'bash run.sh'`), ab_flag "verify 16", 2 rounds,
  seeds 99/7/13. 3.4 GPU-min. Modal H200 us [0 base | 1 g30:unpool off | 2 fr off | 3 both off | 4 base]:
  sl s99 249.4|253.3|251.7|254.4|249.9, s7 255.3|258.8|257.0|259.9|256.3, s13 258.4|261.3|259.8|262.7|258.2;
  full s99 354.0|356.9|355.8|357.5|353.8, s7 350.7|353.3|351.7|353.7|350.6, s13 316.5|318.7|318.2|320.1|316.1.
- Read: g30:unpool off loses +2.2..3.6 us on 6/6 inputs (so the H200's standalone cuBLAS-unpool edge of -1.1 us, n5s21, is
  more than paid back by the in-layer PDL prologue under twm's drain); fr off loses +0.9..2.2 on 6/6 (the n5s21 H100 'tie' was
  H100-only). Both off adds up. T=128 TUNE_BASE {fr, g30:unpool, rg} confirmed on the H200: CLOSED, main unchanged.
- proj claimed NEXT 3 (pr128: T=128 g30 pooler+router hi/lo one launch) at 07:50Z in COORD: layer does not start it.
- Shared Modal ledger 55.8 / 60 GPU-min on 10-28 after this run.

## night5 s23 (2026-10-28 07:44Z - ~08:40Z): v163 scored; in-layer PREFIX ORACLE (H100); no layer-lane item >= 1%
- submit 7693598b (v163, d02f7a6, per lead 07:35Z): **2.1311x**, prepare_s 89.8; d1 68.3/75.5, d16 127.9/206.7, d32 154.9/295.3,
  v1 110.4/122.9, v16 267.3/350.7, v32 327.4/503.0. Board best stays 3e3b9fd3 2.1580x (#37). v163 needs no more scored runs.
- Modal ledger 55.8/60 GPU-min used for 10-28 before this session: no H200 oracle possible today (one run reserves >= 11 min).
- tools/prefix_oracle.py <submission.py> <dir>: TUNE = {} variants that return dummy outputs right after qkv (q), attention (a),
  o + first boundary (o), pooler/router/rmsnorm + routing join (p), experts (e: _tail returns before unpool + 2nd boundary;
  at T=1/8 n2u/nu8 carry the unpool, so only the boundary goes). Segment = difference of adjacent prefixes (in-layer, incl.
  launch gaps and lost overlap). H100 tools/ab_files.py 3 rounds, all 12 shapes, medians (us), sliding (full in parentheses):
  | load | qkv | attn | o+bnd | pool/rt/norm | experts | unpool+bnd | total | expert hits (slots, max/expert) |
  | d1   | 14.1 | 4.3 (12.7) | 6.6 | 20.1 | 39.0 (n2u incl.) | 1.4 | 85.8 (105.6) | 3 (3, 1) |
  | d16  | 17.0 | 8.2 (125) | 8.3 | 25.8 | 113.3 | 18.4 | 190.6 (292.8) | 11 (25, 6) |
  | d32  | 16.6 | 12.9 (242) | 10.8 | 27.9 | 99.7 | 19.1 | 188.1 (412.7) | 9 (50, 16) |
  | v1   | 15.0 | 4.9 (16.5) | 8.1 | 24.0 | 86.1 (nu8 incl.) | 2.8 | 140.6 (155.9) | 8 (10, 2) |
  | v16  | 18.8 | 12.5 (133) | 13.9 | 37.0 | 191.0 | 26.0 | 298.9 (444.5) | 17 (162, 45) |
  | v32  | 20.2 | 15.1 (253) | 19.3 | 42.5 | 237.0 | 42.2 | 374.9 (627.9) | 22 (359, 92) |
  Floors at the H100's ~3.0 TB/s stream rate (moe n5s29): qkv 28.3 MB 12-bit 9.4 us (BF16 cuBLAS at T>=128: 12.6); o 16.8 MB
  5.6; pooler 50 MB 12-bit + router 8.4 MB 19.5; expert 26.7 MB each (NVFP4 + scales) -> d1 27, v1 71, d16 98, d32 80, v16 151,
  v32 196 us.
  => Experts are 45-65% of every sliding layer and their excess over the stream floor is d1 ~12, v1 ~15, d16 ~15, d32 ~20,
  v16 ~40, v32 ~41 us H100 (~0.7x on the H200): the ONLY >= 1% items (d1 ~1.5%, v1 ~1.2%, v16/v32 ~1.1% each of total score).
  moe NOTES n5s20/n5s29 decompose the decode part (fixed 14.1 = routing 1.5 + first fill 2.1 + tail spread 2.3 + norm2;
  4.4/extra group) with every piece dead; big-M is proj (bs128) / moe (bs256).
  Layer-lane segments: pool/rt/norm at d1 = floor (20.1 vs 19.5); o+bnd <= floor + 1..3 us; attention sliding closed (4-15 us,
  latency-bound, HBM idle); qkv as first op 14.1 vs 9.4 at T=1 (~0.5% d1 on H200, proj s17: g13 at bare TMA-stream speed),
  T=128/256 qkv 18.8/20.2 vs 12-bit 9.4 (g30 qkv forms dead n5s21, ~0.3% if it hit the floor); unpool+bnd T>=16 = proj n5s14's
  co-bound +1.0% ceiling for pooler+unpool together. No single layer-lane item reaches 1% of total score.
- 0 judge checks, 1 submit (v163), 0 Modal min. Lane request posted in COORD.

## night5 s24 (2026-10-28 07:52Z - 08:15Z): no build, 0 judge runs, 0 Modal min (lane request n5s23 still unanswered)
- H100 timelines of main v163 (tools/timeline.py, seed 99): v1 sliding 144.1 us = g14 qkv 14.5 | vtk 7.5 | g30 o 13.1 | bnd | pr8 30.8
  | rmsnorm | tk2f 84.0 (51.4-135.4; 8 experts = 214 MB at 2.54 TB/s) | nu8 ends +5.2 | bnd ends +3.5. d32 sliding 193.5 us = g30 qkv
  15.3 | nds+combine 18 | o 6.8 | bnd | g30k pooler 28.0 || router_split 19.6 + routing | tk2f 98.0 | norm2 +3.6 | g30 unpool ends +18.6
  (48 MB 12-bit = its H100 stream floor) | bnd +2.9. Every non-expert piece re-matched to a dead/bounded NOTES entry (unpool/pooler
  W prefetch during attention: s61/nds-start PF zero-sum; QG/gate split s54; N2U_BOUNDARY s14; loop64k audit done, line ~2424).
- New board seen: `muse_prefill_layer` (Prefill layer, T=753x4 / 16K chunks, SGLang 387.4 ms, #1 2.025x, 555 rows; we have no
  row; 20 checks + 10 scored/day unused). The overall board still ranks exactly by muse_decode_layer speedup (top 45 checked,
  /api/v1/leaderboard 'solved' max 4), so prefill does not count overall today. Flagged to lead under Requests.

## night5 s25 (2026-10-28 08:11Z-08:45Z): H200 judge DIAG of moe n5s32's tk2f oracles -> rows' structure is 7-37 us/layer on the H200
- Built (c3a234a, OFF on main): TK2F_ORX DIAG flag (moe_src python `TK2F_ORX`; `_moe__lib` turns on `#define TK2F_ORX 1`). tk2f_go
  re-dispatches, generic over every built PF config: g_orx (= cfg / 100000000, stripped in moe_tk2f_launch, also in the pruned copy)
  1 orz = PF | 4096 (a1 = a0, b1 = b0: no rows), 2 orn = PF | 2048 (tk2f_produce ORN: plain arrive on full, no copies), 3 ornc =
  PF | 2048 | 256 (+ no consumer math). TUNE option tk2x 7/8/9 ("diag": timed, never picked) at T = 1 (layer path), 8 (nu8 path), 16
  and 32 (moe kernel_fn path); T=32 also with tk2h 0 (tk2h's consumer ignores PF 256). DIAG copy: `sed 's/^_moe_TK2F_ORX = False/
  _moe_TK2F_ORX = True/' submission.py > variants/diag_orx.py` (md5 38c5890a). Main bench test 13/13 PASS.
- H100 DIAG bench tuner (synthetic skewed routing), sliding base | orz | orn | ornc: T1 86.7|62.6|75.0|68.3, T8 143.3|72.9|111.7|88.9,
  T16 219.1|97.3|168.4|119.8, T32 tk2h0 215.8|103.2|176.8|134.8.
- CHECK 0dc9b93f (DIAG copy of c3a234a, NEVER submit): 2.1286x, prepare_s 83.2, all picks default. H200 tuner whole-layer us:
  | T / layer | base | orz | orn | ornc | rows (b-orz) | STRUCTURE (ornc-orz) | math, instant ring (orn-ornc) | bytes not hidden (b-orn) |
  | 1 sl      | 77.5 | 52.3 | 71.2 | 59.6 | 25.2 |  7.3 | 11.6 |  6.3 |
  | 1 full    | 74.7 | 64.1 | 71.6 | 68.0 | 10.6 |  3.9 |  3.6 |  3.1 |
  | 8 sl      |103.3 | 65.0 | 93.3 | 76.8 | 38.3 | 11.8 | 16.5 | 10.0 |
  | 8 full    |121.1 | 77.2 |111.8 | 91.2 | 43.9 | 14.0 | 20.6 |  9.3 |
  | 16 sl     |166.4 | 80.5 |152.4 |102.6 | 85.9 | 22.1 | 49.8 | 14.0 |
  | 16 full   |238.4 |163.6 |226.1 |189.9 | 74.8 | 26.3 | 36.2 | 12.3 |
  | 32 sl h0  |184.5 | 87.7 |173.0 |124.8 | 96.8 | 37.1 | 48.2 | 11.5 |  (default tk2h 4: 179.8 | orz 89.3 | orn 169.7)
  | 32 full h0|337.6 |245.0 |324.1 |279.5 | 92.6 | 34.5 | 44.6 | 13.5 |
  READING: on the H200 the tk2f rows are CONSUMER-bound: with a zero-byte zero-latency ring the layer is only 3-14 us faster (b-orn),
  while structure + math with no bytes at all take 15-85 us. s81's "tk2f at the stream rate" (ref - nm = 1.5-4.3 us) only showed the
  math alone hides; structure + math together do not. Structure free (additive, x the per-load %/us table) = d1 1.3+0.2, v1 1.3+0.6,
  d16 1.8+0.8, d32 2.2+0.7 = ~8.8% score upper bound (synthetic routing; real is < that: the stream floor is under it, unknown since
  no stream-only oracle exists). moe n5s32's build gate (>= 5 us d1, >= 10 d16/d32/v1 on the H200) is met on every shape. Saves
  moe's queued 15 Modal GPU-min. Next: moe builds the overlapped-row tk2f (its lane); layer integrates + gives it DIAG checks.
- CORRECTION (08:50Z, after moe n5s33): the READING above overstates. ornc-orz is the consumer chain when nothing waits on data, not
  time added to the stream (moe's real BRA = tie on all 12 H100 shapes). What the H200 table does show: a free stream (orn) saves
  only 3-14 us, so on the H200 the consumer chain and the stream are BALANCED (each within ~3-14 us of the rows' time). A faster
  consumer alone gains at most base - stream-only, and moe n5s15/n5s29 put tk2f's ring stream within 0.5-2.5% of the H200 HBM
  ceiling. => no reachable >= 1% from the structure; the '~8.8%' is withdrawn. Value of the probes: orn (b-orn = 3-14 us) bounds
  any stream-side lever (faster loads, deeper ring) at those numbers IF the consumer kept up; both sides would have to shrink together.

## night5 s26 (2026-10-28 09:06Z - 10:20Z): H100 in-layer 'free piece' oracles (lead 08:15Z), 0 judge runs, 0 Modal min
- tools/free_oracle.py <submission.py> <dir> writes base0 (TUNE = {}) + free_{a,b1,r,b2}: piece X runs once in the eager warm-up
  call, its outputs are cloned and kept per (tokens, sliding), later calls (incl. the captured graph) launch nothing for X and
  hand the kept outputs on (same routing downstream). tools/free_check.py: free_a/b1/r outputs = base (0 or base's own run-to-run
  noise) on all 12 shapes; free_b2 differs (as meant). a = attn_kernel_fn, b1 = first _boundary (absent where fused: T=1 pr1bnd,
  T=8 ob8), r = _route (T=16/32/128 side stream) / _routing (T=256 FR) (T=1/8 pr1/pr8 fused: n/a), b2 = _tail's _boundary
  (T=8 incl. the nu8 scatter-combine).
- H100 ab_files 3 rounds, main v163 (5f7841b), seed 99, us saved when free = base - free_X, sliding / full:
  | load | base sl/full  | attention   | b1        | router+routing | b2        | glue b1+r+b2 sl x %/us |
  | d1   |  85.4/105.7   |  2.4/ 13.0  | fused     | fused (pr1)    | 1.4/1.5   | 1.4 -> 0.25% |
  | d16  | 191.0/294.5   |  7.3/124.0  | 1.8/3.0   | 2.6/3.1        | 2.8/3.0   | 7.2 -> 0.60% |
  | d32  | 189.0/413.8   | 14.1/242.5  | 0.9/2.2   | -0.3/-0.2      | 2.0/2.6   | 2.9 -> 0.17% |
  | v1   | 141.7/156.9   |  3.3/ 18.4  | 1.6/2.1   | fused (pr8)    | 3.7/3.3   | 5.3 -> 0.59% |
  | v16  | 299.4/444.2   | 11.4/132.7  | 7.6/5.3   | 1.7/3.3        | 4.1/2.9   | 13.4 -> 0.55% |
  | v32  | 378.0/628.1   | 15.8/253.2  | 14.9/10.2 | 5.6/3.4        | 11.8/9.9  | 32.3 -> 1.07% |
  (%/us = the H200 sliding weights from STATE, applied to H100 us: an upper bound.) All glue free = ~3.2% sliding + ~1.1% full;
  sliding attention free = ~3.2% (closed: s64 at the gather floor, latency-bound); full attention = gather floor (s84).
- Reachable part: the T=128/256 boundaries move 12.6 / 25.2 MB (a + s read, s' + h written) = 4.2 / 8.4 us at 3.0 TB/s, so of
  v32's b1+b2 26.7 us ~17 is bytes. Standalone _boundary (cold L2): T=128 6.8 us (1.85 TB/s), T=256 11.0 (2.29 TB/s); SPL 2/4 and
  4/8 warps worse; 32 warps 6.4 / 10.2 us (-0.8). BUILT BND_W32_MIN (32 warps at T >= 128, both boundaries): in-layer ab_files
  5 rounds ABAB: v16 sl -0.6/+0.2, full -0.3/-0.4, v32 sl -0.4/-0.3, full +0.4/0.0 = TIE (in-layer the boundary sits behind cuBLAS
  drains/launch, not its own fixed cost). Reverted, DEAD. Remaining excess (v32 ~10 us sl, v16 ~5) is launch/serialization
  between cuBLAS GEMMs (no PDL in nvjet) and the boundary: removable only by fusing the boundary into the GEMMs (row norm over
  8192 cols needs a cross-CTA reduce; folding the gain into W changes h's BF16 rounding vs SGLang = routing risk). ~0.3-0.5% at best.
=> No layer-lane piece has a reachable >= 1%: biggest free numbers are v32 glue (32 us sl, ~2/3 bytes) and sliding attention (closed).

## night5 s27 (2026-10-28 09:12Z - 09:35Z): no build, 0 judge runs, 0 Modal min (lane request n5s26 unanswered)
- H100 timeline main v163 verify 32 sliding (tools/timeline.py submission.py "verify 32 sliding"; note the filter is the
  full load name, "v32 sliding" matches nothing): total 385 us = qkv nvjet 22.0 | nvs attn 16.3 | o nvjet 10.3 | bnd 9.1 |
  FR nvjet 39.7 | _routing 3.3 + routegathern 7.1 | moe_hot 105.4-281.3 beside moe_twm 107.9-340.2 (experts 235 us = 61%) |
  moe_norm 316.9-341.9 (1.7 us past twm) | unpool nvjet 33.2 | bnd 9.2.
- H200 (scored run 7693598b) v32 sl: twm 205.0 vs moe n5s25 bytes floor ~156 us (624 MB at 4.0 TB/s); d1 sl tk2f 20.4,
  pr1 17.0, qkv g13 10.6, bnd 2x 6.2, o g13 6.1, n2u 5.2, csl 3.9. The weighted gap is in the expert kernels (twm drain ~20 us
  per T=128/256 call = proj TWS, claimed; consumer excess ~21 us = moe closed), not in the layer's glue (n5s26 oracles).
- Re-checked levers already on record: NVFP4 code/scale compression (codes uniform 4.0 bits; scales 3.71 bits, S6 H200-dead),
  unpool W prefetch during the twm drain (s6 dead), row-subset unpool during the drain (W-bound: doubles W reads).
  Choosing routes among near-ties to cut distinct experts would use the checker's slack, not a real speedup: not done.
- Nothing built; no check spent (10-28 left: 5 checks / 7 submits). main unchanged (v163).

## night5 s28 (2026-10-28 09:31Z - 09:55Z): no build, 0 judge runs, 0 Modal min; lane-change request (lead 09:20Z)
- Lead 09:20Z: take the v1/d1 sliding attention piece + glue, builds that remove bytes or launches only; else ask for a lane.
- Checked the only item never attempted (s86: "make pr1/pr8 light enough to co-reside with o"): d1 pr1 = 58 MB in 17.5 us on the H200
  and d1 qkv g13 = 28 MB in 10.8 us both sit ON moe n5s29's H200 bulk-stream fit (t = 3.8 us + bytes / 4.26 TB/s -> 17.4 / 10.4 us).
  So the only thing left there is one ~3.8 us ramp hidden under o's tail, and s86 already measured the H100 analogue (o end -> pr1 end
  21.4 separate vs 22.5 fused; OB8 at T=8 +1.8 us): bound <= ~1 us d1 (~0.2%). Not built.
- Byte-removal idea at T=256 checked on paper: b1 writes only h (+ a per-row rsqrt), b2 recomputes s' = alpha1 s + beta1 r a from s and
  the o output instead of reading s': 176 vs 192 KB/row = -4 MB per v32 call = ~1.3 us (0.04%), plus a bit-identity risk on s'. Not built.
- Sliding attention d1 2.4 / v1 3.3 us free (n5s26) = latency-bound csl/vtk; every overlap form is dead (s62 CSL_PW, n5s14 flag-spin).
- => layer lane has no build >= 0.3%. Lane-change request posted in COORD (Requests). main unchanged (v163); 10-28 left: 5 checks / 7 submits.

## night5 s29 (2026-10-28 09:48Z - ~11:00Z): tk3 lane (lead 09:55Z) -> built "PR" (pair-role tk2f) on branch layer/tk3 a2ed836; H100 ~tie/loss; 1 judge DIAG check; 0 Modal
- (1) Consumer microbench (dev/n5s29/mb_tk3.py = moe tools/n5s18/mb_warm.py + modes; per stage 8 KB W + 1 KB scales + N x 64 x
  tile; 2 WGs x 128 rows x k64; tk2f's dq). `WMB=24 NL=16 MM=01,41,51,61,81,71 python mb_tk3.py` (WMB 24 = L2 = consumer-bound).
  H100 G132 warm, cycles/stage: ring 308-427 | wgmma moe form (wait<1> per k16) N8 591 / N16 632 / N32 690 | wgmma stage-pipelined
  N8 542 / N16 611 / N32 672 | mma.sync 1 group x-regs 553, 2 groups x-regs 597, 1 group x-smem 532, 2 groups x-smem 671.
  => wgmma is dq-bound like mma.sync (not wait-latency-bound); N16 wgmma only 9% under an mma.sync pair with x from smem; any
  16-token shared-dequant body should cost ~1.2-1.3x one 8-token row in isolation (moe's in-kernel forms measured ~2x).
- (2) Built PR (tk2f PF 16384, layer option "tk2pr" = W + 1 at T=16/32, pair row weight (8+W)/8; TK2PR build flag, OFF on
  main = not compiled): routing splits each expert into a pair space (ng/2 full 16-token pairs) and a single space (the odd
  8-token group); k = rint(G w np / (ns + w np)) CTAs [G-k, G) run ONLY pairs, the rest ONLY singles with tk2f's unchanged
  1-group body (s_sb group base); both roles publish into cnt, gemm2 waits for the sum. Producer = tk2f_produce on SQ stages.
  Correct: dev/n5s29/pr_test.py (24 scored d16/d32 inputs x 6 seeds + bench TESTS = 37/37; d32 bitwise = tk2h).
  BUG found on the way: moe kernel_fn's norm2/norm3 pick decodes PF4 from cfg digits ((c5-80000)//1000 & 4): my 19W204 code
  only got moe_norm3 for W 2..5 (fixed: explicit PR clause). Timings below with W outside 2..5 before the fix were wrong-output.
- (3) Pair body iterations (H100 ab_flag, real routings, d16 sliding s7 / s13; base BAL 154.8/155.7, non-BAL tk2f 157.9/158.6):
  a. tk2g_row TWO as is (x group 0 regs, group 1 smem; n<=8 body merged in: 192-mma 35 KB loop): +15..30 us.
  b. TWO only (128 mma, 20 KB): 166-170 best W. Timeline (dev/n5s29/pr_tl.py, TK2PR_TL globaltimer stamps): pair row 5.3 us
     vs single 2.4 us (DRAM-paced) -> 2.2x; ncu source counters: pair loops stall_no_inst 36-52% even when moved fully past
     0x10000 with a dead pad (TK2PR_PAD) or put in a __noinline__ function (no -rdc: callee is emitted inside the kernel
     section, same offsets). RULE (extends moe n5s17): a HOT loop at kernel code offset >= 64 KB is slow (no_inst), not only
     one straddling it; the compiler always emits single g13, single g2, pair g13, pair g2 (source order / __builtin_expect
     ignored) and the singles' unrolled loops are 20 KB each.
  c. Compact body tk2p_consume (TK2PR_C): ONE run-time-gated body for both phases, both x groups in lane-private smem
     (2 x 64 KB after the ring; dyn smem 200 KB), s_red single-buffered (+1 bar.sync/row): rolled 4.4 KB loop -> latency-bound
     (stall wait 22% / short_sb 15%, ~2000 cyc per 18 KB stage vs tk2f single ~680): 159.7/159.7 at W 5.
  d. + PR skips the dead D>0 tk2w_bound block (-9 KB of code before the loops) + `#pragma unroll 2` (loop 0xdaf0-0xfce0, all
     < 64 KB, REG 166, no stack): d16 s7/s13 158.7/159.2 (W 4) = non-BAL tie, +4 vs BAL base; s99 (no pairs) +3.2 fixed;
     d32 (base tk2h W3=4): s99 186.5 vs 185.9, s7 195.1 vs 195.3, s13 191.3 vs 185.9 (W 4).
  Fixed overhead (s99, k = 0, identical instructions 35.19M): ncu +3.8 us = DRAM 82 -> 79%; ~1.1 us of it is the L1 carveout
  (plain tk2f with +128 KB dyn smem: +1.06 us, variants/sub_nbpad.py); gemm2 long_sb +5% (inter reloads with a 30 KB L1).
- Cold prepare with TK2PR on: 69.2 s (H100). H100 tuner: T=32 sl picked tk2pr 5 (213.4 vs 214.0, tie), T=16 +5 (synthetic
  routing has no pair at T=16).
- (4) Judge DIAG check (see below): variants/chk_pr.py = main build + TK2PR True + TUNE[16]/[32] += {tk2pr 5}, {tk2pr 7}.
  check df997d8f (chk_pr.py sha1 4a05fbc6, = main v163 build + TK2PR + tk2pr tuner options; branch layer/tk3 a2ed836):
  **2.158x**, prepare_s 91.9. Per shape us: d1 65.9/73.8, d16 125.1/204.2, d32 151.2/289.8, v1 107.3/121.3, v16 264.8/351.3,
  v32 335.8/515.0. H200 tuner (synthetic routing; T=16 has no pair expert, T=32 has 2): T=16 sl default 155.1 | tk2pr 5 158.1 |
  7 157.9; full 213.4 | 214.8 | 214.1; T=32 sl 155.2 | 155.3 | 156.7; full 329.1 | 330.8 | 330.1. PR never picked -> the run
  executed main v163's configuration = a redraw of v163 (NOT submitted: no code path of ours changed).
  => H200: PR's fixed overhead ~+3 us (= H100 +3.2), and at T=32 the pair gain ~= that overhead (tie). Next lever: remove the
  fixed overhead (L1 carveout ~1 us: dyn smem 200 KB -> e.g. keep x of ONE group in smem by splitting the pair row's K in
  halves, or alias the x region into the ring; the other ~2 us: ncu gemm2 long_sb / waits) -> T=32 ~-3 us sl, then real-routing
  Modal A/B (after 00:00Z; Modal 55.8/60 used 10-28) at d16/d32 x seeds. If the overhead can't be cut, PR is dead (tie).

## night5 s30 (2026-10-28 10:41Z - ~11:05Z): tk3/PR lane CLOSED on its zero-overhead ceiling; no build, 0 judge runs, 0 Modal
- Plan from s29 was "cut PR's ~3 us fixed overhead". Before building I bounded what a PERFECT cut buys, from s29's own H100
  real-routing A/Bs (W 4, vs main's base): d32 sliding PR - base = s99 +0.6, s7 -0.2, s13 +5.4 us; the fixed overhead measured
  at k = 0 is +3.2..3.8 us. Zero-overhead PR at d32 = s99 -2.6..-3.2, s7 -3.4..-4.0, s13 +1.6..+2.2 -> mean ~-1.6 us sliding
  (H200 tuner T=32: PR +0.1..+1.5 with overhead -> ~-2..-3 without). d16: PR = non-BAL tk2f = BAL base +4 us -> still +0.2..+0.8
  with zero overhead (no gain). Score value: ~2 us x 0.058 %/us (d32 sl) + ~2 us x 0.020 (d32 full) = <= ~0.16% total.
- Why the ceiling is that low: main's T=32 base (tk2h) ALREADY merges an expert's 2 token groups into one consumer pass (GS16 W
  read once), so PR only re-schedules the same pair work; and the pair body's best possible rate is bounded by s29's microbench
  (wgmma N16 611 vs mma.sync pair-from-smem 671 cyc per 8 KB stage = 9%). In-layer the compact pair row is 2.2x a single row
  (5.3 vs 2.4 us, latency-bound 2000 cyc/18 KB stage); even at the microbench rate the gain over tk2h stays inside ~9% of pair work.
- The overhead cut itself (L1 carveout ~1.1 us from 200 KB dyn smem: needs x of one group back in registers = the unrolled body
  that landed past 64 KB of code in s29 b) would be a 2-3 session rebuild for <= 0.16%. Not started. Branch layer/tk3 a2ed836 stays
  as the record (TK2PR off on main, nothing to port).
- => tk3 lane CLOSED (lead 09:55Z: "if tk3 is dead, say so and take tk2f bs1"): tk2f bs1 is on record dead piece by piece
  (moe n5s37: route->copy n5s23, fused norm s9/n5s28, norm2 PDL n5s19, PSC s12, ring n5s15/n5s29/n5s30, structure n5s33) and its
  fixed cost already overlaps the pre-expert kernels by PDL in-layer (layer n5s23: d1 experts excess 12 us H100, that is the stream
  + consumer, not the fixed part). Lane request posted in COORD. main unchanged (v163); 10-28 left: 4 checks / 7 submits.
- Re-looked for >= 1% levers outside the three closed lanes: attention board #1 3.03x vs ours 2.60x; in-layer the only big
  attention excess is verify full (_gw_verify v16/v32 H200 106/196 us vs gather-bound decode full 87/168: ~0.6-0.8% of score
  if it reached the gather rate), but native verify-full kernels tied/lost twice (~/attn s71, s111) and FP8 QK/PV forms are on
  record (~/attn NOTES s33, s81-s90). No new idea there from me this session.

## night7 s1 (2026-10-29 04:50Z-06:15Z): H200 per-kernel timeline of main (lead 04:50Z) + top-5 non-expert excess; v164 RG_XPRE; 0 judge runs
- main 9d2b3a8 (moe's merge) did not rebuild: moe edited 2 TK2F_REL patch anchors in submission.py, not layer.py. Synced layer.py:
  `python build.py` == submission.py again. RULE for ports: edit layer.py (or moe_src.py), never only submission.py.
- Judge profile of today's submit 725fc518 (2.1350x) copied to profiles/ (it lived in moe's /tmp worktree). Its per-kernel rows
  overlap (PDL), so for gaps: Modal H200 `tools/h200.py . main=submission.py --cmd 'python tools/timeline_all.py submission.py all'`
  (output gzip+base64 to fit the 5 KB stdout cap), 2.5 GPU-min. Full timeline: muse_decode_layer/dev/n7s1_tl_h200.txt.
- Segment table, H200 Modal, sliding layers, us (end of previous segment -> end of this one; ideal = 12-bit bytes at 4.3 TB/s
  achievable, or the BF16 compute floor at ~700 TFLOP/s for T=256 cuBLAS):
  | seg | d1 | v1 | d16 | d32 | v16 (T=128) | v32 (T=256) |
  | qkv (1st kernel) | 11.8/6.6 | 11.4/6.6 | 11.8/6.6 | 13.5/6.6 | 15.7/8.8 bf16 | 19.0/14 |
  | attn+o+bnd1 | 11.9 | 14.6 | 18.8 | 23.0 | 26.6 | 34.1 |
  | bnd1 -> experts' first work (pooler+router+routing+rms) | 16.6/13.5 | 19.3/13.5 | 19.1/13.5 | 21.2/13.5 | 35.3/~18 | 42.5/~28 |
  | experts (moe lane) | 26.4 | 59.5 | 79.2 | 71.9 | 151.7 | 203.3 |
  | tail: expert end -> layer end (norm2+unpool+bnd2) | 6.2/2.3 | 8.9/2.6 | 19.5/11.5 | 20.6/11.5 | 27.1/~15 | 38.3/~26 |
  | total (Modal) | 72.9 | 113.7 | 148.4 | 150.2 | 253.6 | 337.6 |
  Full layers: same front/tail plus attention (_gd_full d16 86 / d32 163, _gw_verify v16 100 / v32 190: closed, gather-bound).
- Top 5 non-expert excess (weighted by score %/us sl+full; upper bounds = removing ALL excess):
  1. bnd1 -> experts: d1 +3, v1 +6, d16 +6, d32 +8, v16 +17 (FR cuBLAS 27.6 + routing 3.3 + rg 3.6 serial), v32 +15 => ~4.5%.
     Closed: pr32/pr128/g30sp/fr off/router placement/rms fold (s41 note: tk2f starts 0.3 us after pr1 via TK2FX).
     Open: T=128/256 routing chain (FR end -> rg end 7.2 v16 / 8.6 v32 us). RG_XPRE below takes ~0.6-0.9 us of it.
  2. tail: d1 +4, v1 +6, d16 +8, d32 +9, v16 +12, v32 +12 => ~4.4%; closed (UPF/UPF2/UPF3, N2U_BOUNDARY, BND_SPIN, n2u8,
     bnd-as-epilogue, norm2->unpool fusion, g30 T=256 s100, proj n5s14 oracles: T=32 GEMMs co-bound, T=128 unpool -3.4).
  3. qkv first kernel: +5..8 us every shape (~2.6%): the per-kernel fixed ramp (fit 3.8 us + B/4.26 TB/s); closed (m1 sweep,
     g30 qkv T=128 s101, qkv split).
  4. attn+o+bnd1: free-attention oracle n5s26 (d1 2.4 .. v32 15.8); o+bnd part 4-8 us. NEW this session, DEAD: T=128 o on g30
     with bnd1 as its PDL secondary (O30_BND_PDL, + CB_HB_TRIG): H100 v16 sl combine end -> bnd1 end base 13.7, g30:o 14.6,
     + CB_HB_TRIG 12.7 but combine_hb slows by as much (bnd1 end 46.7 vs 46.9): tie. g30 o 9.6-11.5 vs cuBLAS 6.9 us (K=1024:
     8 stages, ramp-bound). Not kept.
  5. T=128/256 cuBLAS hand-offs (no PDL): bnd1/bnd2 at T=256 8.2/8.3 us serial vs ~5.5 byte floor; ~0.4 us launch gaps.
- v164 = RG_XPRE (layer.py text patch of moe_routegathern, #define RG_XPRE): the CTA's X row (FR output; complete before the
  PDL primary _routing started, since _routing is a plain launch) + sum of squares + gain loads move above griddepcontrol.wait;
  only ids-dependent work stays after it. H100 timeline rg end after _routing: v16 sl 3.5 -> 2.6, v32 sl 4.9 -> 4.3 us.
  ab_files 4 rounds (base | xpre): v16 sl 292.9/292.5 | 293.8/291.1, full 443.4 | 442.9, v32 sl 374.1 | 372.3, full 625.9 | 624.8,
  v1 -0.5 (noise). files_eq: bitwise = base at v16/v32 (d16/d32 diffs = the known base-vs-base atomics nondeterminism, checked).
  test 13/13. Not checked (~0.05% score: below judge noise); bundle it with the next real change.
- Next build candidates (ranked): (a) T=128/256 _routing + rg in one launch (routing per CTA, self-resetting grid barrier: all
  T CTAs co-resident since twm is rg's PDL secondary and launches only after every rg CTA triggered) ~-1.5..2 us v16/v32;
  (b) T=256 boundaries fused/thinned (FP32 stream bytes): ~-2..3 us each at v32.

## night7 s2 (2026-10-29 05:10-06:05Z): RGR built + measured DEAD; attention-full sized as the biggest layer-lane pool
- RGR = n7s1 NEXT (a): Triton _routing moved into moe_routegathern (template RT: per-CTA sigmoid/top-8 bitwise = Triton's
  lowering: logit = (0 + hi) + lo, x * -MULT * log2e, ex2.approx, + 1, div.full; top-8 on u64 (monotone key << 32 | 255 - id)
  warp max; weight sum in Triton's butterfly order ((p0+p4)+(p2+p6))+((p1+p5)+(p3+p7)), div.full), then an epoch-counter
  grid barrier per T (T=128/256 CTAs co-resident), then rg as before. Bitwise = v164 at v16/v32 (3 seeds each, files_eq),
  test 13/13, rg_eq: every fr+rg input identical. H100 ab_files 4 rounds v164 | RGR: v16 sl 293.2 | 294.9, full 443.8 | 443.4,
  v32 sl 372.6 | 375.3, full 624.8 | 626.0 -> LOSES. Timeline v32 sl (TUNE {}): FR end -> rg end 8.0 (v164) vs 9.5 us; the
  barrier alone ~1.0 us (no-barrier probe 8.5). Why: rg was already a PDL secondary launched under _routing with its X loads
  pre-wait (RG_XPRE), so the fusion only saves a ~0.3 us launch gap and pays the barrier. Code kept as
  muse_decode_layer/dev/n7s2_rgr.diff (against v164 layer.py/moe_src.py); main unchanged (v164).
- n7s1 NEXT (b) T=256 boundaries: already bounded at n5s28 (stream-byte cut b1/b2 0.04%): bnd = 24 MB (FP32 stream r+w 16,
  BF16 branch 4, h 4) at 8.2 us H200 = 2.9 TB/s; recomputing stream1 in bnd2 saves net 4 MB = ~1 us = 0.04%. Not built.
- Sizing (H200 Modal timeline dev/n7s1_tl_h200.txt; KV bytes = batch x 80K x 2 heads x 64 x (K+V) x 1 B = 256 B/token;
  floor = gather at 4.4 TB/s; %/us from the STATE table):
  | full shape | attn kernel us | KV MB | floor us | excess | %/us | score % |
  | d1  | _gd_full+_red_fin 12.6 end | 20.5 | ~6 (latency) | ~5 | 0.062 | 0.3 |
  | v1  | _attn_split+_combine ~14  | 20.5 | ~6 | ~8 | 0.039 | 0.3 |
  | d16 | _gd_full 86.1 (3.8 TB/s) | 328 | 74.5 | 11.6 | 0.029 | 0.34 |
  | d32 | _gd_full 163.3 (4.0 TB/s) | 655 | 149 | 14 | 0.020 | 0.28 |
  | v16 | _gw_verify 99.5 + bal tail | 328 | 74.5 | ~29 | 0.014 | 0.41 |
  | v32 | _gw_verify 189.8 + bal tail | 655 | 149 | ~43 | 0.011 | 0.47 |
  Sum ~2.1% (the only layer-lane pool > 1%; sliding attention free-oracle ~3% but latency-bound, 30% of it ~1%).
  Context: muse_attention_decode board #1 3.054x vs ours 2.596x (17.6% geomean): others' attention IS faster. Verify full
  (_gw_verify, consumer-bound on the H200: ~0.9%) is the biggest piece. ~/attn NOTES s1-s126 mined it hard (FP8 QK, native
  nvf/nverify lost, dtail lost), so a fresh-eyes redesign (like moe's REDESIGN.md: study public FA3/FlashMLA/FlashInfer FP8
  decode kernels, name the structural difference) is the lever, not tuning. Request posted in COORD.
- 0 judge runs, 0 Modal. 10-29 left: 18 checks / 8 submits (layer), Modal layer 2.5/15 used.

## night7 s3 (2026-10-29 05:21-05:50Z): batch-1 full attention (d1 _gd_full+_red_fin, v1) sized; 3 oracles, all DEAD; 0 judge runs
- Lane: lead 05:35Z gave layer "(a) routing+rg one launch (n7s2: dead), then the attention plumbing moe needs". moe has posted no
  plumbing request yet, so I took the unowned batch-1 full-layer attention (latency-bound; my n7s2 sizing ~0.6% d1+v1). Claimed in COORD.
- H200 batch-1 K/V gather floors (NEW: the ~/attn hdiag3/4/5 diags were never run). tools/n7s3_floor.py (native gather of all
  80K rows' 128-B K + V through the page table vs rows 0..n-1; CUDA graph, cold L2, median of 40, minus an empty-kernel graph):
  | d1/v1 full (20.5 MB) | H100 us | H200 (Modal) us |
  | random (page table), best grid 8x132 x U4 | 7.86 (2.61 TB/s) | 6.16 (3.32 TB/s) |
  | dense (physical order), best | 7.26 (2.82) | 5.47-5.66 (3.6-3.7) |
  In-layer H200 (n7s1 timeline): _gd_full 10.7 us + _red_fin tail 3.2 (qkv end -> red_fin end 14.4); v1 qkv end -> _combine end 15.7.
  So d1 full has ~4.5 us inside _gd_full + ~1 us merge hop over the random floor (~0.3% score at 0.062 %/us); dense saves only
  ~0.6 us more at batch 1 (physical order is not the batch-1 lever).
- ORACLE 1 DEAD: _red_fin folded into the o GEMV prologue. Ceiling = skip _red_fin entirely (variants/n7s3_B/C, wrong output):
  H100 timeline (6 replays each) first-_boundary end base 37.5 (med 37.15) / skip with o PDL at gd start 36.3 / skip with o at gd
  end 36.8 us -> <= ~1 us before the fused prologue's own L2 trip (~0.5) = ~0.03% score. Not built. (Whole-layer ab_files of
  the skip variants shows -10 us, but that is garbage attention output changing expert routing: use the segment, not the total.)
- ORACLE 2 DEAD: L2 prefetch (cp.async.bulk.prefetch.L2 evict_last) of pr1's packed pooler W during d1 full attention, while
  HBM sits ~60% idle. (a) issued by the resident _red_fin CTA before its wait: +42..80 us (one SM's bulk-prefetch issue is
  ~0.3-0.5 TB/s and holds the merge). (b) issued by the 512 _gd_full programs (4 KB chunks per lane), 25% / 45% of S+N, at
  program start / end: H100 ab_files d1 full base 106.16 | 108.54 / 107.05 | 110.60 / 108.70 us. Timeline: _gd_full +3.6..4.4
  us, pr1 only -1..-1.5 (pr1 end - bnd end 19.0 -> 17.3..18.1): prefetch traffic slows the random gather, and pr1 isn't purely
  HBM-bound. Same zero-sum as the s-era side-stream prefetch notes.
- ORACLE 3 DEAD: d1 full _gd_full plan variants on the H200 (only H100 before; v126's aplan covered T=16..256). Variants via
  ATT_PLAN at T=1 (variants/n7s3ap, all pass the reference check). H100 ab_files 6 rounds: base 106.06, stages 3 106.86, stages 4
  107.19, splits 1023 105.86, BN16 s4 106.98, splits 255 110.42, BN64 s2 111.60. H200 Modal ab_files 8 rounds (one container):
  base 87.51, s3 88.14, p1023 87.94, BN16 s4 88.70 -> base plan is the H200 optimum.
- Left in this pool: only a new batch-1 full kernel reaching the random floor (~-4.5 us d1 full = ~0.28%, v1 similar ~0.25%).
  ~/attn s73 built a native one (cfl: clusters + cp.async 3 stages + DSMEM/global two-level merge) and lost 19 vs 15.8 us H100
  (loads-only 12.2, compute 3 us unhidden, merges 4.5). A redo would need: all of an SM's ~155 KB of K/V in flight in ONE trip
  (fits smem at 132 CTAs), compute pipelined per arriving chunk, and red's atomics merge (0.3-0.4 us) not a two-level merge.
  Multi-session, history of losses: not started.
- Modal: layer 6.2 / 15 GPU-min used 10-29 (this session 1.2 + 2.5). 0 judge runs. Main unchanged (v164).
- v165 (d2bb0c2) = v164 (RG_XPRE) + proj n7s3 BNOS handoff (r-proj e11cc44 muse_bf16_proj/dev/n7s3/bnos_layer.patch, applied to
  layer.py with patch -p0, build.py): T=256 bnd1 skips its 8 MB FP32 stream store (stores rb per row), bnd2 recomputes it bitwise.
  test 13/13; files_eq: v32 bitwise = main (other diffs = the known atomics nondeterminism, main vs main differs at v1 sl too);
  H100 ab_files 4 rounds v32 sl 373.2 -> 370.7, full 625.4 -> 623.8, v1/v16 noise.
  CHECK 4787d92e **2.1632x** (> board 2.1580): d1 66.6/75.3, d16 123.8/203.8, d32 148.9/307.7, v1 107.7/120.7, v16 265.2/347.9,
  v32 320.2/490.1. SUBMIT b87bd6aa **2.1219x** (low draw: v32 sl 361.9 / full 533.7, d16 full 211.4 on the same code). Board stays
  3e3b9fd3 2.1580. Don't resubmit v165 unchanged. 10-29 left after: 17 checks / 7 submits.

## night7 s4 (2026-10-29 05:52Z-): lead 06:00Z BIG-BET step 1 = measured HBM-busy profile (H100, d1 sliding) + early-tk2f oracle
- New tools: tools/hbm_run.py (replays one shape N times: 512 MB flush, ~25 us spin, replay) under
  `nsys profile --gpu-metrics-devices=cuda-visible --gpu-metrics-frequency=200000 --gpu-metrics-set=gh100 --cuda-graph-trace=node`
  (nsys = /usr/local/cuda-13.0/bin/nsys), then `nsys export --type sqlite` and tools/hbm_fold.py <sqlite> <reps>: folds the 5 us
  DRAM read+write % samples onto the replay start (least squares, 0.5 us bins). Sampler drops ~70% of the time (1 s gaps):
  use >= 15000 replays. Calibration: sample timestamps lag by ~5 us (the flush kernel's rising edge): OFF_US=5.
  Check: bytes integrated over the d1 sliding replay = 187.9 MB (expected ~190): the profile is trustworthy.
- H100 d1 sliding (main v165, TUNE on, under nsys 84.9 us): HBM >= 80% of peak (3.35 TB/s) only 48.5 of 84.9 us; < 50% for ~26 us
  in 5 bubbles at op boundaries: qkv drain -> o/csl (10-14.5 us, ~4.5), o -> pr1 (20-22, ~2), **pr1 drain -> tk2f ramp (39-47, ~6)**,
  inside tk2f (66-70, ~3: gemm13 -> gemm2 hand-off), tail tk2f drain + n2u + bnd (75-85, ~8). 188 MB at the busy rate (3.2 TB/s)
  = 59 us vs 85: the d1 non-streaming time is ~26 us H100 (~30%). So at d1 the 'HBM idle at op boundaries' premise HOLDS
  (unlike proj n7s4's d16/d32 piece).
- Early-tk2f oracle at T=1 (fills the pr1 -> tk2f bubble by launch order): rmsnorm triggers tk2f at its START (TRIG) instead of
  after its wait (TRIG2), so tk2f CTAs land on SMs freed by pr1's router blocks (~mid pr1). 1st form (tk2f reads the live ids)
  HANGS: CTAs read ids while pr1's routing block writes them -> inconsistent per-CTA routing -> the gemm2 cnt == 256 waits never
  complete. RULE: any early tk2f needs a release flag from pr1's routing block (ids/lids/wts written, then flag), acquire-spun by
  tk2f before routing. Timing-only form (/tmp files, not kept; tk2f reads a persistent copy of the previous call's ids, copy
  in both variants): H100 timeline tk2f starts 31.8 us (during pr1) vs 46.3, but ends 33.7 us after pr1 ends (base 34.2) and
  pr1 is +1.7 us slower; ab_files 4 rounds d1 sl 88.73 -> 88.25, full 109.05 -> 108.51 = TIE. Why: tk2f's T=1 split is static
  per blockIdx (TK2F 204, non-DYN): early CTAs finish their fixed share early, CTAs landing at pr1's end still do a full share.
- => To turn the measured d1 bubble (~5 us H100 pr1 drain + tk2f ramp) into time, tk2f at T=1 would need DYN work stealing
  (TK2F_DYN, which n2u's y32 order currently rules out at M = 1) + the pr1 ids flag; even then early CTAs take HBM from pr1
  (zero-sum while pr1 streams), so the ceiling is the drain bubble only: ~2-3 us H200 d1 (~0.4% score), similar at v1 (pr8).
  Not built this session. The other d1 bubbles: qkv -> o (4.5 us: csl latency; o W prologue already PDL-streams), tail (8 us:
  tk2f drain + n2u + bnd; UPF/UPF2/UPF3/N2U_BOUNDARY on record dead).
- 0 judge runs, 0 Modal. main unchanged (v165).

## night7 s5 (2026-10-29 07:12Z-): v166 (LTFR) checked = tie; HBM bubble map v1/d16; _routing packed-max (bitwise, -0.2..-0.9 us/call)
- v166 (4b68c55) = v165 + proj n7s5 LTFR (r-proj muse_bf16_proj/dev/n7s5/ltfr_layer.patch, patch -p1, build.py): bench test 13/13,
  files_eq verify 32 bitwise, validate ALL OK (cold prepare 63.2 s). CHECK 93027e57 **2.1341x** (fixed-SGLang 2.1354; v165 check
  4787d92e 2.1632 / fixed 2.1713): d1 68.4/74.2, d16 123.8/205.7, d32 156.1/304.4, v1 110.9/130.2, v16 259.4/349.8, v32 328.7/507.1;
  prepare_s 88.2. LTFR invisible in the noise (v32 sl 328.7 vs 320.2). Not submitted. 10-29 left: 16 checks / 7 submits.
- Judge noise source found: every judge run draws NEW input seeds per case (json case.seed differs run to run), so routing (hot
  experts) differs between runs; v32 sl over 13 runs of near-identical code: 320-378 us. Not controllable; only kernels that
  are robust to skew help.
- hbm_fold (n7s4 tools) on v1 and d16 sliding (H100, main v166, under nsys; replay 140.2 / 189.3 us):
  v1: HBM < 50% of peak ~34 of 140 us: qkv->vtk/o 11-14 (~3), 17-19 (~1.5), pr8 phase 23-31 at ~1.0-1.4 TB/s (pr8 waiting on
  o -> bnd -> rmsnorm; ~4-5 us lost), pr8 drain -> tk2f ramp 47-51 (~4), inside tk2f 102-106 (gemm13->gemm2, ~4), tail 129-140
  (tk2f drain + nu8 + bnd + graph end, ~10). d16: < 50% for ~39 of 189 us: 12-14, 19-21, 25-28 (attention/o hand-offs ~7),
  **50-59 (~9: pooler/router drain -> _routing -> tk2f ramp)**, 166-169 (tk2f drain -> norm2), 184-189 (tail).
  H100 single-replay timeline d16 sl: g24 pooler 30.6-57.4 | _router_split 36.6-56.9 | _routing 39.8-60.7 (PDL; ends 3.8 us after
  the router) | _rmsnorm_gain 57.6-60.1 (parallel) | moe_tk2f 60.9 -> so _routing's post-wait time is on the critical path.
- _routing packed max (dev/n7s5_routing.py): per top-k step ONE int64 max of (order-preserving key bits << 32 | E-1-e) instead of
  max + argmin + score-sum (3 reductions), scores tl.gather'ed once. Bitwise 0/360 (T 1..256, forced exact key ties). Graph chain of
  50 launches, H100 us/launch old -> new: T16 3.86 -> 2.95 (4 warps; 2 warps 3.04), T32 3.85 -> 2.98, T128 2.86 -> 2.64 (1 warp),
  T256 2.87 -> 2.66. Put into layer.py in place (no flag: bitwise), T<=32 now 4 warps (was 2).
- In-layer H100 ab_files (6-8 rounds, v166 vs v167): d16 sl 190.0 -> 189.4 / 189.1 -> 188.4, full 292.8 -> 292.0 / 292.2 -> 291.0, v16 sl
  293.6 -> 291.6, full 442.9 -> 441.4, v32 sl 369.9 -> 368.9, full 624.7 -> 623.7; d32 sl looked +1.9, but a duplicated-file A/B
  (old, new, old', new') gave 185.0 / 186.2 / 186.0 / 184.4: file-placement noise is +-1 us per shape, so read single-pair
  ab_files diffs < ~1.5 us as noise. At d32 _routing is off the critical path anyway (H100 timeline: g30k pooler ends after it).
  files_eq: topk ids identical everywhere; float diffs at d16/d32 full = the known atomics nondeterminism (base vs base differs too).
- v167 (main) = v166 + packed-max _routing: bench test 13/13, validate ALL OK (cold prepare 66.4 s). NOT checked yet: held for the
  lead's 07:05Z bundle (moe's T=256 unpool skip) since ~0.9 us/call x 4 loads is below judge noise.
- Big idea posted (Requests, lead 06:50Z task): T<=8 early tk2f with pr1/pr8 ids release flag + DYN split, ~0.4-0.5%, 2 sessions.
- 1 judge check (93027e57), 0 submits, 0 Modal. 10-29 left: 16 checks / 7 submits; Modal layer 6.2/15.

## night7 s6 (2026-10-29 07:45Z-): v167 checked; hbm_fold summary posted; glue-fusion idea sized dead
- CHECK 59013cf5 (v167, 1f12ceb) **2.1553x** (fixed-SGLang 2.1482; v166 2.1362, v165 2.1721, board 3e3b9fd3 2.1241 fixed):
  d1 67.2/74.7, d16 127.4/205.6, d32 162.6/310.5, v1 108.2/123.4, v16 258.0/345.8, v32 316.7/494.5; < board 2.1580 -> not submitted.
  v16 sl 258.0 and v32 sl 316.7 are our best ever for those shapes; d32 sl 162.6 drew slow (148.9 in v165's check).
- Sized (H100 timelines, no build): fusing _routing into the pre-expert _rmsnorm_gain at T=16/32. d16 sl: pooler ends 58.2,
  router 57.6, _routing 60.5, rmsnorm 58.9-61.4 (parallel, TUNE tk2fj off at d16 sl), tk2f 62.0; d32 sl: g30k pooler is the long
  pole (67.2), routing done at 65.5. The two hand-offs already overlap: fused form <= ~1 us d16, 0 d32. Not built.
- v1 sl H100: o (g30) 14.5-26.5, bnd ends 29.6, pr8 19.5-50.7 (pre-wait L2 prefetch already, pr8_npf 220), tk2f 51.1 (tk2fx PDL).
- 1 judge check, 0 submits, 0 Modal. 10-29 left: 15 checks / 7 submits.
