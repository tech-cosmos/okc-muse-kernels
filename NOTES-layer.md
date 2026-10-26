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
