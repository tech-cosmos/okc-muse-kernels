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
