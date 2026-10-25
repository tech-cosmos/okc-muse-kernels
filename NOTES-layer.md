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
