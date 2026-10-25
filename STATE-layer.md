# STATE-layer (muse_decode_layer): short current picture; NOTES-layer.md is the long log

Updated 2026-10-25 20:25Z (end of session 1).

## Board
- **Best scored: 1a7c4997 = main 5a95d4a (v6), 1.4586x, 15,508 us/step.** (Session start: none; 1.000x baseline
  53d9d188.) Board #1 2.285x (9,900 us). Budget 2026-10-25: 5 of 6 submits, 7 of 20 checks used.
- v6 per shape, H200 (SGLang / ours, us): d1 171.9/104.0, d1f 180.0/110.4, d16 258.6/178.5, d16f 431.9/255.2,
  d32 255.6/219.0, d32f 606.1/378.4, v1 262.8/157.1, v1f 281.3/174.1, v16 518.3/391.5, v16f 690.9/471.9,
  v32 705.3/578.3, v32f 1044.5/737.8. Weakest loads: decode 32 (1.17 sliding), verify 32 (1.22), verify 16 (1.32).
- Floors (us): d1 40/44, d16 77/137, d32 85/208, v1 62/66, v16 127/189, v32 144/267. Weights: sliding 46, full 16.

## Code (muse_decode_layer/)
- `layer.py` = the layer; `attn_src.py` = ~/attn submission.py @82866c6; `moe_src.py` = r-moe submission.py @bb49efd.
  **`python build.py` writes submission.py**: inlines both (attention: kernel_fn -> attn_kernel_fn, plus a
  constexpr_function replacing triton.next_power_of_2 inside a jit body, which SGLang's import breaks in the judge;
  moe: every top-level name prefixed `_moe_` via tokenize). Never edit submission.py by hand.
- Pipeline: qkv F.linear -> attn kernels -> o F.linear -> `_boundary` (Triton, 1 CTA/row, 16 warps) -> side stream:
  router (`_router_fma` FP32 <= 8 tokens, `_router_split` hi/lo tl.dot above) + `_routing` (+ moe bucketing <= 16
  tokens) || pooler F.linear -> `_rmsnorm_gain` -> experts (moe kernels; Marlin at T=32: MARLIN_TOKENS) ->
  unpooler F.linear (1024-row slice <= 8 tokens) -> `_boundary`. `_post_expert_wsum` is for the Marlin path <= 16 tokens.
- bench.py `test` (13 judge tests), `bench`, `prof`; tools/timeline.py (per-kernel start/end of one replay),
  tools/micro*.py, tools/route_stress*.py. Locally everything runs (moe kernels) except Marlin (T=32 uses moe locally).
- diag_experts.py: a prepare-time profiler of the Marlin path, for diag checks only (never submit).

## Where the time goes (local H100 decode 1 sliding 132 us, tools/timeline.py)
qkv GEMM 22.6 (38 MB at 1.8 TB/s), attention 4.8, o GEMM 10.6, boundary 4.3, pooler 29, prenorm 2, experts 46,
unpooler 8, boundary 3.7. GEMMs ~70 us = proj's lane (Handoff awaited); experts = moe's lane.
Judge-side at T>=128: experts dominate (moe kernel ~300-450 us of verify 16/32).

## Next (ranked)
1. Watch ~/COORD.md Handoffs: proj GEMV/GEMM for qkv/o/pooler/unpooler at m=1..256 (biggest small-T lever:
   ~25-30 us of 104 at decode 1); newer moe commits (re-copy moe_src.py, re-check T=32 vs Marlin).
2. Fusions in my lane once GEMMs are ours: boundary into the o/unpooler GEMM epilogues via per-row partial sums
   (sum a^2, (beta a)^2, (alpha s)(beta a), (alpha s)^2) consumed in the pooler/router prologue; prenorm folded as
   a per-token scale after gemm13 (needs moe kernel support: W13 (x g r) = r W13 (x g)).
3. Drop splitK-reduce kernels (1.7-2 us each, 3 per step) when GEMMs are ours.
4. Routing risk: diag check f5fdc570 failed one timed rep's routing (3.18e-2 > 0.03 margin). Local stress 0/120.
   If a real run fails this way, study score error upstream of the router (attention FP8 queries).

## Dead ends
- Multi-CTA boundary with atomics + spin row barrier: 6.6-8.3 us vs 5.0 single-CTA (H100).
- PDL on glue kernels after a cuBLAS producer: no gain.
- Side-stream moe bucketing at T=128/256: +29..+67 us on the judge (fine on H100).
- moe bb49efd at T=32 vs Marlin: 222.0 vs 219.7 (tie/loss).
