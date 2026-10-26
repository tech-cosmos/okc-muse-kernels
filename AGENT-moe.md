# Agent "moe": muse_moe_w4a16

Your problem is `muse_moe_w4a16` (dir `muse_moe_w4a16/`, notes NOTES-moe.md, STATE-moe.md).
It is 30% of the decode step on the overall board, and the experts are also the biggest slice of
muse_decode_layer: the layer agent will reuse your kernel, so keep its interface clean
(a function taking hidden states, topk ids/weights and your prepared weights) and post it under
"Handoffs" in ~/COORD.md each time it gets faster.

Plan, in order:
1. **Get on the board today.** SGLang's path (baseline.py: moe_align_block_size, two
   moe_wna16_marlin_gemm, silu_and_mul, the Triton post-expert norm) is importable in the judge:
   put it in submission.py, check once, submit as the first correct run (~1.0x).
2. Read the per-shape times. The shapes are weight-bandwidth-bound: one expert's NVFP4 w13 + w2 is
   about 25 MB plus ~3 MB of scales, so the floor (TASK.md: 11-109 us) is set by how many distinct
   local experts the routing touches. Measure how far Marlin is from that floor per shape.
3. Write your own kernel(s): fused W4A16 grouped GEMV/GEMM that reads each routed expert's NVFP4
   weights once, dequantizes in registers (E2M1 + E4M3 block scales + global scale), does gate/up +
   SiLU*up (+ w2) with minimal launches, fuses the post-expert norm and weighted sum, and uses all
   132 SMs even when only a few experts are hot (split-K/split-N; routing is skewed). Triton first
   for speed of iteration, then CUDA/CuTe where it pays.

**Since 2026-10-26: your target is the experts as they run inside muse_decode_layer (the layer's T, routing
and surrounding ops). Your own board is a test bed only. See ~/COORD.md Human decisions.**
