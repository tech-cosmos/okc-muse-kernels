# STATE-moe (muse_moe_w4a16)

## Board
- Best: submit 3795f52627474eca @212c0bb, **1.311x** (rank 562 on 2026-10-25). Top of board 2.52x.
- Same code checked as 5d8cdaec8dd34a58: 1.317x. Per shape (H200, us; ours = SGLang / paired ratio):
  | load | bs | SGLang | ours | x | floor |
  |---|---|---|---|---|---|
  | decode 1 | 1 | 77.3 | ~42.5 | 1.82 | 11.4 |
  | decode 16 | 16 | 195.6 | ~159 | 1.23 | 52.5 |
  | decode 32 | 32 | 203.6 | ~211 | 0.97 | 70.8 |
  | verify 1 | 8 | 143.0 | ~106 | 1.34 | 37.3 |
  | verify 16 | 128 | 466.4 | ~340 | 1.37 | 101.3 |
  | verify 32 | 256 | 616.8 | ~471 | 1.31 | 108.9 |
- H200 times ~= local H100 times: we are latency/ALU-bound, not HBM-bound yet.

## What's on main (submission.py)
route (1 CTA) -> gemm13 (persistent, items = active expert x 16R-row tile pair, 8 warps split K,
mma.sync m16n8k16 bf16, NVFP4 repacked into A-fragment order with bit-permuted codes) -> gemm2 ->
norm (per token). Plan: M<=32 (R,G)=(1,1); else (2,2). Builds with load_inline (~35 s).
**Judge rejects `subprocess` ("starts processes")**: keep load_inline.

## Local harness
`muse_moe_w4a16/bench.py test|bench|sweep`. Local H100 bench of main: 51/151/217/115/339/494 us.
Routing at seed 7: bs1 3 experts x1 token; bs16 13 experts (max 4 tok); bs32 18 (max 7);
bs8 10 (max 2); bs128 25 (max 28); bs256 26 (max 49).

## Ranked ideas
1. Big-M path: K-split across warps re-reads activations per 32-row tile (L2 traffic ~ weights);
   use warps along rows + activations staged in smem (or wgmma with A from registers, B smem), all
   tokens of an expert in one pass (G up to 8).
2. Decode path at ~2 TB/s: overlap items (prefetch next item's weights / TMA bulk ring), more
   occupancy (fewer regs), split-K across CTAs when nact small.
3. Fuse norm into gemm2 epilogue (sumsq atomics) and drop a launch; route is ~2 us.
4. Load balance: hot experts' items first (done: active list sorted by count).

## Dead ends
- (R,G) bigger G with K-split design: worse (activation traffic, regs).
