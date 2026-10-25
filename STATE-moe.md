# STATE-moe (muse_moe_w4a16)

## Board
- Best: submit 214fcd1319a24904 @1567586, **1.438x** (2026-10-25 ~20:45Z). Top of board 2.52x.
- Check of same code d82327d852c148a0: 1.424x. Per shape (H200, us):
  | load | bs | SGLang | ours | x | floor |
  |---|---|---|---|---|---|
  | decode 1 | 1 | 80.5 | 41.3 | 1.95 | 11.4 |
  | decode 16 | 16 | 188.7 | 136.6 | 1.38 | 52.5 |
  | decode 32 | 32 | 204.9 | 184.0 | 1.11 | 70.8 |
  | verify 1 | 8 | 143.3 | 96.2 | 1.49 | 37.3 |
  | verify 16 | 128 | 473.2 | 331.7 | 1.43 | 101.3 |
  | verify 32 | 256 | 614.5 | 468.4 | 1.31 | 108.9 |
- H200 times ~= local H100 times (H100 bench: 46.6/136/191/107/333/481): we are latency/issue-bound,
  not HBM-bound. ncu (bs32 gemm13): DRAM 2.0 TB/s (60%), issue 0.48/scheduler, L1TEX-scoreboard stalls.

## What's on main (submission.py)
route (1 CTA, parallel scan + rank) -> gemm13 (persistent, items = active expert (heaviest first) x
16R-row tile pair, 8 warps split K (8 x 512), all 8 k-chunks of weights loaded up front, mma.sync
m16n8k16 bf16, smem atomics reduce, epilogue silu*up) -> gemm2 (same kernel) -> norm (per token).
Plan: M==1 cfg 10 (R1 G1, minBlocks 3); M<=32 cfg 11 (R1 G1, minBlocks 4); else 22 (R2 G2, minBlocks 2).
**Occupancy is everything**: (2,2) at 151 regs (1 CTA/SM) was 431/603 us vs 333/481 at 128 regs.
Builds with load_inline (~35 s). **Judge rejects `subprocess`** ("starts processes").

## Local harness
`muse_moe_w4a16/bench.py test|bench|sweep`. Local H100 bench of main: 51/151/217/115/339/494 us.
Routing at seed 7: bs1 3 experts x1 token; bs16 13 experts (max 4 tok); bs32 18 (max 7);
bs8 10 (max 2); bs128 25 (max 28); bs256 26 (max 49).

## Ranked ideas
1. Big-M path (bs128/256, 2 of 6 loads, ~2.5-3x headroom): K-split across warps re-reads activations
   per 32-row tile and re-streams weights per 16-token pass. Design: CTA = 8 warps along rows
   (each warp gate+up rowtile pair), all n_e tokens of the expert staged per 64-k chunk in smem
   (cp.async, row stride 144 B = conflict-free LDS.128 for my B layout), weights straight to regs
   with prefetch, dequant once per weight, G up to 8 groups. Better still: wgmma m64nNk16 with A
   from registers (same per-warp A fragment layout as mma.sync) and B (tokens) from smem.
2. Decode (bs8-32, 3 loads at 1.1-1.5x): still ~2 TB/s. Overlap items: cp.async/TMA-bulk weight
   ring in smem (each warp's slice of an item is contiguous 4 KB per rowtile in the packed layout),
   so loads never stop at item boundaries / reductions. Fewer instructions per weight (SASS check).
3. Fuse norm into gemm2 (sumsq atomics + last-CTA finish) and drop a launch; route ~2 us.
4. bs1: 3 experts only -> 768 items; maybe split-K across CTAs.

## Dead ends
- Bigger G with the K-split design: worse (activation traffic, registers).
- Warp-independent full-K items with rolling D-chunk register prefetch (no smem reduce): bs16
  155-175 vs 136 us, bs32 213-237 vs 191 (x loads not prefetched, 2 CTAs/SM).
- minBlocks 5 for (1,1): spills, much worse (63/162/230/133).
