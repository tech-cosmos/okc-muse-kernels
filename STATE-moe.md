# STATE-moe (muse_moe_w4a16)

## Board
- Best: submit 5706bc37ac4041ff @6a1cda0, **1.589x** (2026-10-25 ~21:15Z). Top of board 2.52x (#10 2.37x).
- Check of same code 58a1676977ea4978: 1.586x. Per shape (H200, us; SGLang from the same run):
  | load | bs | SGLang | ours | x | floor | experts touched (seed 7) |
  |---|---|---|---|---|---|---|
  | decode 1 | 1 | 79.9 | 42.4 | 1.89 | 11.4 | 3 |
  | decode 16 | 16 | 186.3 | 127.8 | 1.46 | 52.5 | 13 |
  | decode 32 | 32 | 203.7 | 174.5 | 1.17 | 70.8 | 18 |
  | verify 1 | 8 | 140.9 | 89.1 | 1.58 | 37.3 | 10 |
  | verify 16 | 128 | 470.4 | 262.8 | 1.79 | 101.3 | 25 |
  | verify 32 | 256 | 617.8 | 348.2 | 1.77 | 108.9 | 26 |
- The cheaper dequant (6a1cda0) gained 6-7% on H200 decode with NO change on local H100 = proof that
  H200 decode is issue-bound.
- One expert = 18.9 MB w13 + 9.4 MB w2 (incl. scales). Decode runs at ~2.7 TB/s on H200 AND on local H100:
  **decode is issue-bound, not HBM-bound** (old kernel bs32 gemm13: 57% issue, ALU pipe 53% on H100). On
  H200 every instruction per weight word counts; local H100 timings can't show it, use ncu inst counts.

## What's on main (submission.py)
route (1 CTA) -> gemm13 -> gemm2 -> norm (per token). Two GEMM kernels, same packed weights:
- `moe_gemm<R,G,GATED,MB>` (decode, M <= 32; plan 10 for M==1 (minBlocks 3), 11 otherwise): items = active
  expert x 16-row tile (gated: gate+up pair), 8 warps split K (512 each), all weights of the item loaded
  up front, mma.sync, smem atomic reduce, epilogue silu*up. Occupancy is everything here (64 regs, 4 CTA/SM).
- `moe_big<GATED,NS=8,NW=8,MB=2>` (M > 32, cfg 118): item = active expert x (gated: 32 gate + 32 up rows /
  w2: 128 rows), work-stealing counter (meta[66/67], zeroed by route), each warp ONE 16-row tile over full
  K, x for all tokens (<= 64 per pass) staged per 64-k chunk in smem with the warps' weights (8-stage
  cp.async ring, 144 B row stride = conflict-free LDS.128), NG (token groups) is a TEMPLATE (1,2,4,8):
  a runtime `if (gi < ng)` around mma compiles to predicated HMMAs that still issue (was 2x slower).
- dequant dq(): o0 = q&M, o1/o2 = rotl(q,10/13)&M, o3 = ((q&0x001B001B)*16648)&M, M=0x81C081C0; 4 x
  fma.rn.bf16x2 with LUT scale. 12 instr per 8 weights. (_POS in prepare must match.)

## Local harness
Judge JSON summary: `python /tmp/moe_sum.py <okc --json output>` (recreate if lost: prints per-shape median vs baseline).
Per-kernel timing: /tmp/moe_k.py <bs> '[cfg,...]' (gemm13/gemm2 alone, cold L2); /tmp/moe_p.py <bs> <cfg> for ncu.
`muse_moe_w4a16/bench.py test|bench|sweep`. Local H100 bench of main: 51/151/217/115/339/494 us.
Routing at seed 7: bs1 3 experts x1 token; bs16 13 experts (max 4 tok); bs32 18 (max 7);
bs8 10 (max 2); bs128 25 (max 28); bs256 26 (max 49).

## Ranked ideas
1. Decode (4 of 6 loads, 1.1-1.95x): cut instructions per weight word. Old kernel ~22 instr/word of which
   dq+mma = 13; rest = per-item overhead (items are tiny: 1 rt pair x 8 kc per warp): smem red zeroing,
   atomics, 3 barriers, epilogue, pointer setup (LDL!). Options: bigger items (R=2) at same regs; or
   moe_big-style decode with split-K across CTAs + fp32 red.global into a zeroed accum (zero it in a
   multi-CTA route kernel), gemm2 computing silu*up from fp32 partials while staging x, pipelined across
   items (flattened (item, kc) issue cursor). Check every candidate with ncu inst count + issue %.
2. bs1: 3 experts; moe_big gemm2 has only 96 items -> idle SMs; needs split-K.
3. Big-M: ~2.1 TB/s locally (ncu: issue 48%, DRAM 57%). Tail: items/CTA ~3-6. Fuse gemm13+gemm2 in one
   persistent launch with per-expert completion counters; scale LUT LDS -> fewer ops.
4. Fuse norm into gemm2; route ~2 us.

## Dead ends
- moe_big with 2 row tiles/warp (8 warps): spills at 128 regs, worse. 16 warps x 1 rt, 1 CTA/SM: worse
  than 8 warps x 2 CTA/SM (tail: halving item size helped 10%). NS=4: worse than NS=8.
- moe_stream (decode): old item structure + per-lane cp.async ring of weights AND x (no barriers, D=7 ahead
  across items), 1 CTA/SM: slower (bs32 178/115 vs 147/75 us): 2 warps/scheduler can't issue enough.
- Bigger G with the K-split design: worse (activation traffic, registers).
- Warp-independent full-K items with rolling D-chunk register prefetch (no smem reduce): bs16
  155-175 vs 136 us, bs32 213-237 vs 191 (x loads not prefetched, 2 CTAs/SM).
- minBlocks 5 for (1,1): spills, much worse (63/162/230/133).
