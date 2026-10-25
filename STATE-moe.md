# STATE-moe (muse_moe_w4a16)

## Board
- Best check: 5708ff8cb783444c @c7dce17, **1.888x** (2026-10-25 ~21:40Z); submit of the same code: see NOTES / COORD
  (previous best submit 5706bc37 1.589x @6a1cda0). Top of board 2.52x (#10 2.37x).
  | load | bs | SGLang | ours | x | floor | old (6a1cda0) |
  |---|---|---|---|---|---|---|
  | decode 1 | 1 | 80.0 | 31.9 | 2.51 | 11.4 | 42.4 |
  | decode 16 | 16 | 188.7 | 101.0 | 1.87 | 52.5 | 127.8 |
  | decode 32 | 32 | 204.2 | 134.1 | 1.52 | 70.8 | 174.5 |
  | verify 1 | 8 | 142.4 | 71.4 | 1.99 | 37.3 | 89.1 |
  | verify 16 | 128 | 473.5 | 260.5 | 1.82 | 101.3 | 262.8 |
  | verify 32 | 256 | 618.5 | 353.5 | 1.75 | 108.9 | 348.2 |
- One expert = 18.9 MB w13 + 9.4 MB w2 (incl. scales).
- **H200 proxy that works**: ncu kernel durations (ncu locks the SM clock at base 1.45 GHz, memory unchanged:
  same compute/BW ratio as H200 at boost). Judge H200 time ~= 0.72 x ncu total. `/tmp/ncu_t.sh <bs> <cfg>`
  (cfg `t8` = moe_tk2 8 stages; `old11` = old moe_gemm path; scripts below). Local wall clock can't rank
  decode kernels (H100 is HBM-bound where H200 is issue-bound).

## What's on main (submission.py)
M <= 32 (decode, bs8): 3 launches, `moe_tk2<NRT, NC=8, NSTG=8, GATED>` x2 + `moe_norm2`:
- 1 CTA/SM, 8 consumer warps + 1 producer lane. Work = (active expert, 8-token group, row-tile pair rp)
  "rows" R; CTA b takes rows [nrp*b/grid, nrp*(b+1)/grid). Per rp the producer streams 8 blocks; block q =
  8 units (k chunks q*8..q*8+7) = 3 cp.async.bulk copies (gate tile 4 KB, up tile 4 KB, scales 1 KB) into an
  8-stage smem ring (mbarrier full/empty; empty count = 256: every consumer lane arrives).
- Consumer warp w does k chunk q*8+w of every block, so its x fragments (8 kc x 2 uint4) stay in registers
  for a whole segment. Per rp: smem cross-warp reduce (double-buffered s_red, 1 named barrier), epilogue:
  gemm13 -> bf16 silu(gate)*up into inter[pos]; gemm2 -> FP32 y[pos] (plain stores, no atomics/zeroing).
- gemm13 CTAs compute the routing themselves (deterministic: __match_any_sync rank + per-warp counts), CTA 0
  writes meta/sroutes/inv for gemm2 + norm2. Experts with > 8 tokens: one segment per 8-token group (weights
  re-read per group). Scales repacked per pair: S2 [e][rp][kc][lane] u32 (_pair_scales in prepare).
M > 32: unchanged: route -> moe_big gemm13 -> moe_big gemm2 -> norm (cfg 118, below).
- `moe_big<GATED,NS=8,NW=8,MB=2>`: item = active expert x (gated: 32 gate + 32 up rows / w2: 128 rows),
  work-stealing counter (meta[66/67], zeroed by route), each warp ONE 16-row tile over full K, x for all
  tokens (<= 64 per pass) staged per 64-k chunk in smem with the warps' weights (8-stage cp.async ring), NG
  template (1,2,4,8).
- dequant dq(): o0 = q&M, o1/o2 = rotl(q,10/13)&M, o3 = ((q&0x001B001B)*16648)&M, M=0x81C081C0; 4 x
  bf16x2 mul with LUT scale. 12 instr per 8 weights. (_POS in prepare must match.)
- Also in the file (not used by default): moe_sk (register-ring stream-K + atomics, SK_CFG path when
  TK2_NSTG=0), moe_tk (TMA + atomics), moe_gemm (old decode), route2/act. Clean up when convenient.

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
