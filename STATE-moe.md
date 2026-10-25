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
Tools in muse_moe_w4a16/tools/ (copies of the /tmp scripts): ncu_t.sh (H200 proxy), seeds.py (200 routings), cmp.py, moe_regs.sh.
Judge JSON summary: `python /tmp/moe_sum.py <okc --json output>` (recreate if lost: prints per-shape median vs baseline).
Per-kernel timing: /tmp/moe_k.py <bs> '[cfg,...]' (gemm13/gemm2 alone, cold L2); /tmp/moe_p.py <bs> <cfg> for ncu.
`muse_moe_w4a16/bench.py test|bench|sweep`. Local H100 bench of main: 51/151/217/115/339/494 us.
Routing at seed 7: bs1 3 experts x1 token; bs16 13 experts (max 4 tok); bs32 18 (max 7);
bs8 10 (max 2); bs128 25 (max 28); bs256 26 (max 49).

## Ranked ideas
1. Big-M (claimed by moe for s4 in COORD Requests; layer's notes on budget there) (bs128 1.82x, bs256 1.75x; 2 of 6 shapes) is now the largest gap: 2.7 / 2.1 TB/s on H200 vs ~4 for
   decode. Profile moe_big on the ncu proxy (`/tmp/ncu_t.sh 256 old118`-style; sk_ncu.py's "old" mode
   forces the old path for any M). Ideas: TMA weights + x per kc staged once per CTA (moe_big structure, but
   bulk copies, 1 CTA/SM, consumer warps = row tiles), persistent rows balanced by cost; fuse route.
2. (Matters MORE in the layer: its d16/d32 top expert has 13-20 tokens, COORD Requests.) Decode bs32 (1.52x): experts with > 8 tokens re-read weights per 8-token group (+9.4% bytes at bs32,
   +3.2% bs16 on typical routings). 16-token groups with an NG=2 path (tried, see dead ends) need (a) no
   spills (NG=2 path out of the hot loop or a separate kernel instantiation chosen per CTA) and (b) CTA ranges
   balanced by cost (NG=2 rows ~1.3x), not row count.
3. Launch overhead: PDL (programmatic dependent launch: cudaLaunchKernelEx + griddepcontrol) so gemm2/norm
   prologues (barrier init, LUT, meta) overlap the previous kernel; norm2 fused into gemm2 is hard (needs all
   of y per token: a grid-wide dependency).
4. Decode instruction trims (H200 is close to issue-bound again at bs8/16): LUT index ops (12/unit), 4-rt units
   (needs 16 KB/stage), all-lane mbarrier arrive vs lane-0 arrive.
5. Clean-up: remove moe_sk / moe_tk / moe_gemm / route2 / act from submission.py (compile time ~40 s now).

## Dead ends
- moe_tk2 with 16-token groups + per-segment NG=2 path (s3): proxy 3-14% slower on most routings (spills
  48-64 B; rows balanced by count while NG=2 rows cost more), one bs32 routing 12% faster. Reverted.
- Register-ring stream-K decode (moe_sk, s3): ~19-21 instr/word, latency-bound (13.9 cycles/issue at 4
  warps/scheduler); TMA ring + x in registers is what worked.
- moe_tk with NC=16 consumer warps: worse than NC=8 (proxy 186 vs 159 at bs16).
- moe_big with 2 row tiles/warp (8 warps): spills at 128 regs, worse. 16 warps x 1 rt, 1 CTA/SM: worse
  than 8 warps x 2 CTA/SM (tail: halving item size helped 10%). NS=4: worse than NS=8.
- moe_stream (decode): old item structure + per-lane cp.async ring of weights AND x (no barriers, D=7 ahead
  across items), 1 CTA/SM: slower (bs32 178/115 vs 147/75 us): 2 warps/scheduler can't issue enough.
- Bigger G with the K-split design: worse (activation traffic, registers).
- Warp-independent full-K items with rolling D-chunk register prefetch (no smem reduce): bs16
  155-175 vs 136 us, bs32 213-237 vs 191 (x loads not prefetched, 2 CTAs/SM).
- minBlocks 5 for (1,1): spills, much worse (63/162/230/133).
