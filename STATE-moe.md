# STATE-moe (muse_moe_w4a16)

## Board
- Best submit: 9a73c24a84054989 @9aa37e0, **2.176x** (2026-10-26 01:05Z; check 74a049c0 same code 2.180x), #46.
  Previous: ee945887 2.131x @128b11a. Top of board 2.55x (#10 2.37x, #20 2.31x).
  Used 2026-10-26: checks 4 of 20, submits 2 (lead cap 9). Per-shape (submit 9a73c24a / check 74a049c0):
  bs1 28.0/28.9, bs16 94.5/96.4, bs32 134.1/129.4, bs8 72.2/67.5, bs128 202.1/195.6, bs256 250.0/247.0.
  Table below = s7 (ee945887) for reference.
  | load | bs | SGLang | ours | x | floor | s6 (fa39ed7) |
  |---|---|---|---|---|---|---|
  | decode 1 | 1 | 80.0 | 29.6 | 2.70 | 11.4 | 30.6 |
  | decode 16 | 16 | 189.5 | 99.9 | 1.90 | 52.5 | 103.4 |
  | decode 32 | 32 | 204.4 | 135.2 | 1.51 | 70.8 | 139.7 |
  | verify 1 | 8 | 143.6 | 70.1 | 2.05 | 37.3 | 72.9 |
  | verify 16 | 128 | 463.9 | 199.5 | 2.32 | 101.3 | 198.0 |
  | verify 32 | 256 | 619.9 | 244.1 | 2.54 | 108.9 | 252.2 |
  Decode shapes vary +-4% between identical-code checks (bs16 99.8-107.2): A/B decode changes against 2-3 checks.
- One expert = 18.9 MB w13 + 9.4 MB w2 (incl. scales).
- **H200 proxy**: ncu kernel durations x 0.72 works for DECODE (moe_tk2). For big-M it does NOT rank changes:
  moe_tw: ncu said -9% (judge -19%); K-split tail: ncu -1% (judge +5%). For big-M trust a judge check.
  `tools/ncu_t.sh <bs> <cfg>`, `TW=<cfg> ncu ... python tools/sk_tw.py <bs>` (big-M kernels only).
- **The judge clones the prepared state** ("copies": 2): never keep scratch buffers or raw addresses
  (tensor maps!) in the state. bench.py test runs a cloned state now.

## What's on main (submission.py)
M <= 32 (decode, bs8): 2 launches, moe_tk2f (gemm13 rows then gemm2 rows in ONE persistent launch: one routing
prologue, the ring continues, so the producer streams w2 while gemm13 drains) + moe_norm2.
- s8: template <NC, NSTG, KPS, DYN>; KPS = k chunks per consumer warp per ring stage (stage = NC*KPS*1152 B).
  Launch cfg = (D+1)*1000 + KPS*100 + NSTG (tk2f_go dispatch in moe_tk2f_launch). Defaults: TK2F = TK2F_DYN = 204 (static, all M:
  the submitted code); DYN = cfg 5402 (judge: no gain, check b890d69b). KPS 2/4 cut per-stage overhead: H200 -3..4% at bs8-32 (check 74a049c0);
  locally the H100 is DRAM-bound (ncu dram 83%) and shows nothing: use smsp__inst_executed (tools/ncu_tk2f.sh) as proxy.
- DYN: CTA b takes static rows [b*S, (b+1)*S) of each phase (S = rows/grid - D), then single rows from the tail
  [grid*S, N) via cnt[1] (gemm13) / cnt[2] (gemm2) atomics, pulled by the producer one row ahead; row ids reach the
  consumers via s_rid[4] (written before the row's first full-barrier arrive). Phase-1 end = soft marker (-1, no
  stage), phase-2 end = an empty stage. Per-CTA phase ends spread ~10% of the phase (per-SM speed, not row counts):
  DYN fixes that on the H100 (bs8-32 -3..4%) but NOT on the H200 (consumer-bound: SMs already even). Row ids use
  arrive-only mbarriers ridb[4] (a phase-end wait on a data stage cost an HBM latency at bs1).
gemm13 consumers add their finished rows to cnt[0] (bar.sync 2 + thread-0 threadfence + atomicAdd); gemm2 consumers
spin (ld.acquire) until cnt[0] == n13, then load inter with __ldg. cnt = st["cnt"] (int32[128], zeroed in prepare,
re-zeroed by moe_norm2 after each call: no memset node). DO NOT go back to per-segment counters or ld.global.cg
inter loads without a many-routing sweep (tools/sweep_routings.py, also with +64 KB dyn smem): see NOTES s7 race.
TK2F = 0 = the old 3-launch path below (moe_tk2 x2 + norm2), kept as reference (tools/race_ab.py compares bitwise).
Old 3-launch decode: `moe_tk2<NRT, NC=8, NSTG=8, GATED>` x2 + `moe_norm2`:
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
M > 32 (TW_CFG = 418, TWF = 1): route -> moe_gather -> moe_twf (gemm13 AND gemm2, one persistent launch) -> moe_norm.
- moe_twf<RT=4, NSTG=8> (s6): 384 threads = 2 consumer WGs + producer warp (setmaxnreg 40/232), 1 CTA/SM. Work
  counter meta[66] hands out all gemm13 items, then all gemm2 items (heaviest experts first). Item = (expert,
  32-token pass (TWC::PS), row block of 8 warps x 4 16-row tiles; gated: 2 gate + 2 matching up tiles per warp).
  gemm13 item end: fence.proxy.async + bar.sync 1,256 + threadfence + atomicAdd(done[e] = meta[68+e]); gemm2 producer
  polls done[e] (ld.acquire) == npass*16 before its first TMA of inter. Deadlock-free (only running CTAs take
  items; gemm13 items never wait). meta is 100 ints; moe_route zeroes counters + flags.
- ptxas rule (C7512 "insufficient register resources"): it sizes the wgmma pipeline for the WHOLE function, so the
  largest variant decides: RT=4 needs N <= 32 (hence 32-token passes), and gated + non-gated loops in one function
  serialize everything -> one shared main loop (tw_item<..., DYN=true>), epilogue chosen at run time (tw_epi).
  Always check the SASS: `DEPBAR.LE gsb0, 0x1` in loops, no 0x0 except at item ends.
- Stage (1 k-chunk of 64) = TMA tensor-map copy of the x tile (box 64 k x {8,16,32} rows by nv, SW128) + bulk copy
  of the item's weights (prepare: _tb_pack(rt=4) -> w13b4/w2b4 [e][rb][kc][warp][RT][lane]) + 1 of scales.
  Consumers: dq() into registers = wgmma RS A fragment, wgmma m64nNk16 (N = 8*ceil(nv/8)), ring of 2, wait_group 1.
- Older big-M kernels still in the file: moe_tw (TWF = 0; TW_CFG 218 = RT 2 / 64-token passes; 1118 = 16 consumer
  warps x RT 1), moe_big (TW_CFG = 0), moe_tb.
- moe_gather: xg[pos] = x[sroutes[pos]>>3] with each 64-chunk permuted to the _pack_weight K order. xg / inter
  are per-call torch.empty; tensor maps are encoded inside moe_tw_launch from the passed buffer.
- dequant dq(): o0 = q&M, o1/o2 = rotl(q,10/13)&M, o3 = ((q&0x001B001B)*16648)&M, M=0x81C081C0; 4 x
  bf16x2 mul with LUT scale. 12 instr per 8 weights. (_POS in prepare must match.)
- Also in the file (not used by default): moe_sk (register-ring stream-K + atomics, SK_CFG path when
  TK2_NSTG=0), moe_tk (TMA + atomics), moe_gemm (old decode), route2/act. Clean up when convenient.

## Local harness
Tools in muse_moe_w4a16/tools/ (copies of the /tmp scripts): ncu_t.sh (H200 proxy), seeds.py (200 routings), cmp.py, moe_regs.sh,
moe_sum.py (judge JSON summary), s4: tw_t.py <TW_CFG...> (big-M correctness + local time bs128/256), seeds_big.py
(125 big-M routings incl. skewed), clone_t.py (cloned-state test), sk_tw.py (ncu driver, TW=<cfg> env).
Judge JSON summary: `python /tmp/moe_sum.py <okc --json output>` (recreate if lost: prints per-shape median vs baseline).
Per-kernel timing: /tmp/moe_k.py <bs> '[cfg,...]' (gemm13/gemm2 alone, cold L2); /tmp/moe_p.py <bs> <cfg> for ncu.
`muse_moe_w4a16/bench.py test|bench|sweep`. Local H100 big-M of main (moe_tw): bs128 244, bs256 304 us.
Routing at seed 7: bs1 3 experts x1 token; bs16 13 experts (max 4 tok); bs32 18 (max 7);
bs8 10 (max 2); bs128 25 (max 28); bs256 26 (max 49).

## Session 8 summary
- 2.131x -> 2.176x: tk2f stages of KPS = 2 k chunks per warp (NSTG 4, same 72 KB ring): fewer per-stage ops
  (-5% instructions), H200 bs8-32 -3..4%. KPS 4 (402): -6.5% instr but judge tie (2.157x). DYN row tail: H100 win,
  judge no gain. Tools: tools/ncu_tk2f.sh <bs> <cfg..> (duration, instructions, issue, dram), ab_decode.py VS=<cfgs>
  (sets TK2F and TK2F_DYN), /tmp/moe8/mkdbg2.py + td.py (per-CTA %globaltimer timeline; recreate from NOTES if lost).
- LESSON: the local H100 is DRAM-bound for decode (ncu dram 83%), the H200 is consumer-issue-bound: rank decode
  changes by smsp__inst_executed and stalls, not by H100 time; then a judge check.

## Session 7 summary
- 2.061x -> see Board: decode gemm13+gemm2 fused into one launch (moe_tk2f); found + worked around a race
  in it (NOTES s7). Tools added: sweep_routings.py, race_ab.py, ab_decode.py, readbound.py.

## Ranked ideas
0'. (s8) Cut consumer instructions further (that is what moved the H200): 16-token segments (idea 0 below, now
   on top of the KPS loop: tk2f_consume), fewer scale-LUT ops (4 LDS + extracts per kc), per-row reduce.
   bs1 (28 us vs 11.4 floor): 2-3 experts only, 2.1 TB/s locally; long_scoreboard stalls highest there: look at the
   routing prologue (1.8 us) and the per-row epilogue; norm2 ~4 us is a PDL secondary already.
0. (s7) 16-token segments in moe_tk2f (2 token groups share each dequantized A fragment, 2nd group's x in smem):
   draft tools/tk2f_gs16_draft.cu (166 regs, no spill), apply on top of 128b11a's wait/__ldg scheme; sweep-test
   at GS 16 with tools/sweep_routings.py. bs32 has 10% (up to 40%) extra-group rows: expect bs32 -5..8%.
   Root-cause the s7 race (ld.global.cg first-row-after-reload errors) if time allows: it may hide elsewhere.
1. Decode bs32 (1.49x) / bs16 (1.83x) are now the weakest. s6 SASS (tk2<512,8,8,true>): 130 instr per stage
   = 104 dq+mma + ~22 overhead (6 mbarrier wait/branch/sync, 4 LDS weights/scales + arrive, ~16 scale-LUT
   extract/add/LDS). NSTG 8 is special: = blocks per rp, so ptxas constant-folds the stage index (6/12 don't). H100 ncu bs32 gemm13: DRAM 84%, issue 51%, 18.8 instr per
   256 weights vs 13 for dq+mma (overhead: per-stage mbar/LDS/scale LUT, per-rp cross-warp reduce + named barrier).
   On H200 both GEMMs run at ~65% of HBM: consumer issue/latency limited. Ideas: cut per-stage overhead (scales
   for 2 stages per LDS, fewer LUT lookups), reduce per rp reduce cost (accumulate 2 rps before reducing), prefetch
   the next segment's x into registers (one global-latency bubble per segment change).
2. Shared dequant for experts with > 8 tokens in decode (judge bs32 routings have hot experts with 9-32 tokens;
   locally time tracks 8-token groups, not experts): 2 groups per pass reusing A fragments (x of group 2 in
   registers: 1 CTA/SM allows ~224 regs). Earlier 16-token try spilled (s3): keep NG=2 path separate.
3. Big-M: gemm2 tail (only 8 items per expert at RT=4; 200 items / 132 CTAs at bs128): give the last items 2-tile
   halves or let idle CTAs start moe_norm rows whose experts are done (per-token flags); fuse moe_gather into
   moe_route (multi-CTA route) -1 launch (~3 us).
4. Clean-up: remove moe_sk / moe_tk / moe_gemm / route2 / act / moe_tb / moe_big / RT 2 moe_tw (~55 s compile).

## Dead ends
- (s7) norm fused into moe_tk2f behind a grid barrier (worse, bs1 +3 us); L2 bulk prefetch of next rows in tk2f
  (much worse); memset node for counters (+2.6 us; keep counters in state, zeroed by norm2).
- (s6) moe_tw with 16 consumer warps x RT 1 (cfg 1118): 115M vs 78M instr, local +10%. Decode NSTG 6/12 (judge -5%).
- (s5) decode PDL (gemm2 secondary + routing recompute) and tk2_map blk 2 row order: judge worse (checks 52e5359f, 9acf8e84, 715085cb).
- (s5) tw2 L2 prefetch of first item before griddepcontrol.wait: no local gain.
- (s4) K-split tail for moe_tw (last `grid` items as 2 K halves, FP32 partials, 2nd finisher reduces): SM idle
  20% -> 13% in ncu, judge WORSE (check 2c41a133 1.972x: bs128 249 / bs256 292 vs 233/278). 4 parts: worse in ncu.
- (s4) moe_tb: TMA ring + mma.sync for big M (8 consumer warps, RT=4, x as 64 per-row 128-B bulk copies):
  local 299/402 vs 259/332 (consumer latency + tiny copies).
- (s4) moe_tw RT=4 / KG=2 / A ring of 4 at 384 threads: serialized wgmma (see above).
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
