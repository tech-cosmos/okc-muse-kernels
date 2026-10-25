# STATE-moe (muse_moe_w4a16)

## Board
- Best submit: 52169da1ba624f00 @fa39ed7, **2.061x** (2026-10-25 23:00Z; check cb18fe27 of the same code 2.061x).
  Previous: 4f0ae2c7 2.033x @a0531b2, 1561eff8 1.985x @c39a176. Top of board 2.55x (#10 2.37x).
  Used 2026-10-25: checks 18 of 20, submits 9 of 10 (lead cap 9: none left today).
  | load | bs | SGLang | ours | x | floor | s5 (c39a176) |
  |---|---|---|---|---|---|---|
  | decode 1 | 1 | 76.6 | 30.6 | 2.51 | 11.4 | 30.8 |
  | decode 16 | 16 | 188.7 | 103.4 | 1.83 | 52.5 | 99.7 |
  | decode 32 | 32 | 208.1 | 139.7 | 1.49 | 70.8 | 137.2 |
  | verify 1 | 8 | 142.7 | 72.9 | 1.96 | 37.3 | 77.0 |
  | verify 16 | 128 | 460.5 | 198.0 | 2.33 | 101.3 | 238.7 |
  | verify 32 | 256 | 623.0 | 252.2 | 2.47 | 108.9 | 280.5 |
  Decode shapes vary +-4% between identical-code checks (bs16 99.8-107.2): A/B decode changes against 2-3 checks.
- One expert = 18.9 MB w13 + 9.4 MB w2 (incl. scales).
- **H200 proxy**: ncu kernel durations x 0.72 works for DECODE (moe_tk2). For big-M it does NOT rank changes:
  moe_tw: ncu said -9% (judge -19%); K-split tail: ncu -1% (judge +5%). For big-M trust a judge check.
  `tools/ncu_t.sh <bs> <cfg>`, `TW=<cfg> ncu ... python tools/sk_tw.py <bs>` (big-M kernels only).
- **The judge clones the prepared state** ("copies": 2): never keep scratch buffers or raw addresses
  (tensor maps!) in the state. bench.py test runs a cloned state now.

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

## Session 6 summary
- 1.985x -> 2.061x: big-M RT=4 (32-token passes, a0531b2: 2.033x), then gemm13+gemm2 fused into one persistent
  launch (moe_twf, fa39ed7: 2.061x; bs128 239 -> 198, bs256 281 -> 252 us). Decode unchanged.
- Big-M is instruction-bound (ncu: ~170 instr per k-chunk per warp, 96 of them dequant; issue ~57%). More warps
  (NW=16 x RT=1) = more instructions = worse. Amortizing per-k-chunk overhead (RT=4) and killing the gemm13 tail
  (fusion) is what worked. ncu (seed 7) still shows SM active cycles spread min/max ~0.84 (gemm2 tail).
- Decode on the judge: tk2 ring NSTG 12 and 6 both ~5% worse than 8 on every decode shape. moe_twf for decode
  (TK2_MAX knob) locally +11% at bs32, +13% at bs16: not checked.

## Ranked ideas
1. Decode bs32 (1.49x) / bs16 (1.83x) are now the weakest. H100 ncu bs32 gemm13: DRAM 84%, issue 51%, 18.8 instr per
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
