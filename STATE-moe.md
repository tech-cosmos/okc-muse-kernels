# STATE-moe (muse_moe_w4a16)

## Board
- Best submit: 6a92803ae4fc4a11 @a670c82, **2.190x** (2026-10-26 02:16Z). Board is a TEST BED only (lead 01:45Z): the layer counts.
  Used 2026-10-26: checks 17 of 20 (s12: 5 DIAG checks, never submit those), submits 3. s11 checks: c1659443 2.179 (K-split tail: tie), 7dda218c 2.130 (RT 2 all:
  worse), d4909e24 2.170 (twm mixed: neutral). Per-shape best-ish: bs1 27-29, bs8 67-71, bs16 93-99, bs32 128-135,
  bs128 196-207, bs256 243-251 (judge noise +-4%).
- **H200 layer profiles**: ~/r-layer/muse_decode_layer/profiles/submission.<job>.txt = judge per-kernel times per shape.
  USE THEM (they show the experts inside the layer). moe_norm's time there = mostly twf's tail (PDL secondary).
- **H100 is NOT a proxy** for big-M changes (s11: RT 2 H100 -3..9 us, judge +8%) nor for decode (DRAM-bound locally).
  Decode: ncu x 0.72 / instruction counts; tools/l2warm.py = consumer-bound proxy (1 expert, warm L2).
- The judge clones the prepared state: never keep scratch buffers or raw addresses in the state.

## H200 DIAG tool (s12) -- use it for every decode/big-M A/B
`DIAG = 1` (only in a /tmp copy: `sed 's/^DIAG = 0$/DIAG = 1/' submission.py > /tmp/x/submission.py; okc check /tmp/x/submission.py`)
makes prepare() time DIAG_VARIANTS (python-global overrides) on the layer's embedded routings (5 seeds x d1/v1/d16/d32
+ v16/v32), cold L2, graph replay; output: grep "diag12 MEAN" <cwd>/profiles/submission.<job>.log. Repeatable +-0.5 us.
tools/mktl_diag.py = same for a per-CTA globaltimer timeline (diagtl lines). H200 fit (tk2f + norm2, layer routings):
t = 14.1 + 6.97 * experts + 4.4 * extra 8-token groups (us); 6.97 us/expert = 4.06 TB/s (85% of HBM peak).

## What's on main (submission.py)
s11: M > 32 now runs **moe_twm** (TWF_MIX = 1): moe_twf with the item shape chosen per expert (RT 2 / 64-token passes,
packing w*b2, for experts with > 32 tokens; RT 4 / 32-token passes otherwise; stage stride = RT 4 stage). prepare packs b2 too.
Options (off): TWF_SPLIT / TWF_SPLIT_M (K-split tail items, FP32 partial + ticket/flag in st["tk"], split starts at an
expert boundary), TWF_RT / TWF_RT_M (whole-call RT 2), tk2f NC 16 (cfg +20000). moe_route orders an expert's tokens
nondeterministically: any per-pass treatment difference makes outputs nondeterministic (see NOTES s11).
s10 (c33acae): RACE FIX in tk2f_consume: a consumer arrives on empty[stage] only AFTER the stage's dq/mma (was: right
after issuing the LDS -> the producer could overwrite the stage before the LDS read it; = the s7 race). Never move the
arrive back up. moe_tk / moe_tk2 / tk2d (DYN) still have the early arrive (not default paths): fix before using them.
GS16 option (TK2G=0 off): tk2g_row/tk2g_consume, cfg +10000 (+ digit w: weighted split tk2g_split), 2nd token group's x
in smem (+64 KB dyn), s_red x2. Correct (tools/gs_ab.py), slower (see Session 10).
s9: TK2SA = TWSA = 1: e4m3 block scales decoded arithmetically (sca<K>(w) = __byte_perm(w,0,0x4K4K)*16 + 0x77007700)
in tk2f_consume and tw_item (moe_twf) when prepare() found every scale positive normal (st["sa"]); else the smem LUT.
moe_twf's choice is a host flag (moe_set_sa, set in kernel_fn). TK2FN = 0: fused norm (moe_tk2fn_launch) kept, dead.
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

## Session 12 summary
- H200 diag tool + model (above). Measured on the H200, all on layer routings: GS16 (old, merged, any split weight)
  +4..8 us DEAD; merged-phase h8 = main; per-segment counters (PSC) = main; deeper rings (208/206/403/108/402) lose;
  DYN +3..10; NC16 +1..2; TWF_MIX 1 vs 0: v16 -3.3, v32 -4.4 us (keep 1; layer should take it).
- GS16's d1 loss was i-cache (4 unrolled row bodies): tk2h one-body consumer fixed that, the d16/d32 loss remained.
- Timeline: CTAs balanced (spread 1-3 us); phase 2 (w2) streams ~3.5 TB/s vs phase 1 ~4.0 TB/s; 3.6 us to first data.

## Sessions 8-11: see NOTES (s10 race fix + layer harness, s11 twm/K-split).

## Ranked ideas
-3. (s12) Phase 2 (gemm2/w2) of tk2f streams ~12% slower per byte than phase 1 on the H200 (timeline): ~3 us at d32,
   ~2 at d16. Find out why with the diag timeline (stamp per-row times in phase 2; try phase-2 rows = 2 rps per
   ring pass, or w2 tile pairs adjacent instead of 2048 rows apart (prepare repack), or x (inter) prefetch earlier).
   Also: first data at 3.6 us (routing 1.5 + 2.1 latency): let the producer warp do the routing alone (it only needs
   s_rs/s_e) and start bulk copies while the consumers build s_srt; or prefetch expert-0 weights of ALL experts? no.
-2. (s11, measured s12) TWF_MIX = 1 wins v16 -3.3, v32 -4.4 us on H200: make sure the layer re-copies (COORD).
-1. (s10) Re-check the layer copied the race fix (COORD Requests). Then: why are GS16 phase-2 rows slower (timeline tool in
   NOTES s10: globaltimer per CTA at phase ends)? If fixable, GS16 cuts 8-17% instructions on the layer's d16/d32 and halves
   DRAM when L2 cannot dedup. Suspects: 64 KB extra smem (L1 carveout: inter __ldg x loads), xs LDS before mbar_wait, s_red x2.
   Try: xs in the ring's slack / smaller (bf16 x of group 2 only for the warp's k chunks is already minimal), or GS16 in gemm13
   only (phase 1 tied) with GS8 segments in gemm2 (needs inter rows of 16-token segments readable as 2 x 8: they are).
-0. (s10) Cheaper dq is the only other lever (69%): ideas: fold the e4m3 scale into the x side per k16 block is impossible
   (per-row scale); 2-group sharing = GS16 above.
00. (s9) Instructions still rule the H200 decode: tk2f gemm13 row loop = 1391 SASS instr/warp/row vs 832 for dq+mma
   (before SA; SA took ~100 of the ~560 overhead). Next targets: ~137 address IMADs + 43 LEA (precompute smem
   addresses per stage: NSTG=4 -> unroll the ring so stage offsets are constants), per-row reduce/epilogue (accumulate
   2 rps before the cross-warp reduce), CS2R zeroing. Measure with tools/ncu_tk2f.sh (inst_executed), then judge.
   bs1 (27-29 us vs 11.4 floor): routing prologue 2 us + first stage 2.4 us + norm2 4.4 us are fixed costs.
0'. (s8) Cut consumer instructions further (that is what moved the H200): 16-token segments (idea 0 below, now
   on top of the KPS loop: tk2f_consume), fewer scale-LUT ops (4 LDS + extracts per kc), per-row reduce.
   bs1 (28 us vs 11.4 floor): 2-3 experts only, 2.1 TB/s locally; long_scoreboard stalls highest there: look at the
   routing prologue (1.8 us) and the per-row epilogue; norm2 ~4 us is a PDL secondary already.
0. (s7) 16-token segments in moe_tk2f (2 token groups share each dequantized A fragment, 2nd group's x in smem):
   draft tools/tk2f_gs16_draft.cu (166 regs, no spill), apply on top of 128b11a's wait/__ldg scheme; sweep-test
   at GS 16 with tools/sweep_routings.py. bs32 has 10% (up to 40%) extra-group rows: expect bs32 -5..8%.
   Root-cause the s7 race (ld.global.cg first-row-after-reload errors) if time allows: it may hide elsewhere.
3. Big-M: gemm2 tail (only 8 items per expert at RT=4; 200 items / 132 CTAs at bs128): give the last items 2-tile
   halves or let idle CTAs start moe_norm rows whose experts are done (per-token flags); fuse moe_gather into
   moe_route (multi-CTA route) -1 launch (~3 us).
4. Clean-up: remove moe_sk / moe_tk / moe_gemm / route2 / act / moe_tb / moe_big / RT 2 moe_tw (~55 s compile).

## Dead ends (older ones: NOTES-moe.md)
- (s12, H200 diag on layer routings) GS16 in any form (old, merged one-body tk2h, split weights 2-6): +4..8 us at d16/d32;
  per-segment gemm13->gemm2 counters (PSC): 0; deeper tk2f rings: +1..5; DYN: +3..10; NC16: +1..2. Weighted static
  split (TK2W): H100 worse, and the H200 timeline shows balanced CTAs (nothing to fix).
- (s10) GS16 16-token segments as default (judge bs32 135.7 vs 128.7; H100 layer d16/d32 +5..18 us); weighted row split for
  2-group rows (worse at every weight); ld.global.cg / asm nc loads for inter (no help: the race was the ring).
- (s9) Post-expert norm fused into tk2f (last-M-CTA tickets, one-pass register norm): judge bs1 +2 us (168fa04c).
  Also s7's grid-barrier version. A separate norm2 kernel is as cheap as any in-kernel tail on the H200.
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
- moe_stream (decode): old item structure + per-lane cp.async ring of weights AND x (no barriers, D=7 ahead
  across items), 1 CTA/SM: slower (bs32 178/115 vs 147/75 us): 2 warps/scheduler can't issue enough.
