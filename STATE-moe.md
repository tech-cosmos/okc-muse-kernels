# STATE-moe

## LANE since 2026-10-27 09:45Z (lead LANE DECISION 2): muse_attention_decode as a problem (dir muse_attention_decode/)
Own budget: 20 checks + 10 scored/day on muse_attention_decode. Targets: batch-1 sliding (csl d1, vtk v1) and verify sliding
first; layer owns the full kernels (_gd_full d16/d32, _gw_verify v16/v32): don't work on those. Winners -> COORD Handoffs
(layer ports into r-layer attn_src.py). Precision: FP8 KV, Q >= E4M3, FP32 partials. ~/attn STATE.md dead-ends index = 126
sessions of history: read it before any idea. Board: 2.5962x (1450.7 us, submit 957cabd7 s33; was 2.566x); #1 pjmullin3 3.032x (1242.3 us).
Per-us value (overall attention score): d1 sl 1.94 %/us, v1 sl 1.48, d1 full 0.67, v1 full 0.51, d16 sl 0.44, v16 sl 0.35,
d32 sl 0.24, v32 sl 0.2. (Score = geomean over 6 loads; load = 46 x sliding + 16 x full.)
H200 per shape (us), 06456b7e (= seed 410dcb6 code, 2.570x): d1 4.34/12.25, d16 7.68/87.03, d32 12.29/167.67,
v1 5.49/16.60, v16 11.45/104.89, v32 15.51/194.99 (sliding/full). Judge records: median_ns = raw - ~4.28 us replay cost.
Tools (run from muse_attention_decode/): tools/bench.py test|bench; tools/jsum.py A.json B.json (per-shape deltas of okc
--json outputs); tools/s31/chk_sc.py (scored sliding shapes vs reference, 3 seeds); tools/s31/csl_ab.py A.py B.py (paired
H100 A/B of the csl/vtk kernels from two files); tools/s31/cslstamp.py [B] (per-CTA clock64 timeline of csl, VAR=... variants).

## Board
- Best submit: febfcb1f @72d1274 (SCB), **2.2469x** (2026-10-27 09:20Z; check ed55a5c0 2.2635x). Was 6a92803a 2.190x. Board is a TEST BED only (lead 01:45Z): the layer counts.
  Checks 20/20 used on 10-26 (DIAGs since s12: never submit those); budget resets 00:00Z. Per-shape best-ish: bs1 27-29, bs8 67-71, bs16 93-99, bs32 128-135,
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
s27 (40de385): SCB: scp<P>() decodes 2 e4m3 scales per PRMT+IMAD, blo/bhi broadcasts fold into HMUL2 .H0_H0/.H1_H1 (free) at
every SA scale site. Bitwise = before. H200 diag 5ad98078: v16/v32 twm -5, d16/d32 -1 us. Layer port: tools/s27m/patch_layer_scb.py.
s21: TK2F_L2H = 4 / TWM_L2H = 1: L2 cache hints on tk2f/twm weight copies (compile-time #defines substituted in _lib()).
(Older options/kernels still in the file -- GS16, DYN, TK2F 0, moe_tw/moe_big/moe_tb/moe_sk/moe_tk: see NOTES s3-s10.)
s11: M > 32 now runs **moe_twm** (TWF_MIX = 1): moe_twf with the item shape chosen per expert (RT 2 / 64-token passes,
packing w*b2, for experts with > 32 tokens; RT 4 / 32-token passes otherwise; stage stride = RT 4 stage). prepare packs b2 too.
Options (off): TWF_SPLIT / TWF_SPLIT_M (K-split tail items, FP32 partial + ticket/flag in st["tk"], split starts at an
expert boundary), TWF_RT / TWF_RT_M (whole-call RT 2), tk2f NC 16 (cfg +20000). moe_route orders an expert's tokens
nondeterministically: any per-pass treatment difference makes outputs nondeterministic (see NOTES s11).
s10 (c33acae): RACE FIX in tk2f_consume: a consumer arrives on empty[stage] only AFTER the stage's dq/mma (was: right
after issuing the LDS -> the producer could overwrite the stage before the LDS read it; = the s7 race). Never move the
arrive back up. moe_tk / moe_tk2 / tk2d (DYN) still have the early arrive (not default paths): fix before using them.
s9: TK2SA = TWSA = 1: e4m3 block scales decoded arithmetically (sca<K>(w) = __byte_perm(w,0,0x4K4K)*16 + 0x77007700)
in tk2f_consume and tw_item (moe_twf) when prepare() found every scale positive normal (st["sa"]); else the smem LUT.
moe_twf's choice is a host flag (moe_set_sa, set in kernel_fn). TK2FN = 0: fused norm (moe_tk2fn_launch) kept, dead.
M <= 32 (decode, bs8): 2 launches, moe_tk2f (gemm13 rows then gemm2 rows in ONE persistent launch: one routing
prologue, the ring continues, so the producer streams w2 while gemm13 drains) + moe_norm2.
  Launch cfg = (D+1)*1000 + KPS*100 + NSTG (tk2f_go dispatch in moe_tk2f_launch). Defaults: TK2F = TK2F_DYN = 204 (static, all M:
  the submitted code); DYN = cfg 5402 (judge: no gain, check b890d69b). KPS 2/4 cut per-stage overhead: H200 -3..4% at bs8-32 (check 74a049c0);
  locally the H100 is DRAM-bound (ncu dram 83%) and shows nothing: use smsp__inst_executed (tools/ncu_tk2f.sh) as proxy.
gemm13 consumers add their finished rows to cnt[0] (bar.sync 2 + thread-0 threadfence + atomicAdd); gemm2 consumers
spin (ld.acquire) until cnt[0] == n13, then load inter with __ldg. cnt = st["cnt"] (int32[128], zeroed in prepare,
re-zeroed by moe_norm2 after each call: no memset node). DO NOT go back to per-segment counters or ld.global.cg
inter loads without a many-routing sweep (tools/sweep_routings.py, also with +64 KB dyn smem): see NOTES s7 race.
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
- moe_gather: xg[pos] = x[sroutes[pos]>>3] with each 64-chunk permuted to the _pack_weight K order. xg / inter
  are per-call torch.empty; tensor maps are encoded inside moe_tw_launch from the passed buffer.
- dequant dq(): o0 = q&M, o1/o2 = rotl(q,10/13)&M, o3 = ((q&0x001B001B)*16648)&M, M=0x81C081C0; 4 x
  bf16x2 mul with LUT scale. 12 instr per 8 weights. (_POS in prepare must match.)

## Local harness
Tools in muse_moe_w4a16/tools/ (copies of the /tmp scripts): ncu_t.sh (H200 proxy), seeds.py (200 routings), cmp.py, moe_regs.sh,
moe_sum.py (judge JSON summary), s4: tw_t.py <TW_CFG...> (big-M correctness + local time bs128/256), seeds_big.py
(125 big-M routings incl. skewed), clone_t.py (cloned-state test), sk_tw.py (ncu driver, TW=<cfg> env).
Judge JSON summary: `python /tmp/moe_sum.py <okc --json output>` (recreate if lost: prints per-shape median vs baseline).
Per-kernel timing: /tmp/moe_k.py <bs> '[cfg,...]' (gemm13/gemm2 alone, cold L2); /tmp/moe_p.py <bs> <cfg> for ncu.
`muse_moe_w4a16/bench.py test|bench|sweep`. Local H100 big-M of main (moe_tw): bs128 244, bs256 304 us.
Routing at seed 7: bs1 3 experts x1 token; bs16 13 experts (max 4 tok); bs32 18 (max 7);
bs8 10 (max 2); bs128 25 (max 28); bs256 26 (max 49).

## Attention: what's on main (r-moe muse_attention_decode/submission.py @ s31 code) vs checked
- = 410dcb6 + agent2 halfpfkr (v32 sl cl, csl/nds pt prefetch, nds K-early) + verify 16 on the pre-cl nvs build (_NVS0) +
  csl/vtk owner fast divides. SUBMITTED s32: 90b7ce85 2.5653x (tie with board 2.566x). The same code timed 2.619 / 2.639 /
  2.589x inside s32's DIAG checks (node noise +-3% whole-run): board best stays 2.566x (06456b7e).
- Checks used 2026-10-27: 9/20 (s32: 5 DIAGs, s33: 2 DIAGs). Scored 2/10 (s33 submit: see NOTES).
## Attention: H200 batch-1 full facts (s32 DIAG tool tools/s32/mkdiag.py; NOTES s32)
- Floors (pt trip + 20.5 MB random 128-B rows, all in flight, incl. launch): ~8.0 us d1 and v1; dense 7.4.
- Main: d1 full 12.2-13.0, v1 full 17.0-17.3. With ALL gathers in L2: d1 11.0, v1 16.7 -> NOT memory-bound.
- Every plan knob loses (stages, splits 63-1023, BN, warps 8, pre/pf/fix, qk16, Gluon gw 127-263 +-gfuse, gbal).
- Next: H200 per-program timeline of _attn_split (v1) / _gd_full (d1) via the diag hook; build only once it names the cost.
## Attention: v1 full (s33, H100 timeline tools/s33/mktl.py; NOTES s33)
- Loop = flat 0.94 us per 64-key tile, same all-L2; per-SM saturation (1/2/4 programs per SM: 0.66/0.92/~1.67 us/tile);
  q phase 2.0, FIX tile 1.1, merge tail 2.0 us. H100 = valid proxy here (matches s32 H200 diag). peel/mif dead. CLOSED.
- s33 ON MAIN: spec=True in the v1-full plan: H200 DIAG -0.63/-0.7 us (4/4 pairs, checks 4fe17fa9, 28658e59). d1 spec stays (tie).
## Attention: ranked ideas / bounds (s31)
- csl (d1 sl) post-gather tail is ~1.5 us cold H100: compute 0.45, premerge+cluster 0.2, push/slowest-CTA 0.37, owner 0.33.
  The slowest CTA's 12 random gathers per thread set the push time; owner merge is now ~0.33. No >0.2 us lever seen in the
  csl structure (many tried in ~/attn s72-s126). Untested: fewer distinct row loads per thread (wider per-lane rows) to cut
  the max-of-12-latencies tail; 2 query heads per mma (rows 8..15 of the m16 tile are zero today).
- v16 sl: merge launch (_combine_hb ~2 us) is the gap; 4-CTA clusters don't fit at 1 CTA/SM (30 < 32), 2-CTA S 2 loses.
  An nvs at 2 CTAs/SM (smaller ring) would let 4-CTA clusters fit (max 62): unexplored.

## moe sessions n3-n30: see NOTES-moe.md (moe lane closed by lead 2026-10-27 09:45Z, LANE DECISION 2).

## Ranked ideas
-9. DEAD (s26 re-read of s19): warp-specialized twm consumer (dq WGs -> BF16 A in smem -> SS wgmma WG) was already
   microbenchmarked in s19 (tools/s19/mb_ws.py): full ~= T + M (no overlap even across warps), SS slower than RS (smem
   bandwidth: 64 KB A written + read per RT4 stage). s22 pingpong: also nothing. Do not rebuild it.
   Also: mma.sync for N8 items (s18 microbench -7%) buys nothing on the H200: light N8 items already run at the per-SM HBM
   share (s23); N16 only -3%. Big-M levers left = the drain (~20 us/call, TWF_SPLIT_M halves already a layer TUNE option).
-8. (s17) Hot experts (> 64 tokens at v32: 2 RT2 passes, ~35% of v32 CTA time for ~17% of bytes): a pass that serves
   all their tokens needs N 96-128 accumulators = a SEPARATE kernel (ptxas sizes registers for the whole function).
   E.g. hot experts on their own launch on a subset of SMs concurrently with twm/twq for the rest (2 streams or one
   kernel with blockIdx-split roles: the dependency counters already work across CTAs). Est. v32 -20 H200 us.
   s18: such a kernel could also take finer N (40/48: hot-pass stage -24/-14%) that twm cannot (register limit).
-7. (s17/s18) tk2f fixed cost: weights-before-wait is DONE by layer (v62/63 TK2FX/TK2FJ); per-chunk gemm2 readiness
   is likely ~0 (s12 per-segment counters gave 0 on the H200). d1 is at ~85% of HBM (2-3 experts, 20 us).
-5. (s14) Drain of twm is still ~20-30 us (items ~40 us). Half-size tail items (TWM_TAIL) clash with the merge order;
   try tail class only for the LAST B items after the merge window, or K-split of the last wave. And: fewer dq per token
   for hot experts (128-token passes need N 128: registers).
-4. (s13) Consumer dq is ~73% of instrs and both int pipes are half rate: only fewer dq ops per weight (sharing a dequant
   across >8 tokens = GS16, unexplained H200 loss) or a cheaper decode can move d16/d32 much. Phase-2 12% per-row gap on
   the H200 is NOT the segment change / counter spin (s13 diag). Untested: phase-2 timeline with no extra groups
   (synthetic 10 experts x 8 tokens) to tell L2-dedup effects from per-row cost.
-3. (s12) Phase 2 (gemm2/w2) of tk2f streams ~12% slower per byte than phase 1 on the H200 (timeline): ~3 us at d32,
   ~2 at d16. Find out why with the diag timeline (stamp per-row times in phase 2; try phase-2 rows = 2 rps per
   ring pass, or w2 tile pairs adjacent instead of 2048 rows apart (prepare repack), or x (inter) prefetch earlier).
   Also: first data at 3.6 us (routing 1.5 + 2.1 latency): let the producer warp do the routing alone (it only needs
   s_rs/s_e) and start bulk copies while the consumers build s_srt; or prefetch expert-0 weights of ALL experts? no.
-1. (s10) Re-check the layer copied the race fix (COORD Requests). Then: why are GS16 phase-2 rows slower (timeline tool in
   NOTES s10: globaltimer per CTA at phase ends)? If fixable, GS16 cuts 8-17% instructions on the layer's d16/d32 and halves
   DRAM when L2 cannot dedup. Suspects: 64 KB extra smem (L1 carveout: inter __ldg x loads), xs LDS before mbar_wait, s_red x2.
   Try: xs in the ring's slack / smaller (bf16 x of group 2 only for the warp's k chunks is already minimal), or GS16 in gemm13
   only (phase 1 tied) with GS8 segments in gemm2 (needs inter rows of 16-token segments readable as 2 x 8: they are).
3. Big-M: let idle tail CTAs start moe_norm rows whose experts are done (per-token flags); fuse moe_gather into
   moe_route (multi-CTA route) -1 launch (~3 us).

## Dead ends (older ones: NOTES-moe.md)
- (s26) L2 prefetch of the unpool W from twm (spread over its main phase, evict_last): v32 +5..10 H100 us, unpool -1.
- (s27) RSPLIT: weighted tk2f gemm13 split favouring CTAs that start early during pr1: +1.4..10 us (any imbalance costs).
- (s14) tw_item next-stage weight/scale LDS prefetch: twm v32 +8, twf +9.5 us (H100). TWM_TAIL13; TWM_TAIL + merge order.
- (s12, H200 diag on layer routings; REOPENED n4s5: its weighted split was slow + count split imbalanced) GS16 in any form (old, merged one-body tk2h, split weights 2-6): +4..8 us at d16/d32;
  per-segment gemm13->gemm2 counters (PSC): 0; deeper tk2f rings: +1..5; DYN: +3..10; NC16: +1..2. Weighted static
  split (TK2W): H100 worse, and the H200 timeline shows balanced CTAs (nothing to fix).
- (s10) GS16 16-token segments as default (judge bs32 135.7 vs 128.7; H100 layer d16/d32 +5..18 us); weighted row split for
  2-group rows (worse at every weight); ld.global.cg / asm nc loads for inter (no help: the race was the ring).
- (s9) Post-expert norm fused into tk2f (last-M-CTA tickets, one-pass register norm): judge bs1 +2 us (168fa04c).
  Also s7's grid-barrier version. A separate norm2 kernel is as cheap as any in-kernel tail on the H200.
- (s7) norm fused into moe_tk2f behind a grid barrier (worse, bs1 +3 us); L2 bulk prefetch of next rows in tk2f
  (much worse); memset node for counters (+2.6 us; keep counters in state, zeroed by norm2).
- (s6) moe_tw with 16 consumer warps x RT 1 (cfg 1118): 115M vs 78M instr, local +10%. Decode NSTG 6/12 (judge -5%).
