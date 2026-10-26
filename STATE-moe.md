# STATE-moe (muse_moe_w4a16)

## Board
- Best submit: 6a92803ae4fc4a11 @a670c82, **2.190x** (2026-10-26 02:16Z). Board is a TEST BED only (lead 01:45Z): the layer counts.
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

## Night4 session 11 (22:06-22:30Z, no judge runs): DIAG dry run 107 s total (safe), tests PASS. READ the 00:00Z DIAG with
  `python muse_moe_w4a16/tools/n11/diag_read.py ~/r-moe/muse_moe_w4a16/profiles/submission.<job>.log` (or the cwd's profiles/).
  Decision rules: bal wins d16/v1 -> tell layer (TUNE bal); h16w*p4 wins d32 -> layer's tk2h TUNE value; l2h0/l2h1 beat gs8
  -> layer #define TK2F_L2H; h32 vs h24/h40, h32ne0, h32rg1 -> layer TUNE at T=256; gs8 - nm < ~4 us at d16/d32 -> stop
  tk2f consumer work for good. Post the per-load table under COORD Results/Requests for layer.

## Night4 session 10 (21:44-22:15Z, no judge runs) -- KPS 4 re-closed; DIAG er + nm (NOTES n10, r-moe 77fa9b6)
- H100: tk2f with NO consumer math (nm, PF & 256 probe) is only 1.0/3.1/3.7/2.8 us faster (d1/d16/d32/v1), and copy size
  does not matter without math (nm402 = nm204); stream "tkp4" gains came from its missing L2 hints. ER (early ring release,
  fence.proxy.async; cfg 94204/91204, bitwise) = tie. 00:00Z DIAG: read gs8 - nm (upper bound of ANY consumer-side gain).

## Night4 session 9 (21:27-22:15Z, no judge runs) -- H200 HBM-ceiling probe in the DIAG (NOTES n9)
- r-moe 804b9f5: moe_stream + _diag_stream: the 00:00Z DIAG check also prints `diag9s MEAN <load> <v>` (tkp = tk2f's
  copy pattern, uq = unique bytes, sq16..64 = contiguous single copies). H100: tk2f = its copy pattern's rate; 16 KB+
  copies ~6% faster than 8 KB. tk2f KPS 4 (84404/84403/404) DEAD (H100 +2..6 us, bitwise). Read diag9s vs diag12 gs8.

## Night4 session 8: BAL ported into the layer (tools/n8/patch_layer_bal.py); layer v97 runs it (TUNE bal at T=16/32).

## Night4 session 7 (20:50-21:50Z, no judge runs) -- tk2f phase transition (NOTES n7)
- G88 timeline: rows cost the same in both phases; the loss is the transition (wait for the slowest CTA 0.8-2.3 + first
  gemm2 row +0.5-0.8 proxy-us). ILV (rows strided over CTAs) DEAD: each CTA reloads 64 KB x per segment (+200 MB L2->SM).
- BAL (TK2F_DYN 98204, bitwise = main): per-segment counters + extra rows of the two phases on different CTAs. G88 proxy
  v1 -1.4, d16 -0.6..1.0, d32 0, d1 +1.4 proxy-us; H100 d32 +1.5..2.6. In the 00:00Z DIAG as bal / bal1.

## Night4 session 6 (20:18-21:00Z, no judge runs) -- r-moe 3710aa8 + 82d6781: tk2h PF4 + layer port (NOTES n6)
- tk2h (merged-phase consumer, GS8/GS16) now has PF4 (slot-ordered pre-scaled y32 -> moe_norm3, like main tk2f 84204):
  python TK2H_PF4 (cfg + 4) or TK2H_M = {M: W} (GS16 + PF4, weight W). Bitwise = main (tools/n6/h4_ab.py 0/1680).
- #define TK2H_P4 1 builds only the SA tk2h PF4 kernels (no TK2_EXP); non-EXP builds TORCH_CHECK any other tk2h cfg.
- tools/n6/patch_layer_h16.py ports it into a layer moe_src.py (layer s58 is wiring it as TUNE option tk2h at T=16/32).
- H100 in the layer harness: h16w3p4 LOSES d16 +4, d32 +2..7 us (DRAM-bound H100); G88 proxy d32 -6 proxy-us. Only the
  H200 decides: the 00:00Z DIAG has h8p4 / h16w2p4 / h16w3p4 / h16w4p4 (moe-only), the layer's tuner does the rest.

## Night4 s5 (1378d3e): GS16 H200 loss = row split; DIAG has h16w0/2/3/4 (tk2h); s1-2: NORM_EARLY + PDL .nc fix.

## Night4 session 3 (19:31-20:05Z, no judge runs) -- r-moe 1f3b71b
- TK2F_L2H_RT (python, None = #define 4): run-time L2 hint mode, cfg + 1000000 * (mode + 1). H100 mode 4 vs 0: d16 -5.1, d32 -4.3.
- The hints never ran on the H200 (moe checks predate s21; layer board run is v70). DIAG "dec" now = gs8 (4) / l2h0 / l2h1.
- FIRST CHECK AT 00:00Z (unchanged plan + this): `sed 's/^DIAG = 0$/DIAG = 1/' muse_moe_w4a16/submission.py >
  /tmp/x/submission.py; okc check /tmp/x/submission.py --json`; grep "diag12 MEAN" in the job log. Report to layer:
  hot G (h24/32/40), RG, NORM_EARLY (h32ne0/offne0) at v32/v16, and L2H mode (l2h0/l2h1 vs gs8) at d1/d16/d32/v1.
  n4s5: + h16w0/2/3/4 (GS16 at d16/d32) vs gs8; n7: + bal/bal1; n9: + "diag9s MEAN" (HBM-ceiling probe, read vs gs8);
  n10: + er / nm (gs8 - nm = max consumer-side gain on the H200; nm output is WRONG by design, ignore its maxdiff).

## Ranked ideas
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
