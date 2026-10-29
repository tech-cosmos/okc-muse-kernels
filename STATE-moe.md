# STATE-moe

## NOW (n6s6 end, 2026-10-29 ~01:40Z): main b39e4d2 (= 7d2a079 kernels; TK2F_REL 0); layer branch moe/layer-twm 8309328
- n6s6 TK2F_REL (tk2f/tk2h gemm13 publishes by a helper warp; tools/n6s6/patch_rel.py ports it to any copy): judge DIAG
  8c5f1376: tk2h (layer T=32) d32 -0.75 us every seed; tk2f 1-group paths d1/v1 +0.25..0.3, d16 tie. moe default 0; layer
  branch has TK2F_REL 2 (tk2h only), validate ALL OK. ~0.1% layer score: not worth a check on its own.
- Layer branch moe/layer-twm = v163 + twm EPI/KSC2/REL (check 74dcb2bb 2.1569x vs board 2.1580, v16 -6.6/-6.8 us) + TK2F_REL 2.
  Waiting on the lead (COORD Results n6s5): submit under the 09:28Z rule / merge into layer main? We are #49 overall, #50/#51
  at 2.158/2.157: one noise draw away from dropping out of the top 50.
- tk2f release lever: CLOSED (n6s6). Remaining moe items each <= ~2 us/call: (1) twm N16 gemm13 epilogue 2.2-3.8 us/item
  (N24 duplicated gated body, store drain); (2) moe_hot REL. Tools new: tools/n6s6/ab_dec.py (paired H100 decode A/B,
  bitwise), mkdiag_trel.py (decode-only DIAG, 2 libs: judge prepare 225 s), sass_tk2f.py, rundiag.py.
- GIT: worktree /tmp/n6s5/wt is on moe/layer-twm (upstream set to origin/moe/layer-twm in n6s6; before that `git pull
  --rebase` there rebased onto main: never pull in it without checking upstream).

## WAS-n6s4 (n6s4 end, 2026-10-29 ~01:00Z): main 7d2a079 = TWM_KSC_DUAL 1 + TWM_REL 1 (bitwise, 15/15), H200-verified small wins
- Judge H200 DIAGs on layer routings: dual (twm without K-split code for calls with no split) v16 -1.9 / v32 tie; REL
  (helper warp 9 releases gemm13+gemm2 items) v16 -0.5 / v32 -1.4 us per call, every seed. Modal board: bs128 -1.7, bs256 -2.5.
  Plain check a4054bf3 2.2790x = tie with board 2.2792x (decode drew slow) -> not submitted. Handoff posted for the layer.
- H200 twm timeline (a18cfaed): epilogue 6-11 us per CTA per call (was 9-14), gemm13 N8 item end 1.4-2.2 us, N16 2.2-3.8.
- LAYOUT RULE: after ANY twm/tk2f code change run `python tools/n5s17/loop64k.py <k.o> 'moe_twm|moe_hot'` (sass.py builds
  k.o under /tmp/n6s4/sass_<hash>): an MMA loop across 64 KB = +4..12% (n5s17). NM (one tw_item per RT) DEAD.
- DIAG copies: tools/n6s4/mkdiag_ksc.py / mkdiag_rel.py pattern (plain library, big-M routings only: prepare ~216 s).
  NEVER run 2+ DIAG checks at once (34547d56 hit 323 s > 300 s cap while 3 compiled concurrently).
- NEXT (ranked): (1) tk2f release analog (bar.sync 2 + thread-0 fence at each segment end stalls the next row's bar.sync 1;
  no idle warp in tk2f: defer the non-final publishes / merge into the CTA-end publish when !BAL), H200 DIAG on d1/d16/d32/v1;
  (2) remaining twm epilogue (N16 gemm13 2.2-3.8 us/item: N24's duplicated gated body, store drain); (3) moe_hot same REL.
- Board stays 2.2792x (aeba42c5). Checks used 10-29 (after n6s6): moe 5 of 20 (4 DIAG + 1 plain), layer 1; submits 0. Modal 9/60 min.
- Tools: tools/n6s3/ab2.py A.py B.py (paired H100 A/B, board bs128/256 + layer v16/v32, max|A-B|); tools/n6s4/sass.py <file>;
  /tmp/n6s4/rundiag.py <diag copy> (local run of a DIAG prepare). The judge REJECTS `import reference`.

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
- Best submit: **aeba42c5 @a60019d (TWM_EPI), 2.2792x** (21:30Z 10-28; check fc7e86d6 2.3127x). Was 4fb734c6 @eb8c03c 2.2703x. Was 951c17e0 @21deb8a (bs256 hot32+split66), 2.2535x (2026-10-28 05:45Z; check cefcc133 2.2632x). Was febfcb1f @72d1274 (SCB) 2.2469x. Was 6a92803a 2.190x. Board is a TEST BED only (lead 01:45Z): the layer counts.
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
- s36: csl SASS audit: independent work already hoisted under the gathers; csl closed. No open >= 0.1% attention item.
- s40 H200 DIAG d5bf959a (tools/s40/mkdiag.py): random-row gather ceiling ~4.1 TB/s (d32 floor 159, d16 79; dense 144/72).
  Decode full at 92-95% of it (headroom d32<=9, d16<=7 us). Verify full consumer-bound (all-L2 v32 170 / v16 93 vs main 188/102,
  floors 157/80): headroom only via a faster _gw_verify consumer. Full kernels at perfect overlap = ~-4% attention score max.
- s42/s43 (NOTES): T=128/256 GEMM prototypes xr3 / g31 / g32 / g33 all DEAD vs layer g30 / cuBLAS.
- s45 (NOTES s45): re-audit, no untested lever on any attention shape; lane request to lead (COORD). 0 checks.
## Attention: H200 batch-1 full facts (s32 DIAG tool tools/s32/mkdiag.py; NOTES s32)
- Floors (pt trip + 20.5 MB random 128-B rows, all in flight, incl. launch): ~8.0 us d1 and v1; dense 7.4.
- Main: d1 full 12.2-13.0, v1 full 17.0-17.3. With ALL gathers in L2: d1 11.0, v1 16.7 -> NOT memory-bound.
- Every plan knob loses (stages, splits 63-1023, BN, warps 8, pre/pf/fix, qk16, Gluon gw 127-263 +-gfuse, gbal).
- Next: H200 per-program timeline of _attn_split (v1) / _gd_full (d1) via the diag hook; build only once it names the cost.
## Attention: d1 full (s35; NOTES s35)
- ON MAIN 9855088: plan qf=True (query rows loaded before _gd_full's prologue copies): H200 -0.28 us (5/5 pairs, DIAGs
  95f8158e, 1259baad), H100 tie. Not yet in a scored run (rides with the next submit). Handoff to layer posted.
- H100 timeline (tools/s35/mktl_d1.py + run_d1.py): q 1.1, first tile +2.8, 1.17/tile (gather rate), red 0.45, _red_fin
  1.5 after the last program. H200: deeper stages, splits 640/1023/255, qf+640 all lose/tie. Unbuilt: 2-warp programs
  splitting each tile (per-warp chain 0.86 us/tile).
## Dead ends (older ones: NOTES-moe.md)
- (n5s8) pr32 at T=8 (g30n pooler + FP32 router FMA + routing, one launch) instead of pr8: H100 in-layer +1.8..4.3 us (pr8 already
  ~22 us after bnd in-layer); H200 L2W proxy 17.25 vs pr8 ~17: no win. PR_IL (router stages interleaved) worse.
- (n20, H200 13b528a8) LOSSLESS SCALE COMPRESSION (S6 6-bit codes, base 80; 5-bit windows would need escapes): loses at d16/d32
  (+1..2.7 us), wins only v1 -0.5..0.8: extra decode ops on a latency-bound consumer. Weight codes are uniform (4.0 bits): no gain there.
- (s26) L2 prefetch of the unpool W from twm (spread over its main phase, evict_last): v32 +5..10 H100 us, unpool -1.
- (s27) RSPLIT: weighted tk2f gemm13 split favouring CTAs that start early during pr1: +1.4..10 us (any imbalance costs).
- (s12, H200 diag on layer routings; REOPENED n4s5: its weighted split was slow + count split imbalanced) GS16 in any form (old, merged one-body tk2h, split weights 2-6): +4..8 us at d16/d32;
  per-segment gemm13->gemm2 counters (PSC): 0; deeper tk2f rings: +1..5; DYN: +3..10; NC16: +1..2. Weighted static
  split (TK2W): H100 worse, and the H200 timeline shows balanced CTAs (nothing to fix).
- (s10) GS16 16-token segments as default (judge bs32 135.7 vs 128.7; H100 layer d16/d32 +5..18 us); weighted row split for
  2-group rows (worse at every weight); ld.global.cg / asm nc loads for inter (no help: the race was the ring).
- (s9) Post-expert norm fused into tk2f (last-M-CTA tickets, one-pass register norm): judge bs1 +2 us (168fa04c).
  Also s7's grid-barrier version. A separate norm2 kernel is as cheap as any in-kernel tail on the H200.
