# STATE-proj (muse_bf16_proj) — short current picture; NOTES-proj.md is the long log
Updated 2026-10-27 12:30 UTC (s83). Layer v129 (6621403) took its own nu8d fix. NEW tools/stream_fuzz.py (r-layer branch proj/fuzz
19b5ba1 + validate.py step 'fuzz'): torch.cuda._sleep injected at every side/hi-stream fork (side late / main late, eager + graph),
every TUNE set: unfixed v128 caught (12/204 eager, 10/156 graph); v129 all T 0/1492 eager, 0/1216 graph. Static audit: no other
cross-stream shared scratch. Next: run validate (incl. fuzz) on every new layer bundle. NOTES s83.
Updated 2026-10-27 12:10 UTC (s82). FOUND+FIXED a T=8 race in layer (v128 and earlier): TUNE {'pr1': False} + nu8d -> rmsnorm reads
side-stream local_ids before the join (14-20/60 wrong under GPU contention; H200 tuner never picks pr1 False: +1.7..2.6 us).
Fix + tools/nu8_race.py on r-layer branch proj/nu8fix 357859d; bundle proj/b82 d4b2aba = attn31 + fix: validate ALL OK,
opt 0/906 + replay 0/480 under a GPU hog. Lesson: run opt/replay under contention. NOTES s82.
Updated 2026-10-27 11:55 UTC (s81). Ported moe s31 attention deltas layer lacked (csl/vtk fast divides, nds pt L2 prefetch +
K-early) -> branch proj/attn31 c71dbb6: tests 13/13, attn_probe 29/29, H100 TIE (ABA d32 -1.9 was a middle-slot artefact of
tools/ab_files.py: reverses in BAB; always run both orders). Stackable, not check-worthy alone. NOTES s81.
Older session headers (s46-s80): see NOTES-proj.md (sessions are logged there in order).
## Board
- **Best: submit 1ff4721e87c844c2 = ca7e557, 1.3233x** (s56; its check ed67748e 1.3277). Before: 3810db09 (2480c53) 1.3110x (s16: gemv14 qkv m8 without split-K; its check a9b13469 1.3161).
  Before: 71145e34 (a1f0af7) 1.307x (router3 at router m9-128).
- 12-bit header format is now bit-exact (human 23:30Z): header stores true/scale, kernels subtract 2^-15, round-trip
  checked in _compress (fallback per layer); `python bench.py esc` = adversarial escape test. Run it after any 12-bit change.
- Budget 10-27 used (s72): checks 8/20, scored 1/10. (0cfca9f had broken cpp decls -> fixed c2b541b: grep prepare output for 'build failed'.) prepare() limit is 300 s: diags must stay small (s56 fail).
## Router H200 now (check a478daea, a1f0af7): m1 3.79, m8 4.72 (router2), m16 5.55, m32 6.36, m128 9.00 (router3),
## m256 11.18 (cuBLAS splitK GEMM 6.4 + splitKreduce 2.0 + hi+lo add 1.6). SGLang: 10.5/26.4/15.0/15.5/16.8/17.7.
## router3 (dev/router3.cu, sweep dev/r3.py): split-K mma GEMM, hi+lo into one accumulator, partials -> float4 atomics
## into out zeroed by a tiny PDL primary kernel. m256: red volume (2M floats ~4 us) + mma.sync compute: only ties cuBLAS
## (prepare() picks per GPU since a566ba9: ROUTER_R3_TUNE).
## Per-shape H200 (check 7ba50b0a, 21bec6b): m1 qkv 11.2 o 6.3 pool 15.8 unp 4.6 router 5.1; m8 qkv 13.3 o 6.8 pool 16.6
## unp 6.6 router 8.1; m16 qkv 13.6 pool 19.1 unp 17.9; m32 qkv 13.8 pool 18.9 unp 18.9 (m>=16 = cuBLASLt, uncompressed).
Floors: qkv 7.9, o 3.5, pooler 14.0, unpooler 1.75 (m<=8) / 14.1, router 1.8; m256: 9.8/4.5/17.4/17.4/2.7.
SGLang H200 us: qkv ~14 (m<=32) 17 (m128) 20 (m256); o ~7/7.2/9.2; pooler ~21.5/24.8/30.7; unpooler 7 (m<=8) 18.7/20/24.
## Code on main (submission.py is GENERATED: edit dev/sub_template.py, dev/*.cu, dev/lt_cfg.txt; run `python dev/build_sub.py`)
- Router: m<=8 true-FP32 dev/router2.cu (block = 2 whole rows, x in regs, no split K; dev/r2.py sweeps); m 9-32 hi/lo mma dev/gemv7.cu; m>32 one cuBLAS GEMM vs stacked [hi; lo].
- BF16 m=1: tuner picks gemv13 (compressed; qkv, pooler) or gemv10 (o, unpooler-1k).
- BF16 m=8: tuner incl. gemv14 (compressed; wins qkv, pooler on H200), m16 pooler also gemv15 (not picked on H200).
- BF16 m>=8 (8/16/32/128/256): **prepare() `_tune`** times, per (layer, m), as CUDA-graph replays after a 256 MB read
  flush (cold L2, like the judge), F.linear vs gemv12 (BF16_G12: m8 all, pooler m16) vs 5 cuBLASLt configs
  (BF16_LT from dev/lt_cfg.txt), keeps the fastest in st["pick"]; kernel_fn dispatches via `_run`. Prints "tune ..."
  lines (visible in the judge log: profiles/*.log). On the H200 lt wins every shape except unpooler m32 (F.linear).
- dev/lt.cu: cuBLASLt matmul with explicit algo config (algo 66 = sm90 kernel; tile, stages 35=auto, split-K,
  reduction, swizzle, custom option, cluster shape). Separate load_inline (`-lcublasLt`): if it fails, F.linear.
- Sweep tooling: dev/lte_lib.py (load_inline helpers: lt_enum = all 45,792 algo-66 configs; lt_desc decodes a blob),
  dev/lte.py `python dev/lte.py 16:qkv_gate 256:pooler ...` (screen all configs, 2 reps eager w/ flush, refine top 40
  in graphs; saves /tmp/pj/lte_<m>_<layer>.pt), dev/lt_gencfg.py (-> lt_cfg.txt, top 5 distinct per shape).
  Run with SUB=<abs path to submission.py> and cwd /tmp/pj (scripts expect /tmp/pj/lt.py: copy dev/lte_lib.py there).
- bench.py: `python bench.py test` (39 cases), `python bench.py bench [filter]`.
## What I learned (numbers in NOTES)
- A graph's kernel launch costs ~1.8-2.6 us over an empty graph (more blocks -> more). Reader slope 3.24 TB/s (H100).
- Read pattern: only high-occupancy grid-stride or block-contiguous (threads interleaved at 16 B) reads approach
  peak; per-warp contiguous streams spread over the buffer are ~10-15% slower.
- x staging traffic = blocks * (m*k_chunk*2): keep blocks fat (1024 threads) and few. Issue x BEFORE W (x requests
  queued behind a 17 MB W prologue took 5-8 us).
- float atomicAdd on smem = CAS loop. Cluster size 4 at 1 block/SM: only 30 clusters fit (2 waves for 32).
- H200 gains exceed H100 gains for these kernels (H100 +1% -> H200 +6% at qkv m8). Don't drop marginal H100 wins.
- gemv12 m8 qkv (S=4): end phase (slot sum, stcg partials, fence, counter, re-read) ~2.8 us; warps finish with a
  ~1.8 us spread inside a block. H200 gemv12 at m16: qkv 0.99, o 0.91, pooler 1.08, unpooler 0.95; m32 0.72-0.93.
  R=2 row groups per warp (spills), gemv9 warp stream-K, CUDA-core math at m>=8, cuBLAS swap-AB.
## Lossless 12-bit weights (session 5): the big lever
- m=1 was already AT a bare reader's speed: only fewer bytes help. _compress in dev/sub_template.py: sign|mantissa byte
  + 4-bit exponent code (window 16 below max exp), escapes (~1/row) fixed from a per-row header. 25% fewer bytes.
- gemv13 (m=1, CUDA cores) and gemv14 (m=8, gemv12 + decode) are in; tuner picks per shape. Not yet: m16/m32 (needs a
  better kernel than gemv12's structure), m128/m256 (cuBLASLt; would need own wgmma + decode), router (FP32: exponent
  compression gives 28/32 bits only).
## gemv19 (session 10): the layer's T=16/32 GEMMs (lane since 10-26 01:45Z). Handoff posted (COORD proj s10).
- dev/gemv19.cu, layer interface dev/g19_lib.py (pack19 from _compress; build; g19_gemm; g19_cvt), test dev/t19lib.py,
  sweep dev/g19.py "M,KC,AP,NS,IS,TAILPB[,XB,TS,RG,RPW]" (env MODE=2 PRED=3000 = after a 3 us PDL predecessor, the
  layer case; ESCW=1 adversarial escapes x40; DEFS=G19_NOW (no W loads: consumer-only), G19_TRACE + STAGES=1 timelines).
- Design: TMA ring (one bulk copy per stage), FP32 out [m, n] zeroed IN the kernel (finisher warps; epoch-counter grid
  barrier before the first red, counter never reset), split-K partials by red.global.add.v4 (subnormal sums -> 0), PDL:
  producer streams npre W stages BEFORE griddepcontrol.wait, then x. One launch, no reduce kernel.
- The win is the PDL overlap (W streams during the predecessor). Standalone (judge board) it only ties cuBLAS.
- H200 (judge diag, us, after a 3 us PDL predecessor; pick = best of cuBLAS/lt/own): pooler m16 AP8 (1024,8,3,xb1)
  15.65 vs 19.0; qkv m16 (1024,4,4) 11.2 vs 13.2; unp m16 tie; o m16 loses (+0.8); all m32 lose (+1.5..5).
- H100 lessons: consumers are not memory-bound on H200: no-W-load mode = 17 us at pooler m16; per stage ~6 TB/s
  equivalent, the losses are x arrival (~2 us before the first stage completes) + ~1 us per segment flush + tail.
  Dead: deeper rings, x first, interleaved layout, dynamic tail, per-SM calibration (not stable), ts 2, rpw 2, rg 16.
## Session 11: pr1/pr8 (layer T=1/T=8 pooler + FP32 router + routing, one launch each). Handoff posted (COORD proj s11
## 05:15Z), wiring dev/pr1_layer.patch. H100 layer: d1 -2.5/-2.7 us, v1 -1.1/-1.4 us. Await the layer's H200 check.
- Lesson: under saturated HBM each dependent memory round trip costs ~3-4 us: keep ticket/last-block chains OFF the
  streaming blocks (extra finisher block spinning on an acquire counter).
## Session 14: gemv19 run end = the escape fixup loop (~1.06 us per run end, serialized shfl/branch chain) -> ballot
## version (bit-identical; H100 -0.5..0.8 us per g19 call). Handoff posted, patch dev/g19_ballot_layer.patch (NOTES s14).
- g19 at m32 is NOT smem-bound (RPW2 no gain); per stage decode ~0.4 + mma/ldsm ~0.4 + base ~0.5 us (H100 consumer-only).
## Session 15: g19 escape fixups on the stored partials (warp walks escapes, lane = token): H100 -0.15..1.0 us per call;
## layer patch dev/g19_s15_layer.patch (d16 -1.8/-4.6 us H100; with g19 at T=32 d32 -5.3/-6.3). H200 m32 now tie/-0.6.
- gemv19 mode 3 (BF16 out, tile tickets, self-cleaning ws, tm stage order) for the layer's qkv first op: H200 loses
  3-6 us standalone (g19 needs a PDL predecessor): DEAD for qkv. Dead too: finisher-side escape list, per-lane RMW.
## Session 16: gemv22 (m128/256 12-bit wgmma GEMM) DEAD (per-SM smem ingest of W + x box); qkv m8 g14 no split-K WIN
## (-1.2 us H200, handoff); lt picks handoff (pooler@256 etc.; the layer found pooler@256 loses beside the router).
- Dead too (NOTES s16): g22 at m16/32 (H100 consumer-only no better than g19), g22 cluster x multicast, gemv13 cp.async
  ring / L2 prefetch, gemv13 for o, fused [pooler; router hi/lo] g19 on 132 blocks (H100 26.2 vs 23.3 us pooler alone;
  routing then on the critical path), g19 test_wait spin. g19 m32 H200 bound = consumers (H100 NOW: skeleton 9.8 +
  decode 2.3 + mma.sync 4.4 us at pool m32) and no variant tried so far helps.
- **H200 proxy for g19** (NOTES s16): H100 `GRID=88 MODE=2 PRED=3000 python dev/g19.py ...`, times x 0.73 = judge g19
  after-pred numbers for pooler/unpool (qkv off by ~3 us); GRID 72 ~ the layer's 108 blocks. Use it before H200 diags.
## bitwise same; in layer v80+, dev/s32/evl_all_layer.patch). Dead: evict_first on W streams, O_PF/boundary/pr8 hints. NOTES s31/s32.
## s33: g13o (12-bit T=1 o, PDL secondary) handoff as TUNE option g13o (dev/s33/g13o_layer.patch on v82). After the wait a GEMV is ISSUE-bound (~2.3 us decode+FMA);
## Session 35: DEC3 (dev/s35/dec3.py layout + dec3_src.py transform; on main in ALL 12-bit kernels + packers via _dec3):
## same 12 bits re-arranged (code bit 0 in the byte, sign + c1..c3 in the word at shift-friendly bits; prmt injects 0x38):
## 3 instr per BF16 pair (dec8 11 vs 16). H200 proxy g19 B-A: m16 -0.4..-1.65, pool m32 -0.48 us; layer H100 tie, tests
## 13/13 + scored 24/24, T=1/8 bitwise same. Layer agent integrating it (g22 kept on old layout). Old decode: -DPJ_DEC4.
## s19-s52 dead ends + details: NOTES. s36-s40: pr8 bulk L2 prefetch npf 212 after the wait = WIN (H100 layer v1 -1.6 us, in layer v88). pr8
## tail = random ~3 us block-finish spread; fixing it needs K split + combines, <= ~0.1% layer: deprioritized. s39 dev/s39/fill.py:
## a side HBM stream beside a latency chain is zero-sum if it queues its whole range at once; ONE 32 KB TMA in flight per SM
## streams at ~95% rate and dilates the chain 40-60% less (rule for any layer weight-overlap/side-stream idea). s40: throttling
## g19's own few-stage pre-wait prologue (npre 1-2) beside a memory-touching pred: dead (keep deep npre).
## s46: T=1 pooler W pre-fill into L2/SMEM beside csl/o/bnd (side-stream throttled TMA filler, dev/s46): DEAD, +2..6 us d1:
## csl's window already carries o's pre-wait W stream; a side kernel alone costs ~2-3 us; bnd window = pr1's prologue.
## s50: _diag19d (print only) in submission: g19 pool m32/m16, unp m32 x {main, NOW, NOW+NOMMA, NOW+NOX, NOFIX} x G132/108
## after a 3 us PDL pred (+~1-2 min compile). gemv23 at m16/m32 (BF16, no decode) = W-stream-bound, ties/loses cuBLAS: DEAD.
## H100 G19_TR2 trace: per consumer stage ~0.35 us outside the atom loop (loop top tag/int math 0.23, arrive->got 0.13).
## s51: G19_TAG2/G19_NOTM/G19_CLK switches (off by default). NOW-mode H100: TAG2 -1.0 (m16) / -0.1..-0.4 (m32) us vs layer-like
## NOTM; full kernel tie on H100 (W-bound). Loop top 0.30 -> 0.14 kcyc/stage (clock64). NOTES s51.
## s54: g24 SK (r | 256): per-warp equal stage ranges, flush at task boundaries (fixes coarse task quantization; G24_TRACE
## shows m32 static warps doing 2 vs 3 tasks). H200 proxy: SK wins m32 + qkv m16, loses o and m16 pool/unp (G132); at G108
## SK fixes static (16,1024,2,4,16). DYN smem chunk claims DEAD (+4 us). NOTES s54.
## s55: gemv24 tail balancing DEAD (dynamic tail pools gemm_dt dev/s55, +1..5 us proxy; finished-warp L2 lend-prefetch PF
## +0.5..1 us). Per-warp end spread = HBM share; static SK/r1 stay. Also dead: m32 KC1024, x copies over 32 lanes, lazy
## out-zero fence (LZ/LZ2 +0.4..2). Upper bounds (illegal probes): x arrival 0.4-1.4 us, zero+fence 0.4-1.7 us at m32. NOTES s55.
## s56: g24 EZ/ZN zero-ahead chain at T=16 (each g24 zeroes the next g24's out; consumer skips zero+fence+epoch barrier):
## H100 layer d16 sl -0.3..-1.7, full tie; T=32 loses (+2..4, off). Handoff dev/s56/g24z_layer.patch. dev/s56/diag_read.py
## = readout for the 00:00Z log. NOTES s56. 00:00Z check 636c89fc FAILED setup: prepare() 325.6 s > 300 s (diag24 SK twins):
## KEEP DIAGS SMALL (prepare limit 300 s). Diag read: H200 pooler T=16 (4,3,12) -3.0 us vs layer cfg -> handoff
## dev/s56/g24cfg_h200_layer.patch (H100 d16 -0.9/-0.4); TAG2 dead; pooler T=32 SK (4,3,12) H200 -1.15 but H100 +1.2..1.7.
## s57: gemv27 (dev/s57/gemv27.cu, g27.py): block owns whole row groups, warps split k, smem sum, plain stores (no zero/
## barrier/atomics); PD = W decoded into registers before griddepcontrol.wait. H200 o m16 5.25 vs g24 6.43 (layer default),
## m32 loses to cuBLAS. Handoff dev/s57/g27_layer.patch (T=16 o). Left: x arrival ~1.8 us/2.6 us (m16/m32) = bytes from L2
## with all 128 blocks reading the same x: try cluster TMA multicast of x (g27 and g24), and PD for qkv/pool T=16 slices.
## s58: G27_MC 1 = x as bulk copies spread over threads (layer d16 -1.8/-1.05); cluster multicast (MC2/4) DEAD for PD.
## s59: G24X (dev/s59/g24x_layer.patch): first-op qkv g24 = setup round trips behind the W stream; EZ across calls -1.3/-1.9 us standalone.
## s60: wgmma WG mode (dev/s60/gemv24.cu, RR bit 9) correct but DEAD; nvcc drops a 2nd wgmma accumulator's sum unless
## it is one local array summed explicitly + operand fences (NOTES s60). g24 R=2 at m32 dead too.
## s61: g24 m32: ncu NOW 30% issue / 22% tensor (latency-bound, loop already well scheduled: G24_G dead); XSP/DF/WPB16 dead on H200.
## Next (ranked):
00. LANE (b) (lead 08:55Z): read COORD Requests for layer bundles; port + validate on proj/<name> branches (tools/validate.py on proj/validate).
0c. Layer: watch for s68 ltalt:pooler at T=128 (Handoffs proj s68) in a layer check's 'layer tune T=128' lines.
0a. Layer: watch for G24P (Handoffs proj s63) in a layer check; read 'layer tune T=32' {'g24p': True} vs default (sl/full).
0. Layer: watch for G24X (Handoffs proj s59) in a layer check; read the 'layer tune T=16' {'g24x': False} entry.
1. Layer: watch for g27 (Handoffs proj s57/s58) in a layer check; read its "layer tune T=16" g27:o line.
2. g27-style (no cross-block split-K, PD) for other T=16/32 calls is limited by registers (PD 127 regs x 512 thr): look at
   2 CTAs/SM g27 with fewer regs that keeps x reads per SM low (x is read by every block: +4 MB = +3 us at m16).
2b. m32 consumer: decode-overlap / split-role ideas are moot (s62: no-decode BF16 rows cost the same consumer time).
3. PD (pre-wait decode into registers) for other T=16 g24 calls where a warp slice fits (qkv/pool need split-K: g24 PD).
3b. (s64) g28 ping-pong variant (MMA WGs decode their own B tiles, no cross-WG handoff per stage, KA>=2, BN 256 at 384 thr):
   only if a NOW skeleton shows < ~0.1 us/stage on the H100 first (NOTES s64).
3d. (s70) T>=16 cuBLAS no-PDL gaps: nvjet cannot be a PDL secondary; only an own GDC-enabled CUTLASS GEMM could (<=1 us/GEMM).
3c. (s65) Dead too: g27 PD at T=8 for o. Remaining v1 item: none cheap (pr8 tail balancing ~0.1-0.2%, NOTES s37/38).
4. Below the line: pr8 K-split tail, 11-bit, g22/g23/DeepGEMM m128/256, hybrid BF16 rows (s62): dead.
