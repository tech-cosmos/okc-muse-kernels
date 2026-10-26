# STATE-proj (muse_bf16_proj) — short current picture; NOTES-proj.md is the long log

Updated 2026-10-26 ~21:45 UTC (s51: g19 loop-top 0.35 us/stage explained = runtime div/mod (tm) ~160 cyc/stage + NOW-mode producer turnaround; G19_TAG2 (bit-identical, precomputed ks/st/tr) queued in _diag19d as NOTM vs NOTM+TAG2; H100 full-kernel tie. s50: _diag19d = H200 g19 m32/m16 decomposition queued for the 00:00Z check; gemv23 BF16 wgmma at m16/m32 DEAD; H100 trace: ~0.35 us/stage of g19's consumer time is outside the decode/mma loop, unexplained. s49: 'fewer bytes at m=1' bounded: exponent entropy 2.54 bits -> 10.54-bit floor, no fixed-rate format < 12 bits; closed; lane request to lead. s48: g19 m32 header/fixup cost measured, FIX2/HPF dead; s47, sandbox okc-night4: PDL .nc-hoist SASS audit of proj kernels in layer v91 = no live hazard; unpool T=32 grid tie; checks 20/20 until 00:00Z; main unchanged).

## Board
- **Best: submit 3810db09edb0479e = 2480c53, 1.3110x** (s16: gemv14 qkv m8 without split-K; its check a9b13469 1.3161).
  Before: 71145e34 (a1f0af7) 1.307x (router3 at router m9-128).
- 12-bit header format is now bit-exact (human 23:30Z): header stores true/scale, kernels subtract 2^-15, round-trip
  checked in _compress (fallback per layer); `python bench.py esc` = adversarial escape test. Run it after any 12-bit change.
- Budget 10-26 used: checks 20/20 (s25 5abb71f1 1.3040: diag, DeepGEMM bf16 dead), scored 5/10 (cap 10). s23 check 2722e8c4 1.3108 (diag: g13 for o m1 loses on H200). s22 check ccf41026 1.3162 (diag: qkv m32 bmm slabs). s17 check 29a2b16b 1.3173 (diag: m=1 = bare stream). s16 checks fb2fe4ee 1.3071 (g22 diag), a9b13469 1.3161; submit 3810db09. s15 checks ce84da07 1.3001, 4e5fad5b 1.2951 (diag only). s13 check 96165c87 1.3091 (diag: gemv21 loses). s12 checks 2ef3aafd 1.2953, 88800611 1.2934 (diag only). s10 checks 9b09c536 1.3136, 60ddb481 1.3296, f1319777 1.3146, s11 6b50e844 1.3064 (diag only).

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
- Dead ends this session: TMA bulk x staging (slower than cp.async), shared float atomics, 4-block clusters,
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
  Bare TMA stream 50 MB = 13.7 us H200 / 18.3 H100.
- H100 lessons: consumers are not memory-bound on H200: no-W-load mode = 17 us at pooler m16; per stage ~6 TB/s
  equivalent, the losses are x arrival (~2 us before the first stage completes) + ~1 us per segment flush + tail.
  Dead: deeper rings, x first, interleaved layout, dynamic tail, per-SM calibration (not stable), ts 2, rpw 2, rg 16.

## Session 11: pr1/pr8 (layer T=1/T=8 pooler + FP32 router + routing, one launch each). Handoff posted (COORD proj s11
## 05:15Z), wiring dev/pr1_layer.patch. H100 layer: d1 -2.5/-2.7 us, v1 -1.1/-1.4 us. Await the layer's H200 check.
- Lesson: under saturated HBM each dependent memory round trip costs ~3-4 us: keep ticket/last-block chains OFF the
  streaming blocks (extra finisher block spinning on an acquire counter).


## Session 13: T<=8 ramps QG (csl beside qkv) + gemv21 (TMA-ring m=1) DEAD; g13 needs its whole register file. NOTES s13.
- Key facts: g13 needs its whole register file (60 x 1024) for speed; co-residency also needs the smem carveout attr.
  One global ticket counter cannot feed 132 CTAs per row (~0.5 us per round). mbarrier release-arrive waits for the
  same thread's pending atomics/stores.
- pr1.cu: x32/x after the wait now __ldcg (robust vs ptxas hoisting; the layer's built SASS was already safe).

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

## s19-s26 (NOTES): QSS (d32 qkv as 2 FP32 bmm slabs) in layer v70; G19_RAW dead; DeepGEMM bf16 dead (image: flashinfer
## 0.7.0, deep_gemm 0.2.1+muse, sgl_kernel 0.4.9); tinygemm2 port = proj tuner candidate at unpooler m8 (not judge-checked).
## Session 31: pr1 router-W L2 prefetch with evict_last = WIN at T=1 (H100 layer d1 ~-1.3/-1.1 us, bitwise same): handoff
## dev/s31/pr1_evl_layer.patch; all layer prefetch sites evict_last (dev/s31/pf_last.py) d1 -1.8/-1.7. evict_first on W streams null. NOTES s31.
## Session 32 (in layer v80+): evict_last on ALL 9 layer L2 prefetch sites + KV_PF (g13's prefetch.global.L2 of the d1 KV rows): layer
## v78 H100 d1 -2.0/-0.9 and -1.3/-1.4 us (2 runs), v1 noise, bitwise same at d1, tests/scored/replay clean. Handoff
## dev/s32/evl_all_layer.patch (supersedes s31's). Dead: O_PF/boundary hints, pr8/other prefetch.global.L2 hints. NOTES s32.
## Session 33: g13 (12-bit) for the layer's T=1 o as a PDL secondary: H100 d1 full ~-1 us, sliding bimodal 0..-1.9:
## handoff as TUNE option g13o (dev/s33/g13o_layer.patch on v82). After the wait a GEMV is ISSUE-bound (~2.3 us decode+FMA);
## pre-decoding before the wait works (g13p) but loading all W beside csl slows csl as much: dead. NOTES s33.
## Session 34: DEC4 (dev/s34/dec4.py source transform, on main in gemv19/14/13, pr1, pr8): 12-bit decode 4 instr per BF16
## pair instead of 5 ((s & M2) | C had two immediates = 2 LOP3s; constants now opaque registers, one 3-reg LOP3). Bit-identical.
## H200 proxy g19 -0.5..-1.6 us per call (m32 pool -1.36 unp -1.22); in the H100 layer a tie (g19 there is HBM-shared with
## _router_split). Handoff dev/s34/dec4_layer.patch (applies on r-layer f5fe1ec). ab_files has a ~1.5 us file-position bias
## at d16/d32 sliding: use A B B A. g19 m32 = consumer-bound (NOW 27.1 vs full 29.6 at G72); load-group size G: null. NOTES s34.
## Session 35: DEC3 (dev/s35/dec3.py layout + dec3_src.py transform; on main in ALL 12-bit kernels + packers via _dec3):
## same 12 bits re-arranged (code bit 0 in the byte, sign + c1..c3 in the word at shift-friendly bits; prmt injects 0x38):
## 3 instr per BF16 pair (dec8 11 vs 16). H200 proxy g19 B-A: m16 -0.4..-1.65, pool m32 -0.48 us; layer H100 tie, tests
## 13/13 + scored 24/24, T=1/8 bitwise same. Layer agent integrating it (g22 kept on old layout). Old decode: -DPJ_DEC4.
## s36-s40 (detail in NOTES): pr8 bulk L2 prefetch npf 212 after the wait = WIN (H100 layer v1 -1.6 us, in layer v88). pr8
## tail = random ~3 us block-finish spread; fixing it needs K split + combines, <= ~0.1% layer: deprioritized. s39 dev/s39/fill.py:
## a side HBM stream beside a latency chain is zero-sum if it queues its whole range at once; ONE 32 KB TMA in flight per SM
## streams at ~95% rate and dilates the chain 40-60% less (rule for any layer weight-overlap/side-stream idea). s40: throttling
## g19's own few-stage pre-wait prologue (npre 1-2) beside a memory-touching pred: dead (keep deep npre).
## s44: g19 m32 LONGRUN DEAD (proxy +0.3..+5 us, NOTES s44). s45: pooler T=16 npre 2 vs 3 in layer = tie (keep 3).
## s46: T=1 pooler W pre-fill into L2/SMEM beside csl/o/bnd (side-stream throttled TMA filler, dev/s46): DEAD, +2..6 us d1:
## csl's window already carries o's pre-wait W stream; a side kernel alone costs ~2-3 us; bnd window = pr1's prologue.
## s47: SASS audit (dev/s47_pdl_audit.py) of all proj kernels in layer v91: no producer output read before griddepcontrol.wait;
## only pr1's non-bnd x/x32 __ldg (layer copy) is fragile -> Request to layer. unpool T=32 G19_GRID 96-132 = tie (H100).
## s48: g19 m32 escape fixups ~0.5-1.2 us/call proxy, header-load stall ~0.2-0.5; pipelined fixups (G19_FIX2) and
## one-run-ahead header prefetch (G19_HPF) both DEAD (NOTES s48). Diag switches G19_NOHDR/NOFIXB in gemv19.cu (off).
## s50: _diag19d (print only) in submission: g19 pool m32/m16, unp m32 x {main, NOW, NOW+NOMMA, NOW+NOX, NOFIX} x G132/108
## after a 3 us PDL pred (+~1-2 min compile). gemv23 at m16/m32 (BF16, no decode) = W-stream-bound, ties/loses cuBLAS: DEAD.
## H100 G19_TR2 trace: per consumer stage ~0.35 us outside the atom loop (loop top tag/int math 0.23, arrive->got 0.13).
## s51: G19_TAG2/G19_NOTM/G19_CLK switches (off by default). NOW-mode H100: TAG2 -1.0 (m16) / -0.1..-0.4 (m32) us vs layer-like
## NOTM; full kernel tie on H100 (W-bound). Loop top 0.30 -> 0.14 kcyc/stage (clock64). NOTES s51.
## Next (00:00Z check, ranked):
## 0. In the 00:00Z check log: diag19d 'NOTM' vs 'NOTM TAG2' (H200): >= 0.3 us win at m16/m32 -> TAG2 patch for layer's
##    g19_src.cu: patch READY dev/s51/tag2_layer.patch (handoff posted 21:50Z; layer H100 tie, tests 13/13): post the H200
##    numbers under Handoffs/Requests. diag19 'o m16' g19 vs pick: if g19 wins now -> layer TUNE option at T=16 o.
## a. First proj check after 00:00Z = main (carries _diag19d): read 'diag19d' lines. main-NOW = W-stream share on H200;
##    NOW-NOMMA = decode+mma; NOMMA = skeleton. Copy into NOTES. Submit only if > 1.3110 (diag adds nothing to timing).
## b. (done s51: TAG2, see above)
## Next sandbox (ranked; sandbox okc-night3 expired ~19:10Z 10-26 with proj checks 20/20, scored 5/10)
1. proj's first check = main as is (DEC3 in g13/g14 m1/m8 picks; tg2 unpooler m8 candidate): read the 'tune' lines in the
   judge log (profiles/*.log); submit if > 1.3110. Layer's first check = its main v88 (carries DEC3, pr8 npf 212, g19 s15).
2. Ask layer for a fresh H200 idle budget from v88's profile; take the proj-owned rows (T<=8 GEMV ramps, T>=16 cuBLAS
   gaps without PDL triggers). (npre 2 vs 3: dead s45; T=1 pooler pre-fill beside csl/o/bnd: dead s46.)
   (s49) fewer bytes at m=1 CLOSED: floor 10.54 bits (exp entropy 2.54), fixed-rate < 12 bits escapes 3.5%+, VL 11.2 bits
   ~ -1.3 us d1 at best (dev/s49_entropy.py, NOTES s49). All proj items now <= ~0.3% layer: asked lead for a new lane.
3. Below the line: pr8 K-split tail balancing (~0.1%), 11-bit format (~0.2-0.3%), g22/g23/DeepGEMM at m128/256 (dead).
