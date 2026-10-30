# FRESH.md: first-principles budget of muse_decode_layer (agent fresh, session f1s1, 2026-10-29 ~19:30Z)

Built from TASK.md/reference.py, the judge json/txt profiles (r-layer/profiles, 11 runs of v153-v168), the H200 timeline
dev/n7s1_tl_h200.txt, and the routing census in the notes (distinct local experts). No code was changed.
T: d1 = 1, d16 = 16, d32 = 32, v1 = 8, v16 = 128, v32 = 256 tokens.

## (a) Bytes per load (what has to come from HBM; FLOPs only matter at v16/v32)
- Dense BF16 weights: qkv 37.7 MB, o 16.8, pooler 67.1, unpool 67.1 (T<=8: an 8.4 MB slice), router 8.4 (FP32).
  With the lossless 12-bit format already in use: x0.75 except the router -> T<=8: 105.9 MB, T>8: 149.9 MB.
- Expert: 28.3 MB each (w13 18.9 + w2 9.4, FP4 + E4M3 scales). Distinct local experts (notes census, judge-like):
  d1 ~2.5, d16 ~8, d32 ~10, v1 ~7.5, v16 ~18, v32 ~19.5. (The judge's "sol" column counts ~0.75x of these.)
- KV: 256 B per token per request: sliding 0.5 MB/request, full 20.5 MB/request (d32 full 655 MB).
- FLOPs that bite: v32 experts ~450 local routes x 100.6 MFLOP = 45 GFLOP (+ N padding) and dense 48 GFLOP; v32 full
  attention 84 GFLOP + 328 M exp. Everything at T<=32 is pure HBM.

## (b)(c)(d) Per load: byte time at 4.3 TB/s (realistic H200 stream) vs ours (v168 a5b4ddab, judge us)
| load | bytes MB (dense+KV+exp) | bytes us | ours us | ours/bytes | excess us | 2.42x team if uniform (-13%) |
|---|---|---|---|---|---|---|
| d1 sl  | 106+1+71    | 41.2  | 67.3  | 1.63 | 26  | 59.6  |
| d1 fu  | 106+20+71   | 45.9  | 76.6  | 1.67 | 31  | 67.8  |
| d16 sl | 150+8+226   | 89.5  | 129.4 | 1.45 | 40  | 114.5 |
| d16 fu | 150+328+226 | 163.7 | 208.5 | 1.27 | 45  | 184.5 |
| d32 sl | 150+17+283  | 104.6 | 154.3 | 1.48 | 50  | 136.5 |
| d32 fu | 150+655+283 | 253.1 | 297.8 | 1.18 | 45  | 263.5 (only 10 us above bytes) |
| v1 sl  | 106+1+212   | 74.1  | 109.6 | 1.48 | 35  | 97.0  |
| v1 fu  | 106+20+212  | 78.8  | 123.3 | 1.57 | 45  | 109.1 |
| v16 sl | 150+8+510   | 155.3 | 267.6 | 1.72 | 112 | 236.8 |
| v16 fu | 150+328+510 | 229.6 | 347.2 | 1.51 | 118 | 307.3 |
| v32 sl | 150+17+552  | 167.2 | 330.4 | 1.98 | 163 | 292.4 |
| v32 fu | 150+655+552 | 315.7 | 503.8 | 1.60 | 188 | 445.8 |
A uniform -13% is impossible at d32 full, so #1's lead must be uneven: big where our excess is big (v16/v32, d1/v1).

Where the excess sits (H200 timeline n7s1, judge profile a5b4ddab):
- d1 sl (~69 us in-graph): qkv 11.8 (bytes 6.6) | attn+o tail to 20.8 (o bytes 2.9) | bnd 4.2 | pr1 pooler+router ~16 (13.7)
  | tk2f 26 span / 21 GPU (16.5) | n2u+bnd tail ~6. ~26 us excess, spread over ~6 op boundaries (H100 hbm_fold: 5 bubbles
  of 2-8 us where HBM is < 50% busy, 26 of 85 us).
- d16/d32 sl: experts already stream at ~4.33 TB/s (moe n7s41: 6.54 us/expert marginal). The ~40-50 us excess is all in the
  non-expert chain: qkv -> attn -> o -> bnd ~36 us for 58 MB (13.5 us of bytes) at d32 sl, pooler||router -> routing ->
  rmsnorm -> tk2f start, and the tail norm2 -> unpool -> bnd (~21 us for 11.7 us of bytes).
- v16/v32: the expert phase itself (twm 160 / 210 us GPU vs ~118 / 128 us of bytes) is ~40 / ~80 us of the excess:
  consumer-bound (dq ALU + wgmma serialize per SM; REDESIGN s3/s6c).
- full layers: attention streams 328/655 MB; decode full ~3.8-4.0 TB/s, verify full _gw_verify 3.3-3.45 TB/s (issue-bound).

## Hypotheses: how a 2.42x team is faster (ranked by expected score)
H1 (all six loads, mostly d1/v1/d16/d32): THE CHAIN. ~10 grid-wide dependency hops per layer, each paid as drain + sync +
   ramp (~2-5 us of HBM idle; moe n7s22: tk2f alone loses 2.3 us prologue + 1.5 us first-data latency). A top team runs the
   layer (or the pre-expert part + tail) as a persistent kernel: all CTAs resident, every CTA pre-issues its NEXT phase's
   weight slice into its smem ring (data-independent: o, pooler, router, unpool; gemm2 once routing is known) before it waits on
   the phase counter, so each hop costs a flag latency (~0.5-1 us) instead of drain+launch+ramp, and HBM never idles.
   Bound: excess at d-loads is 26-50 us/layer; recovering half at d1/v1/d16/d32 ~= +5-7% geomean.
   ORACLE O1 (synthetic, exact bytes, H200): a T=1 and a T=32 chain of the layer's streaming phases with real byte counts
   (qkv 28 MB -> [2 us attention stand-in] -> o 12.6 -> norm -> pooler+router 58.7 -> top-8 -> 2.5 experts' w13 -> w2 ->
   unpool -> norm), as (A) separate PDL kernels in a graph, as today, and (B) one persistent kernel with per-phase counters and
   next-phase smem prefill before each wait. Reading: B vs bytes/4.3 TB/s, and A vs our real layer (A must reproduce ~our
   time, or the stand-in is wrong). Kill: B saves < 1 us per hop over A on the H200.
H2 (v16/v32): big-M experts at stream rate. If twm ran at bytes: v16 load -15%, v32 -22% => ~+7% geomean by itself, the
   largest single item, and it matches the moe board #1 being 18% ahead of us. ORACLE O2 (in-layer, H200, exists in pieces):
   twm with (i) dq free, (ii) wgmma free, (iii) both free (pure stream) at v16/v32. (iii) - real = the prize; (i), (ii) say
   which side to cut. Kill: (iii) within ~15 us of real (then it is ramp/drain, i.e. H1).
H3 (v16/v32 full): verify-full attention at 3.3-3.45 TB/s vs ~4.3: ~23/38 us/layer, ~+1.3% geomean. Lowest; attention is
   ~200 sessions deep (~/attn) and issue-bound there. ORACLE O3: _gw_verify with the softmax/PV math cut (KV stream only).
Not a hypothesis we can act on: routing near-tie slack (dropping up to 15% + 2 distinct experts) would explain a lot of a
   gap at d1/v32, but the human ruled it out (checker slack, not speed).

## Step 4: the record against each hypothesis (read critically)
H1, record: lead 06:00Z "HBM-never-idle" big bet; proj n7s4 gate FAIL; layer n7s4/s5 hbm_fold bubble map; IDF/early tk2f
   (+3 us loss, H100); T=1 o+bnd+pr1 megakernel with last-block/ticket epilogues (+1.4..2.8, s8/s14/s74/OB8); XPF/WAVE (moe n7s23).
   Flaws I can name:
   * proj n7s4 made the piece's BYTES/MATH free but kept every kernel, launch and wait. So -2..-4 us when free says the
     d16/d32 pre-expert excess is NOT bytes. That is evidence FOR a chain/latency cost, not against a persistent design. The
     lead's reading ("big bet fails") tested the wrong half of the idea.
   * The o+bnd+pr1 megakernel attempts synced by last-block tickets (serial: the last CTA does the epilogue, then everyone
     reloads). That's not the design in H1, where every CTA's next-phase ring is already full when the counter flips.
   * IDF/early-tk2f and the bubble map were measured on the H100 (3.35 TB/s): there the chain is bandwidth-bound and filling a
     bubble is zero-sum. On the H200 the same latencies are a larger share. H100-only verdicts on latency items are suspect.
   * Still valid: moe n7s23 XPF/WAVE (H200): inside tk2f the gemm13 -> gemm2 transition is not idle HBM. So H1's gain is at
     the hops BETWEEN kernels, not inside tk2f.
H2, record: dq at its z3 floor (7 int + 4 HMUL2 / 8 w), overlap closed by ncu dispatch stalls (n7s47/48), R1 gate idle
   8-16% only (n7s1), Humming/CUTLASS both slower than ours. I can't name a flaw from reading alone. O2 tells whether there is
   a prize at all; the mechanism stays the moe board #1's secret. Not my first lane.
H3, record: ~/attn s-index (poly exp2, f16x2 ex2, FP8 PV, 6 WGs: all dead; issue-bound ~45%). No flaw named. Dropped.

## Pick for session 2
O1 (H1). Write tools/chain_oracle.cu (+ a .py driver under the GPU lock): H100 first for correctness and shape, then one
Modal H200 run (~3-5 GPU-min). If B beats A by >= 1 us/hop on the H200 at T=1 and T=32, post the table under Requests and
propose the build order: tail (norm2 -> unpool -> bnd, data-independent unpool W) and qkv -> attn -> o -> bnd first, since the
expert phase already streams well.
O2 on Modal when minutes allow (it reuses existing DIAG switches, if moe's oracle flags still exist in moe_src.py).

## O1 first run (f1s1, H100 only, tools/fresh/chain_oracle.cu; PRELIMINARY: the synthetic is not yet a floor)
Synthetic T=1 chain, 191.4 MB (qkv 28.3 | attn 0.5 on 32 CTAs | o 12.6 | router 8.4 -> routing (CTA 0) | pooler 50.3 |
3 experts' w13 56.6 -> w2 28.3 | unpool 6.3 | final norm), BF16 rows + CUDA-core dots, cold L2 (1 GB read), graph replay,
median of 30-50; bytes/3.0 TB/s = 63.8 us on the H100 (44.5 at 4.3 on the H200). Includes a ~5 us memset+graph floor.
| variant | with math | NOMATH (consumer frees stages at once) |
|---|---|---|
| B persistent, producer warp streams across phase boundaries | 161.1 | 121.1 |
| A one kernel per phase, PDL, ring prefill before the wait  | 174.5 | 120.4 |
| A one kernel per phase, plain stream order                  | 202.8 | 136.7 |
Per-phase stamps (B, NOMATH, CTA 0): dependency wait after the CTA's own end 1.7-4.8 us per hop; building x (the previous
phase's output, <= 12K floats, L2-resident) AFTER the wait 1-4 us, and up to 9-17 us when deep weight prefetch is in flight
(qkv's cold input 8.9, pooler 17.4); weights then stream at ~2.6-3.1 TB/s.
Reading so far: (1) the kernel boundary is not the cost: PDL kernels that prefill their ring before waiting = persistent
(120 vs 121 NOMATH); (2) the cost per hop is the dependency wait + the activation fetch after it, ~3-8 us per hop x 8 hops,
larger than the drain; (3) this synthetic still spends ~55 us over bytes, so it does NOT yet say what a good hop costs.
Next (f1s2): a 2-phase microbench that isolates one hop: counter flip -> 132 CTAs see it -> 32 KB x fetched from L2 ->
first FMA, with and without a concurrent bulk stream, H100 then H200 (Modal after 00:00Z). If a hop can't get below ~2 us
under load, H1's prize at d1 is ~(hops x (today's ~4 - 2)) ~= 8-10 us/layer, and the design is "fewer hops" (fusing
norm/routing redundantly into consumers, as the synthetic already does for the boundaries) rather than "persistent".
Caveat to resolve: x-build 17 us in the pooler phase is unexplained (contention with the prefetch stream or a code issue).

## O1 follow-up (f1s1, H100): one hop isolated (tools/fresh/hop.cu) + chain at shallow rings
hop.cu: 132 persistent CTAs x 64 hops; per hop each CTA writes its slice of an x vector, bumps a counter, spins, then loads
all of x (32 KB) from L2. Optional producer warp per CTA keeps a 16 KB bulk-copy stream to smem going (depth = stages/SM).
| bytes in flight | stream TB/s | last arrival -> all CTAs see counter | -> all CTAs hold 32 KB x |
|---|---|---|---|
| 0 (idle) | - | 0.54 | 2.02 |
| 2.2 MB | 2.58 | 0.51 | 2.14 |
| 4.3 MB | 3.10 | 0.74 | 3.30 |
| 6.5 MB | 3.15 | 1.15 | 4.83 |
| 8.7 MB | 3.14 | 1.34 | 6.24 |
| 13.0 MB | 3.14 | 1.38 | 9.12 |
| 17.3 MB | 3.13 | 1.47 | 8.6 (p90 14.1) |
=> HBM saturates at ~4.3 MB in flight on the H100. Anything deeper buys no bandwidth and adds ~0.7 us of hop latency per
extra MB (bufferbloat). The record already has this: proj s39 (dev/s39/fill.py, "dilation is set by bytes in flight") and
s40 (throttling g19's own PDL prologue: dead on the H100), s46 (pooler pre-fill: dead). Not new, and never measured on the H200.
Chain oracle again with the fixed code (ring depth NSTG 2/3/4/10 = 4/6/8/21 MB): B 155.5/155.6/157.0/159.8, A-PDL
168.3/170.8/172.9/173.8; NOMATH B ~121-123 and A-PDL ~119-121 at every depth. Per hop at NSTG 2: dependency wait 2-3 us
(drain spread + 0.5 counter) + x build 2-5 us. My synthetic (123 us NOMATH) is SLOWER than our real d1 layer on the H100
(~85-88 us), so it can't bound the real layer. It only shows that the kernel boundary itself isn't the cost.
VERDICT for H1 as "persistent megakernel": not supported on the H100 (persistent = PDL + prefill). The hop cost is real
(~3-8 us), but its parts (drain spread, queueing behind bytes in flight, the activation fetch) are the same in either design.
Open, cheap: the same two tools on the H200 (knee in MB, hop cost) before closing H1 for good: ~2 GPU-min.

## O1 on the H200 (Modal, 0.3 GPU-min, same binaries): H1 CLOSED
hop.cu: in flight 0 / 2.2 / 4.3 / 6.5 / 8.7 / 13.0 / 17.3 MB -> stream - / 3.00 / 4.31 / 4.57 / 4.61 / 4.63 / 4.63 TB/s;
counter seen 0.51 / 0.51 / 0.70 / 0.80 / 0.86 / 0.83 / 0.86 us; 32 KB x in every CTA 1.89 / 2.08 / 2.82 / 3.74 / 4.64 /
6.27 / 8.38 us. Knee ~6.5 MB (H100 ~4.3), then +~0.45 us of hand-off latency per extra MB.
chain_oracle NOMATH: NSTG 2 (4 MB) B 106.0 / A-PDL 104.1 / plain 124.3; NSTG 10 (21 MB) B 101.6 / A-PDL 101.3 / plain 115.8
(bytes 44.5 us at 4.3). With math NSTG 2: B 151.4 / A-PDL 161.7 / plain 171.2. B beats PDL only when the consumer math is
slow (prefill overlaps compute), which our tuned kernels don't need.
=> A persistent / HBM-never-idle layer kernel buys nothing over today's PDL kernels with pre-wait prefill, on either GPU.
Hop costs are drain spread + queueing behind bytes in flight + the activation fetch, and both designs pay them.
Also checked: v16 unpool W L2 pre-fill during the twm drain: H200 W-free oracle already on record at -3.4 us/v16 (proj n5s14).

## Where the excess is at v16/v32 (largest: 112-188 us/layer), from the H200 timeline, v32 sl, for the next pick
qkv nvjet 19 us for 9.7 GFLOP (~12 at 800 TFLOP/s) | o 9.4 for 4.3 (~5.4) | pooler+router 33.4 for 19 (~24) | unpool 27
for 17.2 (~21.5) => dense GEMMs ~25 us over a good-GEMM rate (wave quantization at M=256: qkv has 36 128x128 tiles on 132 SMs)
| twm+moe_hot ~205 vs ~140 stream => ~65 | glue (routing 3.6, routegather 7.3, moe_norm ~25, 2 boundaries 16.5, attention 15) ~30.

## f1s2 (2026-10-29 19:50-20:20Z): cross-board view + O2 in-layer twm oracles (H100 only; Modal ledger full for 10-29)
### Where the top layer teams stand on the component boards (okc leaderboard, 19:55Z; our rows there may be stale)
| layer rank | team | layer | moe | proj | attention |
|---|---|---|---|---|---|
| 1 | mpeng19 | 2.418 | #4 2.616 | #30 1.569 | #19 2.740 |
| 2 | pjmullin3 | 2.403 | #2 2.685 | #15 1.584 | #2 3.032 |
| 3 | colinlikescode | 2.380 | #6 2.576 | #1 1.637 | #5 2.997 |
| 9 | zzz0906 | 2.321 | #5 2.590 | #11 1.589 | #4 3.008 |
| 63 | tech-cosmos | 2.158 | #188 2.279 | #374 1.357 | #102 2.596 |
Every top-10 layer team is >= ~2.41 moe and (bar 3) >= ~1.55 proj. Our standalone proj row (eb166fc0) is ~cuBLAS at
m=128/256 (x1.01-1.17 vs SGLang) and 1.03-1.10 on o/qkv at m>=16. Per-load arithmetic: a 1.57 proj team can't be at our
v16/v32 level unless its d-loads are >= ~1.8x, so they are faster on dense GEMMs SOMEWHERE (which loads: unknowable from the
board). Bound for v16/v32 from a smem-ingest + split-K-flush model (proj s16: ~50 GB/s/SM L2->smem): x re-reads
(N/BN) * M * K * 2 B vs reduction flush 132 * BN * M * 4 B, best BN ~256-360 -> realistic -14 us v16 sl, -8 us v32 sl
over today's cuBLAS = ~+1.2% geomean; proj's g22/g28 record (dead at BN 128) is consistent. Not my pick.

### O2 in-layer twm oracles (TWM_ORC in moe_src.py, default 0 = bitwise main; tools/fresh/run_one.py for ncu)
H100, bench.py verify loads, us (2 alternating rounds for 0/4/16, spread < 1.5 us); v1 (tk2f, untouched) = control, flat.
| variant | v16 sl | v16 fu | v32 sl | v32 fu | twm alone at v16 sl (ncu, isolated) | instr |
|---|---|---|---|---|---|---|
| base | 297.2 | 444.8 | 368.1 | 624.3 | 245.8 | 77.1M |
| ORC 1: dq ~free, wgmma kept | 287.3 | 439.9 | 364.1 | 620.5 | 223.9 | 48.6M |
| ORC 2/3/4/7: stream + skeleton (ptxas drops dq too) | ~277.5 | ~434.5 | ~351 | ~608 | 205.6 | 14.3M |
| ORC 16: every epilogue skipped at run time | 294.0 | 442.1 | 362.0 | 618.3 | 243.3 | 75.7M |
| ORC 128: epilogue without global stores | 295.3 | 444.0 | 366.3 | 623.1 | | |
- Mistake caught: I first read ORC 4 (epilogue compiled out) = -20 us as an i-cache effect (twm is 6-8k SASS, 2.2-2.6k
  without epilogues; ncu no_instruction stalls 5.4k vs 1.8k cycles/warp). ncu instruction counts show ptxas dead-code-
  eliminated the (non-volatile) wgmma and then the dq once acc was dead, so ORC 4 = stream-only. Empty `asm volatile`
  fences (fence_r) do NOT survive into ptxas. A real dq-only oracle needs a value sink (e.g. LOP3-xor of the A regs, stored
  under an impossible condition).
- Reading (H100): twm at v16 = 206 us stream + skeleton (596 MB = 2.9 TB/s) + ~22 us dq + ~18 us MMA; in-layer the
  whole consumer is worth -20 us v16 sl / -17 v32 sl / -10 / -16 full (6.6% / 4.6% of those loads). Epilogue execution
  is small (-3 v16, -6 v32 sl). On the H100 HBM is slower, so the H200 share is larger (moe n5s20: dq-free alone -22.6 /
  -23.8 us standalone on the H200).
- Consistent with moe's microbench (RS-MMA-only 512 cyc/stage = the tensor PEAK at RT4 N32 / RT2 N64; dq ALU 698;
  both 1007): the MMA is real FLOPs at peak, not padding, and dq + MMA do not overlap (dispatch/RF conflict, moe n7s47).
  So H2's prize is real (H200 TBD) but needs a structure that overlaps dq with the tensor pipe ACROSS SMs or warps
  without RF conflict, which nobody here has found. I did not find one in this session either.
### Next H200 run (Modal, after 00:00Z, ~8 GPU-min): base / ORC 1 / ORC 7 at verify 16/32 sl+full, in-layer
   (tools/h200.py muse_decode_layer base=submission.py orc1=<copy> orc7=<copy> --cmd 'python bench.py bench submission.py verify').

## f1s3 (2026-10-29 20:11-21:15Z): no build pick; the planned H200 O2 run dropped (it can't change a decision)
Re-read the H200 timeline (dev/n7s1_tl_h200.txt) per segment instead of per load, plus an H100 timeline of main at v32 sl
(tools/timeline.py: hot 100-270 || twm 102-332, i.e. same shape as the H200's 94-247 || 96-300).
- v16 sl: twm 152 us on the H200 vs the H200 stream probe on the same routings 119-134 (proj n5s20 / moe n9): the WHOLE
  expert excess at v16 is <= ~20-30 us, and dq-free (proj a9fd9026, H200) is already -22.6. So an H200 ORC 1/7 run would
  only split ~25 us into dq vs MMA; no structure to cut either is known (n7s47/48: no overlap by schedule or more WGs).
  At v16 most of the 112 us excess is the T=128 dense chain: pre-expert ~75 us vs ~30 us of bytes, tail ~27 vs ~12.
- Per-SM model of twm (n7s47 cycles, 9 KB per WG-stage): RT4 N32 consumer ~17.8 B/clk/SM vs H200 HBM share 16.5 B/clk/SM
  (4.3 TB/s / 132 / 1.98 GHz) = co-bound; RT2 N64 11.4 B/clk = consumer-bound; N8: 794 cyc = stream-bound. moe_hot at N128
  is tensor-bound (8 KB x 128 MAC / 2048 MAC/clk = 1024 cyc per 2 stages = ~8.8 B/clk). The exact-N hot fix is on record
  (moe n5s21 HOT_NX: H200 -2.3 us N104, full exact-N ~-3.7 us/v32 call): below the bar, as moe found.
- Dense T=128/256 GEMMs: per-SM L2->smem ingest (~50 GB/s/SM = ~6.6 TB/s, proj s16/s28) makes any smem-fed tile pay
  W + x*(rows/BM): qkv v32 at BM 128 x BN 256 = 113 MB = ~17 us vs nvjet 19-21.5; BM 256 needs split-K and FP32 partials
  (proj s101 g30 split-K: 21.4 vs 20.0 dead). Same verdict as f1s2 (~+1.2% geomean at best).
- Wave/SM balance at v32: twm grid = 132 with dynamic tickets, so the SMs moe_hot frees are refilled; not an idle gap.
Conclusion: every lever this fresh review produced (H1 persistent chain, H2 twm consumer, hot exact-N, dense ingest, hot/twm
balance) is closed or below ~1% on the record, and I can't name a flaw in those measurements. Nothing to build here.

## f1s6 (2026-10-29 20:52Z-): lead 20:50Z BUILD: (1) CUTLASS dense T=128/256 GEMMs, (2) hot exact-N
### (1) CUTLASS sm90 BF16 GEMM gate: FAILS on the H100 (not shipped, no H200 run)
- Image: CUTLASS headers are not in a package I can confirm from here (deep_gemm 0.2.1 / flashinfer 0.7.0 / tilelang are in
  the image, proj s25 `_diag_ext`; their bundled cutlass include dirs unverified). Moot: the gate failed first.
- tools/fresh/cutlass/cg.cu (CollectiveBuilder, swap-AB: D^T[N,tok] = W[N,K] x^T, K-major both; knobs TM TN TK CM CN,
  SCHED coop/pingpong, SK persistent/stream-K, OUTF32; runtime mode: 1x = stream-K splits/decomp nondeterministic, 10x
  = explicit split-K as a batch of K slices into FP32 slabs + torch reduce, 20x = same with W repacked slice-major),
  build.sh (35 s/variant, parallel), cgb.py (cold-L2 graph A/B vs F.linear and the layer's LT_CFG cuBLASLt pick).
- Pitfalls: stream-K/split-K workspace flags must be zeroed before EVERY launch (else the 2nd launch spins forever);
  Deterministic reduction serializes the splits; pingpong + stream-K doesn't build. Sanity: 8192x4096x8192 = 729 TFLOP/s
  (c256), so the builds are healthy; small M is the problem.
- H100 us (best CUTLASS of 8 builds x 3 modes | lt pick): qkv m128 25.1 | 17.8, m256 26.8 | 19.7; o m128 10.9 | 9.6,
  m256 13.0 | 11.6; pool m128 32.3 | 27.3, m256 32.3 | 30.1; unp m128 28.2 | 26.9, m256 31.5 | 29.9. Raw /tmp/fcg/sweep1.txt.
  Persistent 128x128 qkv m128 (18 tiles) = 41 us = ~100 GB/s per CTA; split-K 4..16 slices 29-41 us (the 2nd wave + the
  slab reduce eat the gain); W repacked slice-major (contiguous boxes) = no change, so it is not DRAM page locality.
  cuBLASLt's nvjet (same TMA warp-specialized design, tuned) is simply better here; with DeepGEMM (proj s25) and own
  g22/g23 (proj s16/s28) also losing, the T=128/256 dense GEMM lane has no public or own kernel ahead of cuBLASLt.
### (2) hot exact-N: BUILT behind HOT_XN (moe_src.py), default 16; H100 neutral, H200 A/B pending
- moe_hot (NH 128: experts with 65..128 tokens at T=256) dispatched each item at wgmma N = nv rounded up to 16 (80/96/112/128;
  HOT_XN 8 = 72..128 step 8; 0 = the old N 128). The x box stays 128 rows; only the wgmma N and the accumulator count change.
  _gen_wgmma now emits every N 8..128 (adds 104/112/120). Bitwise = HOT_XN 0 at verify 32 sl+full (files_eq: x16 4 seeds,
  x8 2 seeds). Bench routings: hot experts 91/74, 97/73, 94/76 tok -> x16 drops ~80 padded tokens per call (moe n5s21
  model: full exact-N ~-3.7 us per v32 call on the H200; its N104 probe measured -2.3).
- H100 ab_files verify 32, 5 rounds (base / x16 / x8): sl 368.5 / 367.5 / 367.9, full 621.8 / 624.4 / 625.0 = noise (the
  H100 is DRAM-bound at v32, moe n5s21: not a proxy). Needs the H200.

## f1s12 (2026-10-29 22:47Z-): lead 22:20Z O2 four-way oracle, in-layer, tuner OFF in every variant (`TUNE = {}`)
Note: f1s10/f1s11 missed the lead's 22:20Z answer (it is under Human decisions, not a `lead -> fresh` Request line).
Variants (moe_src.py TWM_ORC, default 0 = bitwise main): base 0; (i) dq free = 1 (wgmma kept); (ii) wgmma free = 8 (NEW:
dq kept alive by an XOR sink of the A regs into acc, ~2 LOP3 per 4 regs extra); (iii) both free = 7 (stream + skeleton:
items, TMA ring, flags, epilogue; ptxas drops dq); (iv) flat = 256 (NEW: moe_twm grid-stride reads every active
expert's w13+s13+w2+s2 bytes, 16 x 16 B loads in flight per thread, moe_hot exits at once, then sets the flags
moe_norme waits on; no x reads, no Inter/Y writes). (i)-(iii) touch moe_twm only; at v32 moe_hot (hot experts, own
consumer) runs unchanged beside it, (iv) also replaces moe_hot. Pitfall hit again: a sink under a compile-time-false
condition (nsplit == 12345, nsplit const 0 in the KS false build) let ptxas drop the volatile loads (twm 5 us);
the sink now tests the runtime arg hx.
H100 (bench.py verify, seed 99; v16 nact 17 = 481 MB of expert bytes; v1 = control, flat within 0.6):
| variant | v16 sl | v16 fu | v32 sl | v32 fu |
|---|---|---|---|---|
| base | 295.1 | 439.7 | 367.7 | 624.9 |
| (i) dq free | 291.0 | 436.5 | 363.1 | 621.3 |
| (ii) wgmma free | 291.4 | 440.0 | 363.2 | 620.9 |
| (iii) both free | 274.8 | 431.6 | 350.0 | 608.1 |
| (iv) flat read | 273.7 | 431.4 | 351.2 | 609.0 |
H100 reading: (iii) = (iv) within ~1 us: twm's skeleton (items, ring, ramp, drain, w13->w2 hand-off) streams as fast as a
flat read; structural cost ~0. The consumer (base - iii) is -20 / -8 / -18 / -17 us, and it is NOT additive: either half
alone saves only ~4 us. So the consumer runs at ~max(dq, MMA) + ~4, each alone ~16 us over the stream: a fix must make
BOTH cheaper (or hide the max under the stream), not one.
H200 (Modal, one container, order base/dqfree/mmafree/bothfree/flat, rounds 1, 7.8 GPU-min; tools/fresh/o2/h200.txt;
v16 nact 17 / 481 MB at seed 99; v1 control flat within 0.4 us):
| variant | v16 sl | v16 fu | v32 sl | v32 fu | bench geomean (v1/v16/v32) |
|---|---|---|---|---|---|
| base | 244.7 | 344.9 | 319.1 | 503.7 | 2.194 |
| (i) dq free | 230.3 (-14.4) | 334.3 (-10.6) | 296.7 (-22.4) | 479.2 (-24.6) | 2.280 |
| (ii) wgmma free | 233.0 (-11.7) | 336.4 (-8.5) | 296.4 (-22.7) | 487.8 (-15.9) | 2.267 |
| (iii) both free | 215.8 (-28.9) | 325.0 (-19.9) | 271.6 (-47.5) | 461.1 (-42.6) | 2.377 |
| (iv) flat read | 219.1 (-25.6) | 336.8 (-8.1) | 285.1 (-34.0) | 475.0 (-28.7) | 2.325 |
Reading (H200):
- Structural cost (iii)-(iv) is NEGATIVE: +3.3 / +11.8 / +13.5 / +13.9 us for the flat read, even though at v32 it also
  removes moe_hot's consumer. twm's TMA-bulk skeleton (items, ring, ramp, drain, hand-off) streams better than a
  16-deep ld.global grid-stride read; the ~50 us 'non-consumer' gap from the H100 ncu (205.6 vs ~152 flat bytes) does not
  exist in-layer on the H200 at this resolution. Nothing structural to cut in twm.
- Consumer (base-iii) is BIG on the H200: -28.9 / -19.9 / -47.5 / -42.6 us (v16 sl 12% of the load, v32 sl 15%).
  Full consumer free = verify geomean x1.083 = ~+4.1% on the 6-load score.
- Unlike the H100 (each half alone ~-4), on the H200 dq and MMA are ~ADDITIVE (v32 sl -22.4 + -22.7 = -45 vs -47.5;
  v16 sl -26 vs -29): faster HBM exposes the per-SM consumer, which runs dq then MMA nearly serially (= moe n7s47's
  'twm = 1.4x of max(ALU, MMA)'). Perfect overlap (sum -> max) is worth ~min(i, ii) = 12 / 9 / 22 / 16 us = ~+1.9% score;
  the remaining ~half needs fewer dq instructions or fewer MMA FLOPs.
Ranked levers, against the record:
1. dq || wgmma overlap inside an SM (~+1.9% ceiling). Record: moe n7s47 (H100 microbench, cycles/stage are SM-local, so
   H100 is valid there): 2 A-sets + wait<1>, 4 A-sets, FA3 pingpong, separate dq/MMA WGs, setmaxnreg: none reaches max;
   ncu = DISPATCH stall on dq LOP3/HMUL2 while HGMMA runs (RF/issue conflict). I can't name a flaw in that measurement.
   Untested on record, on paper only: dq on a cluster peer writing BF16 A tiles over DSMEM to an SS-wgmma CTA. Bytes:
   BF16 A = 4x the E2M1 bytes, ~17 TB/s aggregate at stream rate = ~65 B/clk per SM pair: above what DSMEM sustains
   (~32 B/clk/SM class), and it halves the MMA SMs. Dead on paper.
2. Fewer dq instructions: at its SASS floor (moe n5s35, 7 ALU ops / 8 weights via LUT; no e2m1 cvt on sm_90). Closed.
3. Fewer MMA FLOPs: N padding bound 0-2.6 us (s27 pad_bound.py), hot exact-N -1.5/-2.7 (f1s9). FP8/INT8 MMA breaks the
   BF16-activation rule. Closed.
4. KQ / R1 / WAVE / XPF / TAIL / NORME_TRIG: all are schedule/drain/hand-off items; (iii)-(iv) <= 0 says the schedule is
   not where the H200 time goes in-layer, consistent with them measuring <= ~1-6 us or worse. No re-open.
Verdict: the v16/v32 prize is real (~+4% if free) but it is the per-SM consumer, and only lever 1 attacks it, with no
known structure. A twm rebuild is justified only if someone has a concrete dq||MMA overlap form that beats n7s47's list.

## f1s15 (2026-10-30): bundle + v16 last-CTA merge (dead)
- Bundle per lead 01:00Z = v171 + HOT_XN 16 only (all other Handoffs on record are already on main). Check 8b4e6cf7 2.1539
  raw / 2.1607 fixed-SGLang.
- LCM (nvs merges its own splits at v16 sliding, S=4, NV_HALF): in-layer H100 attention segment 14.5 (combine) vs 15.5 us.
  Lesson: an in-kernel "last CTA merges" costs fence (store ack) + atomic round trip + one L2 trip for the merge, all on one
  SM after the slowest split ends; a PDL-launched combine kernel pays a grid flush + one L2 trip spread over all SMs. Same
  length or shorter, so the launch you save doesn't buy anything. Don't retry it unless the split partials stay on chip (DSMEM, which needs clusters,
  and n8s2 showed 4-CTA clusters don't fit in one wave).

## x1 (2026-10-30 07:12Z-07:30Z): lead 07:20Z lane "an EXACT new expert-kernel design" -- session 1 (H100 microbench only)
Tools: tools/fresh/xd/pipes.py (pipe rates), conc.py / conc2.py / conc3.py (WG0 HGMMA loop || WG1 ALU loop, 1 CTA, cyc per
twm RT4 stage = 16 HGMMA m64nNk16 per WG). H100 SM cycles (moe n11a: H100 = H200 exactly for these stage counts).

### First-principles candidates (written before reading the record)
1. Pipe balance: per 8 weights the dq puts 7 ops on the half-rate ALU pipe (14 cyc/SMSP) and 6 cyc on the FMA pipe.
   Move rotates to the FMA pipe (IMAD.WIDE.U32 q*2^k: lo|hi = rotl, folded into the mask LOP3; bit-identical, no repack).
2. Overlap dq with MMA (they are serial now): find WHY, then pick the structure that avoids the conflict.
3. MMA work per dq'd weight (lead's (c)) / dq shared across a cluster (lead's (a)): arithmetic check only.

### Measurements
- Pipe rates (cyc per warp-instr per SMSP, 2-8 warps/SMSP): LOP3 / SHF / PRMT 2.0 (one ALU pipe), IMAD 2.0 on the FMA pipe
  (LOP3+IMAD mix 1.0 = co-issue), HMUL2/HFMA2 bf16x2 ~1.06 (FMA pipe: IMAD+HMUL2 mix 1.5), FMUL 1.0, **IMAD.WIDE 4.1**.
- twm stage SASS (ovl8 role 1, RT4 N32): per warp 91 LOP3 + 35 SHF (= 504 ALU cyc/SMSP for 2 WGs), 64 HMUL2 + 35 IMAD,
  16 HGMMA, 320 instrs. MMA alone 513. twm 988-992 ~= ALU + MMA: zero overlap (as on record).
- WG0 HGMMA loop || WG1 LOP3 loop (112 LOP3 per stage = 235 cyc alone), WG1 time:
  | N  | RS (acc) | SS (acc) | RS scale-d 0 (no acc read) | RS same-D chains |
  | 8  | 314      | 334      | 250                        | 248 (HGMMA 3x slower) |
  | 32 | 397      | 410      | 264                        | 271 (HGMMA 3x slower) |
  | 64 | 508      | 528      | 280                        | 297 |
  HMUL2 loop (93 alone): 304 next to RS N32, 256 next to SS. scale-d 0 HGMMA alone is as fast as accumulate at N64 (461 vs 523).
  => NEW: the ALU slowdown next to an HGMMA grows with N, is the same for SS as RS, and almost vanishes when the HGMMA does
  not read its accumulator. The dq/MMA conflict is the FP32 ACCUMULATOR READ traffic through the SMSP's register file,
  not the A operand, not issue order (r-outer vs k-outer in one commit group: identical cycles), not RS vs SS.
### Verdicts (gate: <= 0.80x twm consumer cycles/stage at bs128 AND bs256)
1. Pipe balance: IMAD.WIDE is quarter-rate, so 2 rotates -> FMA = 14 cyc (worse); 1 rotate -> ALU 12 / FMA 10 = 0.86x of the
   dq's ALU time = ~0.93x of twm (dq + MMA serial). FAILS the gate alone. (IMAD-only shift layouts: bit 31 forces s = 0 and
   bit 0 forces s = 6; no left-shift-only partition of M = 0x81C081C0 exists, checked by hand; moe n5s39/laysearch covered
   total-op counts, not pipe balance.)
2. Overlap: the conflict is acc reads in the RF. Every Hopper path keeps FP32 accumulators in the RF (wgmma RS/SS, mma.sync),
   FP16 accumulators are banned (FP32 where SGLang's are), scale-d 0 needs an FADD per acc element per k16. Acc read bytes per
   MAC are fixed by k16. So no structure on one SM escapes it; dedicated dq SMs (via L2/DSMEM) give total SM-cycles = dq + MMA,
   the same as serial. FAILS.
3. (c) moot: each weight is already dq'd once per call (moe_hot runs > 64-token experts at N 128); dq per weight streamed
   can't go below 1. (a) arithmetic: each SM still dqs 1 per weight it streams, and smem then carries STS + DSMEM + SS A
   reads (>= 6 B/weight vs 128 B/clk at ~33 weights/clk). FAILS.
- Gate fails for all three. Not built. Open question for the lead (not a build): does #1's 0.76x imply they avoid acc reads
  (e.g. a legal reading of "accumulators FP32" with f16 partials per k-chunk -- banned for us by the 10-24 ruling), or fewer
  RF reads per dq op? Can't be decided by microbench.

## x3 (2026-10-30 07:54Z-08:15Z): lead 08:10Z lane "register read-port contention" (rfport), Step 1 (H100 microbench)
Tool tools/fresh/xd/conc4.py (= conc.py + roles 7-11). 1 CTA, WG0 HGMMA RS m64nNk16 loop (scale-d 1, 16 HGMMA/stage) ||
WG1 loop of 112 LOP3/stage (16 independent chains x 7). SASS checked per role (cuobjdump; loop body = 112 LOP3 each):
(a) `LOP3 R, R, Rb, Rc` 3 regs, no reuse; (b) `LOP3 R, R, imm, Rc` 2 regs; (c) `LOP3 R, R, UR, RZ` 1 vector reg;
(d) 3 regs with b/c = same register for 16 back-to-back instrs (93/112 `.reuse` on both); (e) 2 regs + imm, c reused (88/112).
All five alone: 235-236 cyc/stage. WG1 cyc/stage next to HGMMA:
  | N  | a 3reg | b 2reg+imm | c 1reg | d 3reg reuse | e 2reg reuse |
  | 8  | 341    | 318        | 307    | 321          | 316          |
  | 32 | 402    | 398        | 391    | 419          | 425          |
  | 64 | 505    | 500        | 497    | 547          | 560          |
Verdict: NOT CONFIRMED. The prediction was (c)/(d) near 235 while (a) climbs to ~500; instead 1-read LOP3 is within 2% of
3-read at N32/N64, and reuse-cache hits make it slightly WORSE (+5-12%). The slowdown is per ALU instruction, not per
register source read, so a dq rewrite for fewer RF reads cannot help; instruction count stays the cost metric (the record's
model). Step 2 not started (lead's rule). Still unexplained: why acc reads (scale-d 0 removes the slowdown, x1) tax
instruction issue rather than operand reads: plausibly the HGMMA owns the SMSP's RF read path / dispatch in bursts so any
vector-RF instruction waits regardless of its operand count. Not a lever either way.
