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

## x3 (2026-10-30 07:54Z-07:58Z real clock): lead "08:10Z" lane "register read-port contention" (rfport), Step 1 (H100 microbench)
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

## x5 (2026-10-30 09:04Z-): lead 09:10Z ORACLE SWEEP (fable/codex leftovers), no builds
### (6) router FP32 = BF16 in disguise? DEAD (1 min)
reference router = 0.045*randn FP32: (router.view(int32) & 0xFFFF).ne(0).mean() = 0.99998 on the reference weights. Not shrinkable.
### (1) pipe-selective contention (fable #7): tools/fresh/xd/conc5.py (H100, 1 CTA, WG0 HGMMA RS m64nNk16 scale-d 1 16/stage || WG1
112 instrs/stage of ONE class, 16 independent chains; cyc/stage; SASS checked: 112 of the class per loop body)
| class | alone | WG1 next to N32 (WG0) | WG1 next to N64 (WG0) | slowdown N64 |
|---|---|---|---|---|
| (HGMMA alone) | | (311) | (524) | |
| LOP3 / SHF / PRMT | 235 | 401 (341) | 490 (545) | 2.08x |
| IMAD | 233 | 428 (350) | 559 (561) | 2.40x |
| IADD3 | 188 | 391 (357) | 502 (568) | 2.67x |
| HMUL2 / HFMA2 bf16x2 | 124-126 | 425 (348-353) | 559 (561) | 4.5x |
| FFMA / FMUL fp32 | 125 | 143 (426) | 162 (628) | 1.30x, but HGMMA +103..115 |
| MUFU ex2 | 923 | 1045 (313) | 1110 (526) | 1.20x (8x slower base) |
| F2FP bf16x2 pack | 235 | 395 (361) | 496 (538) | 2.11x |
- No class escapes. FP32 FFMA/FMUL look free on their own clock but push the HGMMA out by ~ their own standalone time
  (fully serialized, just billed to the other WG). The dq's HMUL2s are the WORST class next to HGMMA (4.5x), the INT ops ~2.1x.
- "uniform datapath" roles: ptxas kept the asm on the vector datapath (8 ULOP3 in the whole binary), so not measured; and
  uniform ops can't dequantize per-thread weights anyway. Dropped.
- Verdict (1): DEAD. Every pipe that touches vector registers is taxed while an accumulating HGMMA runs; MUFU (1.2x) is the
  only light one and is 8x too slow to carry a dq.
### (2) cold-start TLB (fable #8): tools/fresh/sw/tlb.py (H100, bare 28 MB uint4 reader, events, median of 35)
| reader grid | A 512 MB flush | B A + 4 GB touched per 64 KB | C A + W pre-touched per 64 KB | D per 2 MB | F per 4 KB | E no flush |
|---|---|---|---|---|---|---|
| 132 | 19.52 | 19.55 | 19.46 | 19.49 | 19.39 | 21.63 |
| 264 | 16.19 | 16.13 | 16.26 | 16.16 | 16.10 | 18.34 |
| 528 | 15.10 | 15.17 | 15.07 | 15.17 | 15.01 | 17.82 |
- TLB thrash (B) and TLB pre-warm (C/D/F) are within 0.1 us of the plain flush (A) at every grid. A naive reader is latency-
  limited (1.5-1.9 TB/s), which is exactly where a page-walk penalty would show, and none does. (E, no flush, slower: L2-resident
  lines are not faster for this reader; not pursued.) Verdict (2): DEAD; the slow first op is not the TLB.
### (4) cache admission (codex #9): ncu dram__bytes_read vs unique expert bytes (tools/fresh/sw/uniq.py, H100, seed 99, the 2 timed calls)
- v16 sl (T=128, 163 local pairs, 17 distinct experts x 28.31 MB = 481.3 MB unique): moe_twm DRAM read 482.9 MB (L2 tex reads
  523 MB), moe_hot idle (largest expert 46 tokens). d32 sl (T=32, 9 distinct = 254.8 MB): moe_tk2f DRAM read 255.1 MB.
- Every weight byte comes from DRAM exactly once; the excess is ~1.5 MB of activations. evict_first can't cut DRAM bytes; it
  could only keep other data in L2, and nothing after the experts re-reads anything big. Verdict (4): DEAD.
### (3) sliding-attention address chain (codex #5): /tmp-built oracle file + tools/fresh/sw/pt_order.py (H100, 2 rounds, us)
Oracle file [1] = main with every page-table read replaced by slot = b*maxp + t (csl/vtk kslot/vslot; nvs/nvscl producers skip
the sm.ptr cp.async ring and compute rows). Inputs: orig = the judge's random pages; ident = the SAME logical input with the
cache rows moved to slot b*maxp + t (page table + cache_loc + k/v_cache permuted together; outputs bitwise equal to orig for
both files, 4/4 runs, checker passes).
| sl load | [0] main orig | [0] main ident | [1] no-table ident | [1] no-table orig (WRONG keys) |
|---|---|---|---|---|
| d1  | 86.22  | 86.31  | 86.29  | 86.57 |
| d16 | 189.40 | 188.93 | 187.98 | 181.61 |
| d32 | 185.99 | 185.25 | 185.57 | 172.98 |
| v1  | 142.24 | 142.25 | 142.10 | 122.85 |
| v16 | 293.57 | 292.46 | 292.64 | 291.69 |
| v32 | 369.99 | 369.50 | 368.21 | 362.82 |
- TRAP (cost me one round): the first A/B ran the oracle on the judge's input, i.e. it read the WRONG keys -> different
  attention output -> different routing/expert load: "-8 d16, -13 d32, -20 v1" was routing, not attention (v1 sl attention is
  only 7.3 us on the H200). Any page-table oracle must permute the cache with the table.
- Physical KV order (orig -> ident, main): <= 1.1 us. Page-table reads (ident, main -> no-table): -1.0 d16, +0.3 d32, -0.15 v1,
  +0.2 v16, -1.3 v32: noise-level. PW already issues the table + KV gathers under the qkv GEMM. Verdict (3): DEAD (ceiling ~1 us).
### (5) layout perturbation (codex #10): tools/fresh/sw/layout.py (H100; all 16 moe-state tensors > 4 MB re-copied at offset OFF)
| sl load | 0 | 1K | 8K | 67K | 1029K | 0 again |
|---|---|---|---|---|---|---|
| d1 | 85.87 | 86.00 | 85.82 | 86.22 | 86.07 | 86.36 |
| d16 | 189.06 | 189.03 | 189.02 | 189.05 | 188.72 | 188.90 |
| d32 | 186.20 | 187.30 | 186.61 | 187.17 | 186.09 | 186.82 |
| v1 | 141.62 | 141.50 | 141.53 | 141.58 | 141.41 | 141.51 |
| v16 | 295.65 | 296.56 | 296.25 | 295.66 | 296.00 | 296.18 |
| v32 | 367.29 | 367.30 | 368.06 | 367.61 | 367.61 | 366.90 |
All within the 0-vs-0 spread (<= 1 us). Verdict (5): DEAD.
### Sweep summary: all six items dead, none >= 1 us on a d-load: (1) no pipe escapes HGMMA contention, (2) TLB 0, (3) chain
<= ~1 us + KV order <= ~1 us, (4) DRAM bytes = unique bytes, (5) placement 0, (6) router is real FP32. 0 judge, 0 Modal.

## x8 (2026-10-31): (a) ROUTING-SLACK SIZE ORACLE (lead 09:55Z). Measurement only, NOT in submission.py.
tools/fresh/slack/slack.py (out: slack_out.txt): reference.py routings, 5 routers (default + the 4 judge-like routers of
tools/n11/judge_routers.py), 12 scored shapes, 8 input seeds each, local_start from case_fields. Exact search: smallest
set S of local experts such that every token can pick 8 distinct experts within ROUTE_TOL (0.05) of its 8th best, keeps
every local ref-top-8 expert whose margin over the 9th > KEEP_TOL (0.03 at <= 256 tokens), local routes >= 0.92 r - 4,
distinct >= 0.85 d - 2. (Search capped at 2M sets per size: if hit, the minimum is overestimated, i.e. conservative.)
```
load    ref distinct  min distinct  saved   us @6.5/expert   (mean over 5 routers x 8 inputs)
d1 sl      2.42          1.45        0.97      6.3
d1 fu      2.55          1.50        1.05      6.8
d16 sl     7.92          6.05        1.88     12.2
d16 fu     7.58          5.62        1.95     12.7
d32 sl     8.75          6.67        2.08     13.5
d32 fu     8.80          7.08        1.73     11.2
v1 sl      7.50          5.33        2.17     14.1
v1 fu      7.47          5.53        1.95     12.7
v16 sl    19.00         15.20        3.80     24.7
v16 fu    18.65         15.30        3.35     21.8
v32 sl    21.55         17.50        4.05     26.3
v32 fu    21.40         17.62        3.77     24.5
```
- Binding constraint = KEEP_TOL (the must-keep set): min distinct is on average 0.36 above |must-keep|, 1.3 above the
  distinct floor. So the slack is "drop the local experts that beat the 9th by <= 0.03 and swap in a near-tie".
- Implied at 6.5 us/expert on our H200 per-shape times (n11 table): load x1.099 d1, 1.095 d16, 1.072 d32, 1.142 v1,
  1.096 v16, 1.077 v32; geomean x1.096 (2.14 -> ~2.35). Upper-ish bound: at v16/v32 the dropped experts' routes move
  to hot experts (more MMA), and 6.5 us/expert is the d1 bytes cost. Still: it is the size of the gap to #1 (2.42/2.14 =
  1.13), so top teams exploiting it would explain much of their lead. Item closed per lead (table, then stop).

## x8 (b) d1 EXPERT SPAN (lead 09:55Z): per-CTA stamps in-layer -> pr1 routing latency -> IDF re-opened and built
Tools tools/fresh/span/: mkstamp.py (stamped copy of a submission: tk2f per-CTA + pr1 block/routing stamps + oracle
modes; anchors are the PRE-x8 pr1 source = branch fresh 3962180), run.py (stamps), ab.py / ab2.py (oracle A/B).
H200 in-layer d1 sl, us from tk2f's first CTA entry (h200_d1sl.txt, h200_pr1.txt), E2 inputs:
```
pr1 entry -19.2 | router blocks end -12.0 (31 of 32) | routing block: ticket -11.8, scores loaded -3.84 (!), top8 -2.05,
ids written -1.54 | pooler blocks end -4.1 / -3.1 med / -1.54 max | tk2f entry 0 (rmsnorm already done: 0.00-0.26)
| routed 1.54 | first TMA 2.05-2.30 | first data 3.84 | ph1 end 14.08 med (max 14.6) | cnt ok 15.1 | ph2 data 15.4 | end 19.5-19.7
```
- tk2f span in-layer = 6.9 us/expert + ~5.7 fixed (first data 3.84 + transition ~1.3 + tail ~0.3). The lead's 26 us was a
  wider timeline span (PDL early launch + n2u overlap); on these routings E2 = 19.7, E3 = 26.4.
- rmsnorm is complete when tk2f enters (stamp: griddepcontrol.wait at entry returns at 0.0-0.26 us), so tk2f's 2.3 us
  routing prologue is fully exposed (moe n7s33 assumed it overlaps the rmsnorm).
- NEW: pr1's routing block spends ~8 us (H200; ~11 H100) loading 256 scores: `__ldg(router_bias)` + `__ldg(local_start)`
  are DRAM trips queued behind the pooler's W stream. So routing is published only ~1.5 us before pr1 ends.
  That is the flaw in the IDF record (layer n7s7/n7s8, moe n7s33 'bound = 0'): IDF's early tk2f spun on a flag that came
  at -1.5 us, so it never had an early routing.
- Single-warp L2 prefetch of the routed experts' first w13 rows from pr1's routing block: +38 us per row/CTA (bulk
  prefetch issue from one SM is ~250 GB/s): dead as designed.
- Oracle (ab2.py, interleaved graphs, 5 seeds, all bitwise): H200 d1 sl/full vs base: hoist 0/-0.1; hoist+IDF (n2u IDF
  path, __ldcg) +1.3/+2.0; hoist+IDF+n2u base path -1.84/-1.54 (gate 1.5 PASS); IDF without hoist +3.0/+2.5.
- BUT the n2u base path is racy under IDF: SASS (cuobjdump of the g10 lib) shows its ids/tw loads BEFORE ACQBULK (the
  layer comment saying ptxas sinks them is not true for this build). n2u IDF path with plain loads after the wait (SASS:
  LDG.E.CONSTANT after ACQBULK; __ldcg compiles to LDG.E.STRONG.GPU and costs ~3 us) = safe.
- BUILT (branch fresh): pr1_src.cu bias/ls hoisted to smem before the wait (bitwise); layer.py n2u IDF loads plain after the
  wait; IDF_DEFAULT True. bench test 13/13; replay_test d1 0/120 (3 routers x 20 x sl/full). H100 ab_files: d1 sl 85.6 ->
  83.4 / 86.1 -> 84.1, full 106.1 -> 104.1 / 106.3 -> 103.8. H200 ab_files 8 rounds: sl 70.94 -> 70.30 (-0.64), full 86.30
  -> 85.04 (-1.26) = ~+0.18% score. Below the gate at d1 sl on the H200 as built (the oracle's extra ~1 us was the racy
  pre-wait n2u read).
- Tried, worse on H100 (not kept): V2 n2u flag spin + __ldcg pre-wait (+1.8/+0.7 vs base); V3 every slot's Y loaded
  unconditionally to overlap the post-wait routing trip (+1.1/+0.3 vs V1).
- Follow-up on the IDF build (h200_idf.txt; mkstamp.py now skips the pre-x8 oracle anchors): d1 sl E2, us from the first
  (early, router-SM) tk2f CTA: pooler blocks end 7.4 med / 8.7 max; routing written 6.1 (the score loads still take ~4-6 us:
  logits __ldcg are L2 hits queued behind the pooler's DRAM traffic, so the hoist alone does not make routing early; it is
  off the critical path now); tk2f CTAs enter 7.7 med / 10.2 max; first TMA 9.5; x ready (past wait) 10.75; first data 11.5.
  * Oracle nowait (tk2f consumers skip the rmsnorm wait, stale x): H100 ab_files +0.4 / 0.0 = x-ready hop is NOT binding
    (DEFNORM-type fusion would not pay at d1 under IDF either).
  * Oracle PFX (CTAs 0..31, landing first on the router SMs, L2-prefetch the first 1/2/4 w13 rows of CTAs 32..131 after
    their routing): H100 +1.0/+1.1/+1.5 sl; H200 (h200_pfx.txt, 5 seeds x 6 rounds) -0.16/+0.42 sl, -0.12/+0.92 full = DEAD.
  * What is left of the d1 tk2f fixed cost: per-CTA routing at entry (~1.3-1.8 us) + first-data DRAM latency (~1.8) +
    transition ~1.3 + tail. Item (b) closed with the Handoff ceb86cf.
- v1 (T=8) early-tk2f timing oracle (rmsnorm TRIG at start instead of TRIG2, no flag; /tmp/fx8/v1early.py): HANGS (40-min
  timeout, no result). Expected: without pr8's release flag tk2f CTAs read routing at different times -> inconsistent
  row splits -> the gemm13->gemm2 counter never completes. A valid v1 IDF needs a pr8 routing flag + nu8's routing path
  after the wait (as T=1). Bound by analogy with d1 ~1 us at v1 (~0.1%); not built.
- d16/d32 routing (item 1 of x8 'Next'): H200 timeline n7s1: _routing ends 0.3 us after the rmsnorm at d16 and before it
  at d32: not on the critical path. Closed.

## x9 (2026-10-31 11:44Z-, lead 11:25Z: re-stamp v174 d1 tk2f prologue, then DRAM-trip hunt in pre-wait/prologues at d1/v1)
H100 only (Modal 18/20 used in x8), 0 judge runs. Tools tools/fresh/x9 (stamp patchers + runners; /tmp/fx9 paths inside),
tools/fresh/icache (i-cache microbenches). All numbers H100, d1 sl unless noted, us.
- v174 d1 re-stamp (IDF on), vs tk2f first CTA entry: pooler blocks end med 10.5 / max 12.0; routing written 12.0-14.1;
  tk2f entry med 10.5-11.3, routed med 14.1-14.9 (most CTAs wait ~3 us for the routing flag), first data med 16.1-17.2.
- NEW: pr1's routing block (one warp) takes ~13 us from ticket to written (H100), split: fence 1.3, logits load 4-8, top-8
  4-7. Two causes found:
  (1) the 8 `__ldcg(logits)` are SERIALIZED in SASS (each LDG.STRONG.GPU after the previous sigmoid/div) = 8 L2 trips
      under the pooler stream. Fix: load all 8 first (one trip). Bitwise.
  (2) the top-8 is slow because its code is COLD: fully unrolled ~18 KB SASS on an SM that never ran it; per-k stamps
      0.5-1.5 us per k. A redux.sync argmax (2 redux + 1 shfl per k, bitwise) did NOT help (~6 us); `#pragma unroll 1`
      on the k loop did: top-8 6.5 -> 1.6 us, routing written ~3 us earlier.
  Both bitwise (files_eq d1 = 0). But layer A/B (ab_files, 16 rounds, 2 base copies): NEUTRAL (c1 84.22 sl / 104.06 full
  between base 84.36/84.58 and 103.42/104.67). Under IDF the routing is off the critical path (x8): early routing only
  lets the early tk2f CTAs compete with the pooler stream. Not kept. Patch: tools/fresh/x9/pr1_redux.cu (+ unroll note).
- i-cache microbench (tools/fresh/icache/ic.py, ic2.py; 1 warp, straight-line FFMA): the first pass over code costs
  ~0.8 us per 16 KB more than the second (1024 FFMA 1.28 vs 0.51 us); a warm-up launch on the same SM does NOT carry over
  to the next launch; > ~32 KB thrashes even when warm. In-layer under a DRAM stream it is much worse (pr1 top-8 above).
- In-situ slope oracles (pad N dead-but-live FFMAs on the critical path):
  * n2u right after its wait (tail, memory system idle): +8 KB ~0, +16 KB 0..+0.8 us end -> n2u's own ~11 KB post-wait
    code is worth <= ~0.35 us. Dead.
  * tk2f entry, all warps (during the expert stream): +8 KB ~+0.4 us graph, +16 KB +3.1/+3.6 us (late CTAs routed +2.4):
    steep and nonlinear. tk2f runs ~19 KB straight-line from entry to its TMA wait (SASS ACQBULK at 0x49a0), ~8 KB of it the
    4 inlined tk2w_bound (64-bit div calls). Rolled into a `#pragma unroll 1` loop (bitwise): late CTA entry->routed 2.3 ->
    1.8 us in stamps, but ab_files neutral (85.05 vs base 83.87/85.46). Combined pr1+tk2f (c3, layout base,c3,c3,base):
    85.01/85.59 vs 83.67/84.84 sl = no gain. Not kept.
- ab_files caveat: file [0] is the fastest in almost every run (~0.5-1.5 us position bias at d1 on the H100). Compare
  against a second base copy in a later slot, or use the base,c,c,base layout.
- d1 SASS audit, post-ACQBULK loads: g13 qkv/o GEMVs clean (W/scales/aux before the wait, only x after). n2u: ids/tw after
  the wait (IDF-correct), then Y; w1 (16 KB, every block) is written pre-wait but ptxas SINKS it after ACQBULK (0x1080):
  oracle pre-wait cp.async.bulk.prefetch.L2 of w1 -> n2u end med -0..-0.5 (~-0.25) us. Below gate. n2u W bulk prefetch
  (PFW) off: ab neutral/worse. nu8 (v1 tail): same pattern (`__ldg(w1 + k0 + tid)` after the wait, 2 KB/block): not tested,
  same bound expected (~0.25).
- n2u stamps (d1 sl, vs tk2f last CTA end): n2u CTAs land 0..0.26 (they cannot co-reside with tk2f), past wait ~1.0,
  ids/tw 2.3-3.3, Y 3.3-4.1, unpool 4.6-5.4, end 4.9-5.6. Tail = landing + 3 dependent trips + W; on record (UPF*, N2U_*).
- v1 (T=8) stamps (pr8 finish block + tk2f), vs tk2f first entry: pr8 router items done -12.3, partials summed -9.5,
  routing (8 warps) -5.1..-5.9: OFF the critical path by ~5 us (tk2f launches only when pr8 completes). tk2f at v1: all
  CTAs enter 0..1.0 together, routed 1.54 med / 2.05 max, first TMA 2.05-2.56, first data med 3.84 / max 5.6-6.4 (burst
  of 132 first TMAs), so the v1 fixed cost ~2 us prologue + ~1.8-4 us first data. Handed to moe (its 11:25Z item).
- d1 late tk2f CTAs: the last 12 tk2f CTA entries vs the last pooler block ends: all land within ~0.3 us EXCEPT one at
  +0.8 (routing-block SM) and one at +2.8-3.0 (most likely the SM of the 1-block _rmsnorm_gain: tk2f cannot co-reside).
  The late CTA's ph1end is only ~0.5-1.5 past the median (it gets bandwidth when others finish), so the bound is <= ~1 us.
  Oracle: tk2f grid nsm-1 / nsm-2 at GS 8 (bitwise eager, files_eq d1 0): result below.
  RESULT: grid nsm-1 is bitwise in EAGER mode (files_eq d1 0) but HANGS under CUDA-graph replay (stamped run_ent and
  ab_files both stalled at 100% GPU, killed after 10-25 min). Some graph-replayed sync assumes one tk2f CTA per SM (not
  found; candidates: TK2F_LATE_TRIG / IDF reset ordering). Not pursued: bound <= ~1 us and needs that invariant found first.

## x10 (2026-10-31 13:12Z-, lead 12:25Z MOE-BOARD BS1 GAP): standalone muse_moe_w4a16 bs1, H100, 0 judge, 0 Modal
- bs1 = moe_tk2f (132 CTAs, cfg 204 = PF 512) + moe_norm2<0> (1 block; NOT a PDL secondary: NORM2_PDL 0).
- Fit over seeds (tools/fresh/bs1/fit.py, graph replay cold L2 minus empty graph): t = 9.5 + 10.0 x E us
  (E = distinct local experts; seeds 100..123: E 2/3/4 = 29.6/39.3/49.8). Per-expert ~10 us = near H100 stream rate;
  the FIXED 9.5 us is the bs1 gap (layer d1 in-layer fixed ~5.7 on the H200).
- norm2<0> after its wait: ids/tw trip (ids/tw are the CALL INPUTS: legal before the wait), then inv (tk2f output) ->
  Y = 2 more dependent trips. The slot path (cfg 84204 = PF 4 -> moe_norm3 = norm2<true>) drops inv. moe's comment
  "M = 1 stays 204 because the layer's n2u reads the old y32 order" is stale: the layer runs 84204 + n2u SLOT at T=1.
- Variants (/tmp/fx10/<v>/submission.py, bench test 15/15 each), E=2 / E=3 medians, fit fixed:
    base 29.58 / 39.28 (9.50) | slot (TK2F 84204) 29.28 / 39.41 (9.29) | pdl (NORM2_PDL 1) 29.17 / 39.44 (9.17)
    both 28.51 / 38.72 (8.57) | late = both + tk2f trigger at its gemm2 rows instead of entry (layer TK2F_LATE_TRIG 3)
    28.58 / 38.51 (8.79) | late2 = late + norm3 reads ids/tw/w1 BEFORE the wait (pinned by asm) 28.45 / 38.30 (9.03)
    | latens (pdl + late, no slot) 29.14 / 39.25.
  => slot + PDL tail: ~-1.0..-1.1 us at bs1 on the H100. CAVEAT: moe n5s19 measured NORM2_PDL LOSING on the H200
  (+3.4 us d1) with the entry trigger; my reading: the 1-block norm2 lands on an SM before the 132-CTA tk2f grid is
  resident and one tk2f CTA starts late (same pattern as the layer's rmsnorm-SM CTA, x9). 'late' moves the trigger to
  the gemm2 rows, so norm2 can only land after a tk2f CTA leaves. Needs the H200. Patches: tools/fresh/bs1/late*.patch.
- Full bench (bench.py bench, H100, seeds 7/8/9), base -> late (-> late2): bs1 39.20 -> 38.21 (38.75), bs16 107.14 ->
  105.23, bs32 152.98 -> 152.58, bs8 86.14 -> 85.04, bs128 227.82 -> 226.27, bs256 265.73 -> 264.45; geomean vs
  SGLang-H200 1.846 -> 1.868 (+1.2%), late2 1.859. 'late' = TK2F 84204 (M=1 slot y32 + norm3) + NORM2_PDL 1 + tk2f
  launch_dependents moved from entry to just before its gemm2 rows. Not in submission.py of moe (moe's dir): Handoff
  posted; the deciding test is the H200 (moe check or Modal), given n5s19's H200 loss with the entry trigger.
- Layer: already has all three at T=1 (84204 + n2u SLOT, PDL n2u, TK2F_LATE_TRIG 3): no layer item from this.

## x11 (2026-10-31 13:34Z-13:56Z, lead 12:25Z MOE BS1 GAP, cont.): stamps + oracles, H100, 0 judge, 0 Modal
- Tools: tools/fresh/bs1/stamp.py (per-CTA %globaltimer in standalone moe_tk2f + norm block), strun.py (reader),
  ab.py (interleaved cold-L2 graph A/B of standalone variant dirs, run from muse_moe_w4a16/).
- Standalone bs1 E=2 timeline (us from the first tk2f CTA entry; stamped copy ~+2.5 us; 'late' variant):
  routed 1.54 | first TMA 2.05 | first stage full 4.35 med (3.6-5.4) | ph1 end 20.2 med (min 16.4 = the 3-row CTAs,
  max 20.7) | cnt ok 21.25-21.75 | ph2 data 22.0 | end 26.6 med / 26.9 max | norm3 pastwait 27.65 | Y loaded 28.9 |
  end 29.7. Base (norm2 not PDL) has the norm entering at 27.9 and ending at 30.5.
- Oracles (ab.py, 6 seeds x 10 rounds, bs1, GPU shared so absolute numbers run ~+5 us; means):
    base 38.81 | late 37.70 | O1 = late + norm3 does nothing after its wait (wrong output) 35.26 (-2.4)
    O2 = late without the gemm13 -> gemm2 cnt wait (timing only) 36.22 (-1.4)
  => bs1 fixed ~9.5 us = pre-data ~4.3 (routing 1.5 + TMA issue 0.5 + first data 2.3) + transition 1.4 + norm 2.4 +
  launch/PDL ~1. Record: fused norm in tk2f (s9 TK2FN, n5s28) = same latency chain, dead; R2 column-stationary never
  built (est. <= 1 us); in the LAYER the norm's successor n2u needs the full vector in every block anyway (no gain).
- In-layer (tools/ab_files.py, H100, v174 = branch fresh = main):
    O2 (tk2f_consume cnt + BAL waits removed), 2 runs: d1 sl -0.6/-1.7, d1 full -0.8/-1.4, d16 sl -0.9/-0.8,
    d16 full +0.2/-0.6, d32 sl -1.6, d32 full 0.0 us  => transition bound ~1.1 us d1, ~0.5-0.8 d16/d32.
    O3 (n2u returns right after its wait + counter zeroing): d1 sl/full -3.5/-4.4.  O3b (n2u without the last-block
    boundary): 0.0/-0.2. n2u = latency chain (PDL release ~0.9 + L2 trips); record: redundant-read fix, flag spin,
    unconditional Y, fused boundary all dead (NOTES-layer s14/s57/v94, fresh x8). Not reopened.
- CANDIDATE CW (chunk-wait; NOT built): gemm2 stage q needs only inter[q*1024, +1024) = gemm13 row-blocks 64q..64q+63
  of its expert (stage q <-> kc 16q..16q+15, warps interleave K). Today every gemm13 chunk completes at the END of ph1
  (each CTA's rows are contiguous and every chunk spans ~16 CTAs). CW: (1) gemm13 rows strided (CTA b: L = b + jG) and
  mapped q-major (L -> q = L / (64 nact), a, rb = 64q + L % 64) so position j of every CTA lands in chunk ~j;
  (2) per-(expert, chunk) publish counters (REL helper warp, off the consumer path); (3) gemm2: at the first row of a
  segment wait chunks 0..2 + load xf qq 0..5, poll chunk 3 + load xf qq 6..7 during stage 2. Hides the barrier +
  imbalance behind ~3 stages (~1.9 us). Pitfalls: at T=1 gemm13's x is the same token for every segment (skip the
  per-segment xf reload or strided rows reload it every row); producer must walk the same permuted order; inter-CTA
  flags need the t16c-style many-routing stress test (moe s7 lesson). Estimate: ~0.6 us d1, ~0.3-0.5 d16/d32, ~1 us
  standalone bs1 (bound 1.4). Lead's 11:25Z bar is 0.5 us at d1 sl: marginal; asked the lead before building.

## x12 (2026-10-31 13:58Z-, lead 13:58Z: BUILD CW): v1 NEUTRAL (wrong permutation for d1's row counts); v3 WINS d1 sl -0.5 / full -1.55 us (H100). 0 judge, 0 Modal.
- (v1 below: patch renamed tools/fresh/cw/cw_v1_dead.patch.)
- Built (tools/fresh/cw/cw_helper_dead.patch on v174 = branch fresh b15f672; layer TK2F_CW, cfg 184294 = PF 4 | 8192, T=1
  only): gemm13 rows permuted in each 256-row segment (cw_row), CTA order (chunk, u, v & 3), producer walks the same
  order, T=1 x fragments kept across segments, per-(segment, chunk) counters cnt[64 + seg * 4 + c], gemm2 waits chunks
  0..2 at a segment start and polls chunk 3 at stages 1-2 (blocking before stage 3). Publishes first by consumer
  bar.sync 2 + tid-0 fence (cw), then by a helper warp (cwh, tk2f_rel_on extended). tools/fresh/cw/stress.py: 300 eager +
  300 graph-replay routings (0..8 local experts) bitwise vs base; bench test 13/13. PF 8196 confirmed in the profile.
- H100 tools/ab_files.py "decode 1 " (us vs base, sl / full): cw +0.0 / -0.2; cwh +0.4 / -0.35; cwh with red.release
  instead of fence + atomic: same; ORACLES: cw without any chunk wait -0.45 / -0.4; cwh without waits -0.5 / -0.9;
  cw without waits and without publishes -0.9 / -1.1 (the whole transition bound, cf. x11 O2 ~1.1).
- WHY (stamps, tools/fresh/cw/stamp.py + strun.py: per-CTA %globaltimer into an enlarged cnt buffer; d1 sl seed 99):
  nact = 3 -> 768 gemm13 rows / 132 CTAs = 5.8 rows per CTA at ~3.3 us/row (~2.6 TB/s H100 stream). Every CTA starts
  gemm13 at the same time (14.4 us; TK2FX wait), but per-CTA stream rates differ ~25%: first (chunk-0) publish med 24.7,
  max 30.4; gemm13 end 30.2..38.1. A segment's chunk spans ~22 CTAs, so chunk 0 completes with its SLOWEST CTA:
  chunk complete times per segment 33.8 / 37.9 / 37.2 / 38.4 (chunks 0..3), 32.1 / 37.2 / 37.1 / 37.2, 32.7 / 37.5 /
  37.2 / 39.1. Chunks 0-2 are not early, and each publish costs a ~1.9 us fence (helper stamp pre/post
  __threadfence: 32.7 -> 34.6 us; red.release the same), i.e. one DRAM-loaded memory round trip.
- Lessons: (1) at d1 a 'chunk' of a segment is ~1.5 rows of each of ~22 CTAs: row ORDER inside a CTA cannot make a chunk
  early when CTA streams are unfair; (2) inter-CTA publish = ~1.9 us fence under the stream, each one; more publishes
  only pay off if they are off the consumers' path AND the data they release is early; (3) the remaining d1
  transition bound is ~1 us total on the H100 (O2 / ob).
- CORRECTION (same session): the 'unfair streams' reading was WRONG. %smid stamps on the base: gemm13 us/row per SM 3.54..3.70
  (sl), 3.44..3.60 (full), split-half correlation 0.15-0.22 = uniform. The v1 permutation (logical 16 u + v -> 16 v + u)
  assumed ~16 rows per CTA (nact 8); at d1 sl nact = 3 -> 5.8 rows/CTA cover 1-2 chunks only, so chunks finished with
  whole CTAs. v2: physical = (i & 3) * 64 + i / 4 inside the segment (any 4 consecutive logical rows hit all 4 chunks),
  CTA order (chunk, l). Chunk completion now 26.1 / 31.8 / 35.2 / 36.3 us (seg 0; others alike). v3 = v2 + gemm2 waits
  only chunk 0 at a segment start, then polls the next pending chunk before each stage and blocks only on the chunk that
  stage needs (compile-time xf indices: unrolled j = 1..3 with pend == j).
- v3 = branch fresh (tools/fresh/cw/cw3.patch on v174): bitwise vs base in tools/fresh/cw/stress.py 1500 eager + 1500
  replay routings; bench test 13/13; tools/replay_test.py d1/d16 seeds 11,12 x 40: 0 fails; files_eq d1 x 6 seeds: same
  as base-vs-base nondeterminism (pre-existing atomics elsewhere).
- H100 timing (tools/fresh/cw/ab_seeds.py = ab_files over seeds; SEEDS=99,7,123,5, 8 rounds, base,cw3,cw2,cw3,base):
  d1 sl base 74.24/74.72, cw3 73.99/73.93 (-0.5), cw2 74.50; d1 full base 90.02/90.11, cw3 88.42/88.62 (-1.55), cw2 89.30.
  Seed 99 runs (ab_files, 12 rounds): cw2 sl -0.6/-0.8/-0.1, full -1.7/-2.0/-1.4; cw3 sl -0.1, full -1.4 (shared GPU).
- Scope: T = 1 only (cfg 184294; cnt[64 + seg * 4 + c] needs seg < 8; TORCH_CHECK nr == 8). v1 (T=8, up to 64 segments)
  would need a bigger counter area (n2u zeroes cnt[0..127]) and its own permutation check; d16/d32 use BAL / tk2h.

## x13 (2026-10-31 14:27Z-, lead 14:30Z: CW for T=8 and T=16/32): T=8 CW NEUTRAL; d32 tk2h per-segment counters WIN (H100 -2.7/-2.2 us)
- CW at T=8 (v1, nu8 path, cfg 184294 non-SQ; cnt grown to 256, nu8 zeroes [0, 192); producer/consumer rescan from the
  CTA's first segment instead of 0): bitwise y32 rows 0..63 vs base (tools/fresh/cw/stress8.py 300 eager + 300 replay;
  y32 rows 64..80 = nu8 atomics, differ even base vs base). H100 v1 (ab_seeds 4 seeds x 10): sl +1.4, full +1.7 us.
  Stamps (stamp8.py/strun8.py, stamps at cnt + 256): gemm2 wait-ok median 54.8 vs 58.7 base (earlier) but the slowest
  gemm13 CTA +3 us (g2 entry max 61.1 vs 58.2): the T=1 chunk-major order switches segments 8x per CTA and at T=8 every
  switch reloads other tokens' x fragments. Segment-major CwIt (chunk-major inside each segment, smaller part first;
  g_cwsm = nroutes > 8): max 59.5, end max 84.7 vs 83.9 base; v1 A/B +0.3 sl / -0.3 full = neutral.
  ORACLE closes it: gemm2 never waiting at all (wrong output) = v1 sl +1.0 / full -0.5 us. At T=8 (~60 rows/CTA) the
  producer streams gemm2 weights without waiting, so the transition hides behind the stream. Patch cw8_sm_neutral.patch.
- d16 (BAL tk2f, already per-segment counters): no-wait oracle -0.3 / 0.0 us: nothing left.
- d32 (tk2h, MRG 1 = ONE global counter: every gemm2 row waits for ALL gemm13 rows): no-wait oracle -2.9 sl / -1.9 full.
  The per-segment variant already exists in moe_src (MRG 2: tk2f_rel_helper<SEGC> publishes cnt[64 + seg], tk2h_consume
  PSC waits on its own segment); the layer only built MRG 1. MRG 2: H100 d32 sl -2.8 / full -1.9 (4 seeds x 8), final
  file vs HEAD -2.7 / -2.2; d1/d16/v1 unchanged code. test 13/13, replay d1/d16/d32 0/120 each, files_eq d32 sl bitwise
  (full differs as base vs base), cnt[] all zero after every d32 call. Record: moe n6 measured PSC on tk2f (h8p) on the
  H200 = 0 (balanced CTAs); tk2h's GS16 weighted split (2-group rows) is less balanced, so the H200 may keep some.
- CHECK 9eeab096 (branch fresh 9325413 = main v176 + MRG 2): raw 2.1736, fixed-SGLang 2.1460 (v176 check 0ac4a2f0: 2.1930).
  Slow draw on unchanged shapes (v16 sl 267.1 vs 250.0, v32 sl 330.6 vs 303.7); d32 sl 157.8 vs 154.6, full 306.8 vs
  306.3. In-run tuner T=32 sl default 157.7 (v176 check 148.5, v175 164.1): too noisy to resolve 2-3 us. NOT confirmed
  on the H200; with moe's H200 PSC = 0 on record, treat as unproven. Decisive next: Modal H200 A/B after 00:00Z,
  `tools/h200.py muse_decode_layer base=<main v176 submission.py> cand=<fresh submission.py> --cmd 'python bench.py bench submission.py decode 32'` (~3 GPU-min).

## x14 (2026-10-31 15:00Z-, lead 15:00Z: where RF_EARLY's wins came from; does the early-trigger pattern apply elsewhere?)
- RF_EARLY (layer n17s1) = the merge kernel triggers at entry, so o (a PDL secondary that fills its W ring pre-wait)
  launches as soon as the merge's programs are resident instead of after the merge's wait. The general pattern: a kernel
  whose pre-wait work is input-only (weights, KV of old tokens, page tables) should be a PDL secondary and launch early;
  its trigger must stay AFTER its wait whenever its own secondary reads a predecessor's output pre-wait (_combine_hb and
  _red_fin load the qkv gate before their gdc_wait).
- Audit (H100 tools/n13/tlmed.py, v176, all 12 loads): o d1/v1, unpool, n2u/nu8, bnd2, pooler are already early-launched
  (or set by resources: d32 g30k pooler waits for o's CTAs to leave; v1 front = pr8's stream, s88/n5s5). The gaps were
  the ATTENTION kernels after qkv: nds (d32 sl), nvs (v16/v32 sl), _gd_full (d1/d16/d32 full), _attn_split (v1 full)
  were launched with plain <<<>>> / no launch_pdl: they started only after qkv drained (d32 sl nds 15.8 vs qkv end 15.0).
- Built (attn_src.py; submission 998 KB: nds's duplicate line comments dropped for the 1 MB cap):
  * NDS_PW: nds via cudaLaunchKernelEx + ProgrammaticStreamSerialization; the KV producer warpgroup (seq_lens, page
    table, old KV slots: inputs only) runs pre-wait; the consumer does griddepcontrol.wait, then launch_dependents, then
    qkv. Timeline d32 sl: nds starts 1.5 us (during qkv), qkv +0.9, nds end 27.9 -> 24.9, o end 37.9 -> 33.8, pooler end -4.7.
  * NVS_PW: the same for nvs (verify sliding T=16/32). nvjet qkv never triggers early, so only ~0.3 us of start.
  * GD_PW (batch > 1 only; d1 full keeps QF and is bitwise unchanged): _gd_full launch_pdl + wait before the queries.
    d1 full form measured tie (latency-bound gathers), so excluded.
  * CB_HB_TRIG on: with NDS_PW, _combine_hb launches only after nds's wait, so its entry trigger lets o fill its ring under
    the merge (n5s4 had measured it as noise without NDS_PW).
- H100 (ab_files, TUNE pinned off, 4 rounds off,on,on,off; all x14 changes): d32 sl 182.7/182.5 -> 178.6/180.4 (-3.1),
  v16 sl -0.7, v32 sl -0.2, d32 full -0.5, rest noise. ab_flag (one state): GD_PW d16 full 292.1/291.8 -> 289.6/289.9
  (-2.2; pinned-tuner run: tie, so it depends on the qkv pick), CB_HB_TRIG d32 sl -1.4/-1.6 (seeds 99/7).
  CAUTION: ab_files/graph_eq with the tuner ON compare different tuner picks per file (d16 full qkv g24 vs g30 seen);
  pin TUNE (sed 's/^TUNE = {1: /TUNE = {} and {1: /') for A/B of whole files.
- Correctness: test 13/13; graph_eq (pinned) d32 sl / v16 sl / v32 sl bitwise, d16/d32 full ids equal + diffs at the
  base-vs-base atomics level; replay_test 0 fails (sliding 0/360, verify 0/360, decode 0/360, all 12 loads 0/240).
- CHECK 05077ac2 (main v176 + x14 attn_src.py, no MRG 2): FAIL at v32 sl recheck rep 1 (routing: token 129 dropped
  expert 218, 3.05e-02 > 0.03). Not a race: x14 is bitwise vs off at v32 sl (graph_eq), the judge's exact input passes
  locally on v176 and x14 with identical errors (tools/fresh/x14/repro_v32.py), and route_judge v32 sl worst margins are
  identical for v176 and x14 (0.0202/0.0229, FP8 reference's own 0.0213/0.0233 over 80 inputs): a tail draw any run
  can hit at T=256. The 11 timed shapes: d32 sl 152.1 (v176 check 154.6), d16 full 200.8 (206.0), d1 full 71.9 (73.5),
  v1 sl 104.7 (110.5), v16 sl 258.4 (250.0); fixed-SGLang with v32 sl/full imputed from v176 = 2.1962 (v176 2.1913).
- Next: Modal H200 A/B after 00:00Z (v176 vs v176 + x14 attn_src.py; d32 sl, v16 sl, d16 full), then the layer merges.

## x15 (2026-10-31 15:32Z-16:20Z): lead 15:25Z "FRESH/LAYER: continue"; layer is checking v177 (= v176 + MRG 2 + x14). 0 judge, 0 Modal.
- CORRECTION to x14 ('the judge's exact input passes locally'): the judge failed on REPETITION 1 of v32 sl (okc status
  05077ac2ccc34568 --json: "recheck failed on repetition 1"), and each run also draws its own router seed (reference.py
  weights_fields); tools/fresh/x14/repro_v32.py uses the case seed + our fixed router, i.e. NOT the failing input. So the
  failing input was never reproduced. Race check by reading the code: NVS_PW's pre-wait producer reads only seq_lens, the page
  table and OLD KV slots; the new tokens' K/V come from the consumer (after griddepcontrol.wait) in registers (isnew), so no
  pre-wait read of qkv output or of slots written this call. Record: 2 routing fails in ~100 layer judge runs (f5fdc570 d1 rep
  44 before any PDL attention, 05077ac2 v32 sl) = ~1-2 % per run: a tail every FP8-query implementation carries. The v177
  check (layer n22s1) is the H200 evidence; if v32 sl fails again there, suspect NVS_PW first (flag NVS_PW 0 in attn_src.py).
- H100 timelines on branch fresh (v176 + MRG 2 + x14, tools/n13/tlmed.py, 25 cold replays, seed 99):
  d16 sl total 189.9: qkv 0-14.1 | csl -25.2 | o -29.8 | bnd -33.9 | pooler 28.4-54.2 || router_split/_routing -56.2 |
  rmsnorm -56.9 | tk2f 56.8-167.3 (10 distinct local experts = 283 MB, 2.57 TB/s) | norm2 -169.9 | unpool -187.5 | bnd -189.9.
  v1 sl total 142.3: qkv -14.6 | vtk -19.6 | o -26.0 | bnd -29.6 | pr8 18.7-50.4 | tk2f 50.7-133.7 (8 experts) | nu8 -139.0 | bnd2 -142.3.
  Nothing new vs the record: the front chain qkv->attn->o->bnd (first-op cold reader + hops) and the pooler window are the known
  items (proj n7s4/s46/s17, layer n10s1, FRESH x5 (2)); o already streams its W pre-wait.
- Next (x16, after 00:00Z): Modal H200 A/B (my 20 GPU-min share): v176 vs v177 (or v176 + x14 attn_src / + MRG 2 separately if
  v177's check is ambiguous), d32 sl+full, v16 sl, d16 full, `tools/h200.py muse_decode_layer base=<v176> cand=<cand> --cmd
  'python bench.py bench submission.py decode 32'` (~3 GPU-min per load pair; one load per call, R small: x? 11-min timeout).

## x16 (2026-10-31 15:51Z-, own pick: the routing-check FAIL risk; 1 judge check 2d0914bf, 0 Modal)
- Why: the judge failed a layer run twice on routing (05077ac2 v32 sl, f5fdc570 d1): ~1-2% of runs. Final ranks are medians
  of reruns on held-out data, so a fail in a rerun could cost the whole problem. Is our router-score error our own or FP8's?
- Reference oracle (tools/fresh/x16/qerr.py, reference only): ref with E4M3 queries vs the BF16-query reference, per scored
  shape, 40-450 inputs, router seeds 11/23/37/41/53: max |d biased| 0.026-0.056, p99.99 0.022-0.030, worst dropped-keep lead
  0.005 (d1) .. 0.020-0.025 (v16/v32), limit 0.03. Per-input worst drop-lead at v32 full: 3% of inputs > 0.015, 1% > 0.02 (tail
  -> a fail every ~10^3-10^4 inputs at v16/v32, which matches ~1 per 100 runs x 60 reps). With two-term queries
  q = e4m3(q) + e4m3(q - e4m3(q)) ('hilo'): max |d| 0.0016-0.0036, worst drop-lead <= 0.0014 at every shape (15x smaller).
  Our kernels' margins equal the E4M3 reference's (route_judge), so the FP8 query rounding IS our routing error.
- Built QLO in nvs (v16/v32 sl, both _NVS_SRC and _NVSCL_SRC): a lo plane next to q in smem, 2 more k32 wgmma per QK tile;
  and in _gw_verify's gbal consumer (v16/v32 full; Gluon chained warpgroup_mma on the async token). Results (H100):
  route_judge drop-lead v16/v32 sl 0.020/0.016 -> 0.0016-0.0048; v16/v32 full 0.013/0.018 -> 0.003/0.007; scored_test layer
  update error 6% -> 1.2-1.6% (bound ~14.5%); test 13/13, replay_test verify 0/360. Kernel cost (tlmed, H100): nvs +0.3
  (v16) / +0.7 (v32); _gw_verify +0.4..1 (HBM-bound on the H100).
- CHECK 2d0914bf (v177 + QLO nvs + gw): raw 2.1745 (CI 2.142-2.207), fixed-SGLang 2.1405: slow node (SGLang v32 sl 756 vs 713,
  v32 full 1120 vs 1057 in v177's a4582885). H200 per-kernel rows (vs a4582885): nvs 9.0 -> 9.4 (v16 sl), 13.8 -> 14.0 (v32 sl);
  _gw_verify 99.1 -> 108.0 (v16 full), 190.2 -> 207.5 (v32 full) = +9%: the H200 verify-full kernel is MMA-bound enough that
  +50% QK^T tensor work shows. ~0.3% score for GW -> DROPPED from the build; nvs QLO kept (~+0.3 us, ~0.02%).
- Branch fresh 4725658 = v177 + QLO nvs only (997512 B; _NVS_SRC's comment-only lines dropped for the 1 MB cap).
  GW variant: python tools/fresh/x16/qlo_patch.py <v177 file> <out> NVSCL,NVS,GW. Decision for the human: GW QLO buys the
  v16/v32 full half of the fail risk for ~0.3% score. d32 sl (nds) / d16-d32 full (_gd_full) not done: ~10% of the risk
  (fails scale with tokens: v32 256, v16 128, d32 32); nds has free A rows 16-31 but in another warp (needs a cross-warp add).
- Cheaper GW forms to try if wanted: none obvious (hi+lo = 2 FP8 MMAs = the tensor cost of one FP16 QK^T; K is FP8, so FP16 Q
  would also need K upconverted).

## x19 (2026-10-31 17:20Z-17:55Z): H200 A/B v176 (b051588) vs v177 (51eaeff), Modal, tuner pinned off (TUNE = {} and ...), 0 judge
- Modal bench.py bench per load, one container per load (absolute times are Modal's, compare within a run only):
  decode 32 ABBA: sl base 146.10/146.19, cand 142.06/141.68 (-4.3); full base 304.42/304.83, cand 301.22/301.28 (-3.4).
  decode 16 (1 round): sl 144.66 -> 144.64 (tie); full 218.18 -> 217.30 (-0.9).
  verify 16 (partial, the first script's tail -12): full 345.76 (base, last round) vs 346.85 (cand) = tie.
- Verdict: everything v177 added (MRG 2, NDS_PW, CB_HB_TRIG, GD_PW, NVS_PW) is neutral-or-better on the H200; d32 gains match
  the H100 (-4.6/-3.1). ~ +0.35% score from d32 alone. Modal 16.6 GPU-min (ledger 49/60 on 10-31).

## x20 (2026-10-31 17:21Z-17:40Z): own pick. The record's "per-SM ingest wall" re-measured: it is the chip's DISTINCT-LINE L2 rate
Why: every own T=128/256 dense GEMM design was closed on paper by "per-SM L2->smem ingest ~47-60 GB/s/SM" (proj s16/s28/s90,
FRESH f1s2/f1s3), but cuBLAS's own v32 qkv pick (nvjet 72x64 tiles, 128 CTAs) ingests ~285 MB in 19.1 us (H100) = ~117 GB/s/SM.
Also: H100 timeline of main v181 (tools/fresh/x20/tl.py = tlmed.py that keeps cuBLAS reduce kernels), seed 99:
  v32 sl 362.5: qkv nvjet 19.1 | nvs 14.1 | o 9.7 | bnd 7.2 | pooler+router nvjet 36.2 | routing 3.5 | rgather 6.5 | moe_hot 157 ||
  twm 220 (96-316) | moe_norm -318 | unpool nvjet 32.6 | bnd 11.2.   v16 sl 301.7: qkv 18.1 | nvs+cb 11.6 | o 7.1 | bnd 5.5 |
  pooler 33.6 | routing/rg -84 | twm 188 (84-272) | moe_norm -275 | g30 unpool -297 | bnd -301.7.
Microbenches (tools/fresh/x20; one CTA per SM, 8 MB L2-resident region unless noted; H100 unless noted):
- ldg.cu (ld.global.cg v4, 256-1024 thr): every CTA reading DISTINCT lines: chip 7.5-8.2 TB/s (H100), 8.6-8.7 (H200) = 57-66
  GB/s/SM. Every CTA reading the SAME lines at the same time: 16.1 TB/s at 1024 thr (H100 and H200 alike), 21 TB/s with 2 CTAs/SM
  = 122-159 GB/s/SM. (The 1 GB rows of ldg.cu overlap windows: ignore them.)
- ingest*.cu / ring6.cu (cp.async.bulk ring): per-SM rate is NOT capped near 50: 33 SMs alone pull 115 GB/s/SM from distinct
  lines; all 132 SMs on shared lines 125-150 GB/s/SM. The chip cap on distinct lines binds at ~57 GB/s/SM.
- pc.cu (g30-like: producer lane waits EMPTY, expect_tx, bulk; consumer warp waits FULL, arrives EMPTY): one producer thread has a
  ~100-110 ns floor per stage (H100 = H200): 6 KB stages <= ~60 GB/s/SM (2 producer threads 92), 16 KB stages 150. HBM-streamed
  (1 GB) rows: H100 23.4 GB/s/SM = 3.09 TB/s, H200 34.6 = 4.57 TB/s, for any stage size/producer count.
  (A ring where one thread waits FULL then re-issues, with a 64-bit % per stage, showed a fake ~450 ns/stage cap: ingest.cu/ring5.cu.)
- H200 (Modal, 0.2 GPU-min): same table (pc + ldg) ran in the judge image: numbers above.
Reading:
- The "wall" the record measured (g22: W 12 KB + x 16 KB per stage at 47-50 GB/s/SM; moe s16 bare ring ~60) is the chip's
  distinct-line L2 rate (~7.5 TB/s H100 / ~8.6 H200 over 132 SMs), NOT a per-SM port. Re-reads of the SAME x lines by many CTAs
  at the same moment cost ~2x less of it (L2 merges concurrent same-line requests; multicast was not needed, consistent with
  proj's "multicast halves L2 reads, no gain").
- So x re-reads in a GEMM are ~free IF the CTAs walk K in lockstep (all CTAs issue the same x box within a few stages), and cost
  distinct-line L2 bandwidth if they drift (dynamic tickets, staggered starts, split-K slices at different K).
- Paper bounds that closed designs on "ingest <= ~50 GB/s/SM" are invalid as written (proj s90 'own m128 GEMM dead', FRESH f1s2/f1s3
  '~+1.2% at best'). MEASURED verdicts stand (g22, CUTLASS f1s6, DeepGEMM, g30 split-K s101, n9s3 DSMEM).
  The real bound for T=128/256 dense GEMMs is HBM + tensor: v32 qkv ~max(8.8 bf16 / 6.6 12-bit, 9.8 us) vs nvjet 19 (H200);
  pooler+router / unpool ~17-20 vs ~30-33; qkv/o/pooler/unpool sum at v32 sl H100 97.6 us vs ~72 ideal; ~15-20 at v16 sl.
  Ceiling ~1.5-2% score if a kernel reached it; nobody has a kernel within 20% of nvjet there yet.
- twm (v16/v32): x boxes are re-read by CTAs on dynamic tickets (out of lockstep): W 4.6 + x ~2.3 TB/s at RT4 N32 = ~80% of the H200
  distinct-line cap. Not a proven limiter (in-layer oracle (iii) stream+skeleton runs fast), but an x-in-lockstep schedule (all
  CTAs of one expert take its row tiles in the same K order at the same time) is the cheap form if the consumer ever gets faster.
## x21 (2026-10-31 17:35Z-18:25Z): own pick (no lead answer to x20's lane question): T=256 dense GEMM headroom, H100, 0 judge, 0 Modal
Question: x20 showed the "ingest wall" that closed the s100 T=256 g30 form (dev/s100_g30_tt2.cu, 38.9-40.7 us vs cuBLAS 27.6) is
not a per-SM limit. So what DID limit it, and is there >= 1% left in the T=256 (v32) dense GEMMs?
- tools/fresh/x21/tri.py: Triton TMA/wgmma GEMM screen (BM/BN/BK/split-K/warps/stages) on (256,4096,8192), (256,8192,4096),
  (256,2304,8192), (128,2304,8192), (128,4096,8192): never beats nvjet (best +1..+27%). (Flush with zero_() = dirty-L2 writebacks
  inside the timed GEMM, +5-15 us: flush by READING, as bench.time_graph does.)
- tools/fresh/x21/tt2o.py + tt2o.cu (s100 tt2 + oracle defines), unpool 8192x4096 at M=256, H100 graph us minus empty, cuBLAS 28.5-28.9:
    base 38.9 | x L2-hot (O_X=4) 39.0 | W L2-hot 38.6 | both hot 38.6 | no x loads at all 38.5 | no x + no W loads (no memory) 37.0
    no MMA 25.7 | no MMA, no memory 17.3 | + no escape fixup 12.8 | skeleton (no MMA/mem/decode) 12.8
    no escape fixup (G30_FIX) 33.8 | MMA only (no mem/decode/fix) 27.4 | no-decode with memory 30.0
  => memory is NOT the limit (-1.9 us with no memory at all). Consumer-bound: in-loop escape fixup ~5 us, sync skeleton ~200 ns
  per stage, and the MMA loop itself 27.4 us. Decode is ~free.
- gen_mma.py (pure RS wgmma loop, 128/132 CTAs x 64 rows x K 4096 x 256 tokens, event timing incl. launch): n128 both-WGs 24.4-25.6 us,
  n256 alternating stages 24.3-24.8 = 690-730 TFLOPS. So a T=256 4096x8192 GEMM has a tensor floor ~22-25 us on the H100 and
  cuBLAS (28.5-28.9 standalone) sits within ~3-4 us of it. Same for pooler/FR (17 GF) and less for qkv (9.7 GF).
Verdict: T=256 dense lane DEAD: ceiling ~3-4 us per GEMM x 3 GEMMs on v32 only, before any real-kernel costs (~0.2-0.4% score max),
and nvjet's tensor rate equals our best loop. (The H200 has the same tensor rate; its faster HBM only makes T=256 more tensor-bound.)
Left open (not measured): T=128 qkv (v16): nvjet ~17-18 us vs ~9 us 12-bit-bytes/~6.6 tensor floors. s101's split-K g30 qkv
(= cuBLAS 20.8 H100) carried the in-loop escape fixup (~5 us/64 stages here) and the ~200 ns/stage skeleton; a GX-escape (out of
loop) split-K form is the only untried variant I can name. Gate before building: tt2o-style oracles on s101 (no-fix, no-memory).

## x22 (2026-10-31 18:24Z-18:55Z): own pick (x21's last open item): T=128 qkv split-K g30 oracles, H100, 0 judge, 0 Modal
tools/fresh/x22/s101o.cu = dev/s101_g30split.cu + oracle defines (O_NOX/O_NOWL no x/W loads, O_NODEC, O_NOFIX, O_NOMMA, O_NOLOOP,
O_NOEPI = CTAs exit after the WG reduce, O_DSM = split-K reduce-scatter in a (1,S,1) cluster over DSMEM instead of global
slabs + tickets, O_E 1/2 cut points; O_E 3 and an O_NOWAIT variant HANG: don't use). Run: SRC=tools/fresh/x22/s101o.cu
PRE="#define O_DSM 1;..." SPLIT=2,3 python dev/s101_g30real.py (qkv_gate 2304x8192, M=128, real weights, 1301 escapes).
H100 cold-L2 graph us minus empty, cuBLAS 20.3-21.2:
    split 3: full 21.6-21.8 | no fix 20.3 | no x 21.0 | no W 20.4 | no memory 20.5 | no MMA 20.1 | skeleton (no mem/MMA/dec) 12.7
             no epilogue 15.95 | + no fix 14.9 | no mem, no epi 15.1 | MMA only, no epi 11.2 (with epi 17.1)
             skeleton no epi 6.2 | no loop 8.6 | no loop no epi (launch + empty CTAs) 2.75
    split 2: full 25.6, no epi 20.5; split 1 no epi 35.9 (36 CTAs); split 4 = 144 CTAs (2 waves) 32.3.
    O_DSM: full split 3 20.9-21.6 (vs 21.8), split 2 24.8-25.1; skeleton no loop 8.0 (slabs 8.6); E1 (tile -> own smem, exit) 4.0,
           E2 (+ rdy/dne cluster mbarrier handshake, no reads) 6.2 => one cross-CTA handshake ~2.2 us; the DSMEM reads ~1.8.
=> memory is free (no-mem = full), decode+fix ~4.7 us over the MMA loop, per-stage sync ~3.5, and the split-K reduce costs
   ~5 us in ANY form (global slabs or DSMEM: it is handshake latency + the slowest split, not bytes). Best case = no-epi 15.95
   + the cheapest reduce measured (~2.9) ~ 18.9 vs cuBLAS 20.6: <= ~1.7 us on v16 qkv (~0.07% score). DEAD below gate.
Closes the dense lane on my side for T=128 and T=256 (x21). Agrees with layer n10 (cluster reduce = latency) and n30/n31.

## x23 (2026-10-31 18:52Z-19:10Z): own pick, COLD CODE after the judge's L2 flush (H100, 0 judge, 0 Modal)
Idea: the 512 MB flush read also evicts every kernel's SASS from L2, so each kernel's first execution of its post-wait code
fetches instructions from DRAM (x9's i-cache mechanism, but L2-cold). The per-kernel i-cache audits on record (proj n15,
moe GS16/twm) looked at code SIZE, never at code-from-DRAM over the whole graph.
Oracle tools/fresh/x23/codewarm.py: twin states (prepare() twice, TUNE pinned off) and twin inputs; per rep flush, then
replay twin B untimed (puts all code in L2; B's data lines do not help A), then time A. Control 'warmflush' = B then a
second flush (= base, as expected). 3 rounds x 5 x 30 reps, medians, us (graph time incl. empty-graph floor):
| load | base | code-warm | delta |
|---|---|---|---|
| d1 sl  | 86.45  | 85.10  | -1.35 |
| d1 full| 105.50 | 105.16 | -0.34 |
| v1 sl  | 143.98 | 144.70 | +0.71 (B's dirty KV/activation lines write back during A: bias against warm) |
| d32 sl | 181.86 | 182.66 | +0.80 (same bias) |
=> cold code is worth <= ~1.4 us at d1 sl, ~0 elsewhere. Not buildable anyway: device code cannot read a kernel's code
address (tools/fresh/x23/fp.cu: ld.global from a __global__ function pointer = illegal memory access), so an L2 code
prefetch has no address; a dry run of post-wait code pre-wait would add more code than it warms. CLOSED.
Side check (tools/fresh/x23/det.py, same state, fresh identical inputs, 4 runs): sliding loads bitwise run-to-run; FULL loads
are not: d1 full 1-2 FP32 ulps in the stream, v1 full 1 element, d16 full ~3-6K hidden elements by 1 BF16 ulp and ~25K stream
elements by <= 0.0154 (= 1 BF16 ulp at |x| 2-4, a BF16 rounding flip from accumulation order). Benign order noise, no race
signature (max diff fixed at one ulp across runs); compare full loads with tolerances, not torch.equal.

## x24 (2026-10-31 19:00Z-19:55Z): own pick, PUBLIC REPOS per shape (0 judge, 0 Modal; tools/fresh/x24)
Web search now finds 8 public OKC repos (lead 10-29 compared board totals only). Board 19:20Z: Madhavan113 layer 2.200
(#182; public repo 721c627 of 10-29 = its 2.08-2.11 runs, 28 judge profiles), vjeux 2.102, ruwwww 2.075, theluckystrike
1.95, lee101 1.79, infraax 1.43; none of them publishes a moe/attention entry ahead of ours (attention: vjeux 2.610,
ruwwww 2.603, ours 2.596, top 3.22; moe: Madhavan 2.357, ruwwww 2.337, ours 2.32 board / 2.37 check, top 2.76).
Per-shape judge medians, top-3 runs each (cmp_profiles.py), Madhavan 10-29 (2.11) vs ours (2.17), us:
  d1 sl 77 vs 65 | d1 fu 83 vs 75 | d16 sl 131-137 vs 124-125 | d16 fu 209-214 vs 202-206 | d32 sl 153-155 vs 149-158 |
  d32 fu 294-307 vs 306-308 | v1 sl 107-115 vs 108-115 | v1 fu 117-120 vs 121-124 | v16 sl 250-265 vs 255-267 |
  v16 fu 336-348 vs 346-358 | v32 sl 323-330 vs 320-331 | v32 fu 490-494 vs 490-526.
  => no sliding shape where they lead (sliding = 74% of the weight); FULL layers d32/v1/v16/v32 they lead by ~3-12 us.
  Per-kernel (judge profiles): their attention is vjeux's one-kernel design (flat q8 = FP8 wgmma QK^T AND P@V, P in E4M3,
  in-kernel finalize; flat q1 for decode full): v32 full 184.6 med vs our _gw_verify 190.7 + _combine_bal tail; v16 full
  101.7 vs 100.7 + tail; d32 full 170 vs _gd_full 167 + _red_fin tail. Their MoE is one kernel (moe_fused): d32 85 vs our
  tk2f 95 + norm2 14; d1 28 vs our 21.
- H100 standalone attention (~/attn tools/bench.py, both 16/16 test): vjeux vs ours d16 fu 125.1/125.6, d32 fu 242.1/244.0,
  v16 fu 135.0/132.9, v32 fu 257.5/253.4, sliding ours faster by 0.1-1.3 us; geomean 2.01 vs 2.065. H100 is HBM-bound here,
  so the H200 decides: tools/fresh/x24/mk_h200_attn.sh (after 00:00Z, ~11 GPU-min).
- Value if their full-layer attention were taken over at d32/v1/v16/v32 full: <= ~0.2-0.4% score (full shapes weigh 16 of
  62 per load). FP8 P@V was measured H200 tie in ~/attn s71 (native CUDA) and s22 (ceiling -10 us all-L2); the lead/human
  decide whether copying an Apache-2.0 public kernel is in the spirit (I'd say it is legal and must be credited).

## x25 (2026-10-31 19:07Z-19:15Z): own pick, re-read of the x24 public profiles + the attention board (0 judge, 0 Modal, 0 GPU)
- Per-kernel judge rows under PDL can't be compared across teams: each row includes its wait for the one before it. Madhavan's
  one-kernel moe_fused at d32 sl is 86-100 us (9 runs) against our tk2f row of 64.8 (c5419971), so x24's "their d32 MoE 85 vs
  our 95+14" doesn't hold. Per-load totals (x24) are the only fair comparison, and they show no sliding lead.
- Attention board (ours 2.596, #1 3.22): each load is 46 sliding + 16 full, so at batch 1 the sliding latency carries the load
  (d1: 46 x 4.44 = 204 vs 16 x 13.6 = 217; floor 0.1 us). #1's lead there is most likely sliding batch-1/8 latency. In the
  layer that piece is capped by the record's free oracle (NOTES-layer n5s26: sliding attention free = d1 2.4 / v1 3.3 us H100,
  every overlap form dead) at <= ~0.6% score even if it cost nothing. Not a lane.
- Our per-load speedups (checks 84eb0719/55b9da18/0ac4a2f0/9117bd11): d1 2.55-2.61, d16 2.08-2.14, d32 1.83-1.93,
  v1 2.36-2.38, v16 1.99-2.06, v32 2.17-2.25. The weakest loads (d32, v16) are the ones the record already mapped in full
  (d32: dense GEMMs free +1.0% ceiling (proj n5s14), experts at stream rate; v16: twm consumer, nvjet T=128).
=> Nothing to build. Own list still empty.

## x26 (2026-10-31 19:26Z-19:40Z): own pick, TUNER-PICK NOISE audit (paper, 39 judge logs in ~/r-layer/.../profiles/*.log; 0 judge, 0 Modal, 0 GPU)
Question: does the prepare()-time tuner's choice add variance to judge runs / final reruns? Per (T, layout), the tuner's best
candidate vs default (its own synthetic timing): median 0.0 (T=1/8), 0.7-1.2 (T=16), 0.3 (T=32), 0.4-0.5 (T=128), 0.6-0.7 (T=256) us;
max 1.4-3.0 (T<=16), 6.6-10.6 (T=32/128/256, single logs, mostly with a 'confirm' line that re-times default within ~1 us).
So the candidates are near-ties and a wrong pick costs ~<= 1 us typical: not a variance source worth pinning (<= 0.1%). CLOSED.

## x35 (2026-10-31 21:47Z-): lead 21:50Z 'per-CTA stamps of d1 sl / v1 sl attention in-layer on v183' (H100, 0 judge, 0 Modal)
Tool: tools/fresh/x35/astamp.py <sub> '<shape>' [reps] patches a timing copy (csl, vtk, g13, g14, g30 entry; %globaltimer to a
fixed device buffer; a stamp that waits for a load reads the timer under a predicate on the loaded value). Stamped replay =
unstamped (105.0 vs 104.3 us d1 sl, same method). Medians of 40 cold-L2 replays, us from the first qkv CTA's entry:
| d1 sl hop (csl<1,bf16,false>, 32 CTAs) | t (med / max)  | hop us |
| qkv g13 last CTA end (132 CTAs end 11.3..14.2) | 14.2 max | - |
| csl entry (plain launch after g13)     | 15.3-15.6      | 1.3 launch gap |
| page table / q arrived                 | 16.5-16.9      | 1.2 first loads (cold code + L2 trip) |
| KV arrived (KV_PF warm in L2)          | 16.75-17.1     | 0.3 |
| QK + PV done                           | 17.3-17.7      | 0.55 |
| premerge + cluster barrier             | 17.6-18.0      | 0.35 |
| merge rx ok -> out stored              | 17.9 -> 18.8   | 0.9 (incl. a stamp fence) |
| o g13 x arrived (PDL wait release)     | 19.6-20.0      | 0.85 |
| o g13 end                              | 22.5-22.9 (max 23.6) | |
| v1 sl hop (vtk<1,bf16,true> PW, 256 CTAs) | t (med / max) | hop us |
| qkv g14 end (96 CTAs end 13.0..15.9)   | 15.9 max       | - |
| vtk entry                               | 14.8-15.3     | (no early start: 16-CTA clusters don't fit beside g14) |
| page table arrived / PDL wait ok       | 16.8 / 17.1    | 1.8 first loads; wait 1.2 after g14's last CTA |
| q arrived / KV arrived                 | 17.7 / 18.2    | 0.7 / 0.45 |
| QK+PV / premerge+cluster / merge rx    | 18.8 / 19.2 / 19.8 | 0.65 / 0.4 / 0.6 |
| out stored                             | 20.8           | 1.0 |
| o g30 entry                            | 16.0           | (PDL, waits) |
Two biggest hops: d1 = the FRONT (qkv end -> csl's q in registers 2.7 us: launch 1.3 + first loads 1.2) and the TAIL (merge ->
o has x: 1.8 us); v1 = first loads after entry (1.8) and merge -> store (1.0) + g14's 2.9 us CTA end spread.
- FOUND: s62's 'CSL_PW at d1 +1.3..1.7 us' was not SM crowding: the csl PW instance sets PreferredSharedMemoryCarveout 100
  (for T=16 co-residence beside g19). H100 ab_files d1 sl, 6 rounds, mirrored order, pinned tuner: base 83.59/83.86;
  PW start-trigger + carve 85.95/85.99 (s62 replica); PW + no carve: start 82.41/82.90, g13 late trigger at last issue
  82.40/82.42, after last chunk 82.65/82.01. -> -1.2..-1.4 us, start trigger is enough (no g13 change).
  Stamps of the late form: csl enters 1.9 us earlier but the griddepcontrol.wait releases ~1.9 us after qkv's last CTA (vs a
  plain launch 1.3): the gain is the code/page-table/KV loads done before the wait.
- BUILT on branch fresh: layer.py CSL_PW_TOKENS (16, 1); attn_src.py carveout only for the FP32-qkv (T=16) PW instance.
  d1 sl/full graph_eq vs v183 BITWISE (8 inputs); pw_race d1 20/20 split 0 (joint2 positive control 20); d16 differences
  1e-3 are base-vs-base noise (same).
- Judge check 4a3434a9 (f064596): checks ok, raw 2.1959, fixed-SGLang 2.1629 (v183 662fa349 2.1576); d1 sl 64.4 vs v181-v184
  checks 65.2-67.8 (consistent with -1.3, within noise); v32 sl 340.3 = slow draw on an untouched shape. Handoff posted.
- Not built (under the 1 us gate): v1 vtk early start (16-CTA clusters wait for free GPCs beside g14; bound ~0.5 us: KV is
  already in by ~1 us after the wait), flag-spin instead of the PDL waits (record: BND_SPIN lost, n5s14).
- Next: H200 A/B after 00:00Z: tools/fresh/x35/mk_h200.sh (d1 sl only, ~3-4 GPU-min).

## x36 (2026-10-31 22:22Z-22:35Z): d1 sl attention TAIL hop (csl store -> o gemv13 sees x), lead 21:50Z lane. H100, 0 judge, 0 Modal
- csl already triggers dependents at entry, so o's 528 CTAs are resident with their W prologue in registers; the x35 0.85 us
  'store -> o has x' hop is csl grid completion -> griddepcontrol.wait release.
- Oracle (wrong output, timing bound): o g13 without griddepcontrol.wait (/tmp/fx36/nowait.py = v183+CSL_PW submission with
  the wait line removed). H100 ab_files pinned tuner, 6 rounds: d1 sl 82.75/83.50 -> 80.86 (-1.9..-2.6; overstates: o then
  ignores csl entirely and its post-wait W stream overlaps csl); d1 full tie (+0.7..+0.9, noise).
- BUILT as a prototype (tools/fresh/x36/oflag.diff, NOT kept): csl's 16 head-merge warps __syncwarp + __threadfence +
  atomicAdd(counter) after their store; o g13 (pdl == 2, counter passed as its unused pft arg) thread 0 spins until 16, the
  last o CTA resets; a counter armed but not consumed is zeroed in-stream. graph_eq d1 sl/full vs base BITWISE (8 inputs),
  0 bad replays. SASS: W prologue LDGs stay above the spin. H100 d1 sl (base ~82.1-83.1):
  ld.acquire spin +1.0/+0.1 (4-way run), +1.8 (5-way); ld.relaxed spin + one fence.acq_rel +2.5; acquire + __nanosleep(200)
  +1.1/+1.3. => DEAD: every flag form loses 1-2.5 us; spinning issue pressure is not the cause (nanosleep same). Unexplained
  (csl-side __threadfence after the store? o's early post-wait W stream competing with csl's tail?); not worth a stamp
  session at <= ~0.8 us expected (~0.1% score).
- v1 sl hops (x35): merge -> store 1.0 us is mostly the store-completion fence in the stamp; vtk early start bound ~0.5 us;
  KV_PF at v1 <= 0.3 (layer n46). With the d1 front (CSL_PW T=1) built, nothing on d1/v1 sliding attention clears the 1 us
  gate. Attention lane CLOSED for fresh.
- Next (after 00:00Z): Modal H200 A/B of CSL_PW T=1 (tools/fresh/x35/mk_h200.sh, ~3-4 GPU-min), then hand off/drop.

## x39 (2026-10-31 23:03Z-): lead 22:55Z WHOLE-LAYER HOP STAMPS at d16 sl and d32 sl (H100, 0 judge, 0 Modal)
Tool: tools/fresh/x39/lstamp.py (hooks torch load_inline: every CUDA kernel gets an RAII per-warp entry/exit stamp and a
pre/post stamp around each griddepcontrol.wait; launched Triton kernels get the same per program) + lan.py (per kernel
instance: first entry, first-arrival-at-wait 'pre', wait release, exit med/max; medians of 30 cold-L2 replays, us from the
first stamp). Run on v185 (main d837bea5) with TUNE pinned to defaults. g27 (o at T=16) left unstamped: with the guard it
dies in prepare with 'unspecified launch failure' (intermittent, o 16 / o 32 test; not the address-cache bug I also fixed;
not explained, compute-sanitizer OOMed beside another agent). Stamped replay 216-224 us vs 215-219 unstamped (d16 sl).
| d16 sl (pinned) | entry | at wait (min/med) | release (min/med/max) | exit med / max |
| qkv g24 (132)     | 0.0   | 1.3 / 3.9   | 1.7 / 4.9 / 6.9     | 14.8 / 18.2 |
| csl PW (512)      | 17.4  | 20.2 / 25.1 | 21.4 / 26.3 / 29.9  | 28.9 / 31.8 |
| o g27 (unstamped) | CUPTI ends 5.2 us after csl |
| boundary (64)     | 33.5  | 35.2 / 36.9 | 37.6                | 39.4 / 39.6 |
| pooler g24 (108)  | 37.4  | 39.7 / 42.1 | 42.8 / 43.6 / 44.9  | 61.2 / 64.1 |
| router_split (128, side) | 43.9 | - | -                 | 61.8 / 63.4 |
| routing (16, side)| 48.0  | 52.5        | 63.9                | 66.4 |
| rmsnorm_gain (16) | 65.3  | -           | -                   | 66.9 |
| tk2f (132)        | 68.1  | 70.6 / 71.0 | 71.0 / 73.6 / 74.7  | 173.3 / 178.3 |
| norm2 (16)        | 169.4 | 172.1       | 179.1               | 181.5 |
| unpool g24 (132)  | 173.3 | 175.1 / 177.7 | 182.3 / 182.4 / 189.5 | 198.1 / 203.1 |
| boundary 2 (64)   | 196.3 | 198.8       | 203.5               | 205.2 |
| d32 sl (pinned) | entry | at wait | release | exit med / max |
| qkv g30n (108)    | 0.0   | 1.6 / 6.1   | 2.0 / 6.7 / 8.0     | 17.0 / 19.8 |
| nds (48 + 208)    | 1.6 / 16.5 | 5.5 / 17.9 | 20.6 / 20.9 / 23.7 | 28.1 / 31.7 |
| combine_hb (256)  | 24.1  | 25.3        | 32.5                | 34.4 / 35.1 |
| o g30n (128)      | 27.2  | 28.8 / 33.1 | 35.8 / 35.8 / 36.3  | 39.2 / 39.7 |
| boundary (64)     | 34.8 (med 39.6: starved of SMs) | 36.3 / 41.8 | 40.4 / 41.9 / 42.1 | 43.6 / 43.8 |
| pooler g30k (104) | 41.5  | 43.2 / 44.1 | 46.5 / 46.7 / 48.3  | 66.6 / 69.2 |
| router_split (side)| 47.2 | -           | -                   | 61.8 / 63.5 |
| routing (32)      | 51.0  | 53.0        | 64.3                | 67.8 |
| rmsnorm_gain (32) | 69.9  | -           | -                   | 72.5 |
| tk2f (132)        | 71.6  | 74.5 / 75.4 | 75.0 / 77.7 / 79.3  | 160.5 / 166.9 |
| norm2 (32)        | 147.8 | 151.0       | 167.6               | 170.3 |
| unpool g30n (128) | 155.1 | 157.6 / 166.1 | 171.1 / 171.1 / 175.3 | 187.0 / 190.2 |
| boundary 2 (64)   | 171.1 / 185.5 | 172.7 / 187.2 | 190.6       | 192.1 / 192.2 |
Hops >= 0.5 us (producer's last exit -> consumer's first release), H100:
- PDL release baseline is 0.4-0.8 us (nds 0.8, combine 0.8, o 0.7, bnd 0.7, norm2 0.7-0.8, unpool 0.8, final bnd 0.4,
  routing 0.5 after router_split).
- LONG releases, 2.7-3.2 us: boundary -> pooler (d16 3.2, d32 2.7) and qkv -> csl (d16 3.2). Both consumers issue their
  pre-wait burst (pooler W ring / csl page table + KV gathers) right as the producer ends; the release comes late. Not idle
  HBM: the burst is streaming then. Bound on record: a FREE first boundary saves only 1.5 (d16) / 1.9 (d32) us (layer n14s3
  orc_nobnd), which caps any fix of the bnd -> pooler hop below the 1 us build gate after realism. Not built.
- d32 boundary programs are SM-starved (entry min 34.8, median 39.6): o g30n (128) + the early pooler g30k (104) hold the
  SMs; its release spread 40.4-42.1 is late arrival, not PDL. Inside the free-boundary bound above.
- pooler end -> tk2f release: d16 64.1 -> 71.0 (6.9 us), d32 69.2 -> 75.0 (5.8): routing tail (d16 2.5 us of routing
  after router_split), rmsnorm 1.6 / 2.6 us, tk2f launch (d16 enters 1.2 us after rmsnorm's end; d32 during it), tk2f
  pre-wait prologue 2.5 / 2.9 us. tk2f can't enter early: its 132 one-per-SM CTAs don't fit beside the pooler's 104-132
  (entry follows pooler exit in both loads). Every piece is on record (moe o7 free-prologue oracle -0.2..-0.85; layer
  n12s1/n14s3 rmsnorm/routing; IDF8 tie for the same SM reason). Not built.
- tk2f CTA end spread: d16 173.3 med -> 178.3 max (5.0 us), d32 160.5 -> 166.9 (6.4): on record (o4 last-wave K-split
  ~0.8/1.1, o7b dead, SM-weighted split dead n13s2).
- attention phase: d16 qkv end -> csl end 13.6 us for ~8 MB of page-1 random gathers (csl CTAs reach their wait at
  median 25.1 = 7.7 us after entry: the gathers before the wait are latency-bound); d32 qkv end -> combine end 15.3 us.
  Attention lane is closed for fresh (x36); listed for the attention owner.
=> No hop >= 1 us at d16/d32 sl that is not bounded on record. Nothing built.
