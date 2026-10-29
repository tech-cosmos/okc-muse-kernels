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
