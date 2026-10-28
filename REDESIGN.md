# REDESIGN (moe, 2026-10-28 n6s1): where the ~18% to the top-5 has to come from

Task (lead 18:50Z): study, then name 1-2 structural differences vs twm/tk2f with a gain estimate per shape.
Board: #1 pjmullin3 2.685x, top-5 >= 2.576x, ours 4fb734c6 2.2703x. 2.685 / 2.270 = 1.183 in geomean.

## 1. Public sources: nothing specific
Web search (2026-10-28) found no write-up of any muse_moe / openkernelchallenge entry, and `okc` shows
other teams' totals only (no per-shape times, no code). So the design below comes from known public
kernels (Marlin, Machete/CUTLASS mixed-input) plus our own measurements, not from a competitor's code.

What the public kernels do, against what we already have:
| feature | Marlin (sm80, SGLang's baseline) | Machete (sm90) | ours (tk2f / twm) |
|---|---|---|---|
| offline weight reorder so a 16 B load = one thread's fragment | yes | yes | yes (_tb_pack, _POS) |
| dq: field placement via LOP3 + one mul by scale*2^bias | yes (8-ish ops / 8 w) | yes | yes, 8 int + 4 HMUL2 / 8 w (z3 search: minimal) |
| async weight stream | cp.async 4 stages | TMA, warp-specialized producer | TMA/bulk ring, producer warp |
| MMA | mma.sync m16n8k16 | wgmma RS (A = dq'd weights in regs) | tk2f mma.sync, twm wgmma RS |
| **work partition** | **striped: every SM gets the same number of (tile, k) iterations over the whole problem; partial sums fixed up in order via global locks** | persistent / stream-K tile scheduler | items = whole (expert, pass, row block) pieces of 34-55 us each, + optional K halves (TWF_SPLIT_M) |
So the one structural difference that shows up is the **work partition**: ours hands out coarse items
by a counter; both public kernels balance work at k-iteration granularity across the whole call.

## 2. Our gap per shape vs a realistic ideal (new table)
Bytes from the real routing (reference.generate_input, 100 seeds, ep8 busiest), 28.3 MB per touched
expert (fp4 w13+w2 + e4m3 scales), at 4.5 TB/s (n5s29 Modal probe: bulk rings stream 4.5 TB/s at
>= 280 MB). Ours = 4fb734c6 per-shape judge times. (The judge's "sol" column implies ~0.75x these
byte counts, so its seeds may touch fewer experts; the ratios between shapes are what matter.)

| shape | experts | pairs | MB | ideal us | ours us | ours/ideal | excess us |
|---|---|---|---|---|---|---|---|
| bs1   |  2.6 |   2.6 |  74 |  16.5 |  26.4 | 1.60 |  9.9 |
| bs8   |  8.2 |  13.6 | 232 |  51.5 |  68.0 | 1.32 | 16.5 |
| bs16  | 11.6 |  25.4 | 328 |  72.9 |  89.5 | 1.23 | 16.6 |
| bs32  | 15.9 |  49.9 | 449 |  99.7 | 120.9 | 1.21 | 21.2 |
| bs128 | 22.5 | 196   | 636 | 141.4 | 201.1 | 1.42 | 59.7 |
| bs256 | 24.5 | 392   | 694 | 154.3 | 236.3 | 1.53 | 82.0 |

Decode bs8-32 are already within 1.21-1.32x of the ideal stream. **Most of the loss is at bs128/bs256
(60 + 82 us) and at bs1 (10 us, mostly fixed cost).** If a team sat at ~1.2x ideal everywhere,
it would be ~1.20x over us in geomean: that is about the #1 gap. So the top teams most likely win on
big-M and bs1, not on decode bs8-32.

## 3. Big-M consumer floor (why the excess is NOT dq)
On record (s18/s19/s22/s27 microbenches): dq ALU and wgmma do not overlap on the SM, even across
warpgroups (RS or SS), and dequant-to-smem SS is bound by smem bandwidth (STS + LDS of bf16 A =
~138 B/clk/SM at 4.5 TB/s, over the 128 limit). So per SM, consumer time = ALU(dq) + tensor:
- ALU: twm runs dq at 43.7 weights/clk/SM (RT4 stage 750 clk / 32768 w). bs256: 24.5 x 50.3 M weights
  = 1.23e9 -> **108 us**; bs128 **99 us**.
- tensor: pairs x 50.3 M MAC at 1892 MAC/clk/SM, ~77% of peak (mb_x RS-only): bs256 **~52 us**, bs128 **~26 us**.
- Consumer floor (ALU + tensor, perfectly packed): bs256 ~160 us, bs128 ~125 us (< its 141 us stream).

So a perfectly scheduled twm would land near **bs128 ~145-150, bs256 ~165-170 us** vs 201 / 236 today.
The missing 50-70 us is NOT per-weight work (that's at its floor). It's the SMs not doing that work:
n5s25 measured busy/G 177 us vs a ~230 us call on v32 => **~25% of SM-time idle or waiting**
(start: route + gather + first data; drain 15-22 us: items are 34-55 us long, ~5 waves; hot experts
on a separate moe_hot kernel/stream, imbalance between them; moe_norm PDL tail 27-40 us in the profile).

## 4. Structural candidate R1 (big M): one cost-balanced persistent kernel ("striped" like Marlin)
- Each CTA gets a precomputed contiguous range of equal **consumer cost** over the whole call's work list
  (cost per k-chunk = ALU(dq of its weights) + tensor(N padded) from the model above, so hot and light
  experts are mixed on one SM instead of hot experts on their own kernel), crossing item boundaries at
  k-chunk granularity. Partial sums at the <= 2 boundaries per CTA reduce in fixed order (deterministic,
  FP32), like Marlin's locks. The plan is computed on device from the routing (shape-only allocation).
- gemm2 chunks of expert e are scheduled after gemm13 of e (per-expert done flags as today), and the cost
  ranges are laid out so a CTA's gemm2 work starts after its own gemm13 work finishes (no polling stalls).
- What's new vs the record: K halves / KQ (s11, n5s11), orders (o76), tails and moe_hot all keep coarse
  items and a work counter. None balanced consumer cost at chunk granularity across hot + light experts in
  one launch. proj's g30 stream-K "tax" (~5 us, n5s16) was a bf16 GEMM that is stream-bound. twm is
  consumer-bound, and there the even split is the point.
- Estimate, upper bound: at ~1.1x the section-3 floor, bs128 201 -> ~160, bs256 236 -> ~180 => board +8..9%.
  Corrected by what is on record: proj n5s11 (H200 5bbb03ca/a763ccaf) splits the bs128 excess into drain ~20-22
  + first-wave ramp ~8-10 + route/norm ~8, and measured each K piece at ~3 us extra (FP32 partial + fence +
  ticket). That's why K halves (+35% per half), KQ (-2.8 us) and TWS stealing (+3.5 us fixed) gave ~0. Striping
  keeps that tax to <= 2 pieces per CTA (<= ~6 us), against drain + ramp ~30 us => **net ~-15..-20 us at bs128
  and bs256 (-8..10% per shape) => board ~+3%**. More only if mixing hot and light chunks on one SM also
  removes the separate moe_hot imbalance at bs256.
- Risk: the 25%-idle figure is from layer v32 routings on proj's H200 timelines, not the board's bs256.
  **Gate (first, cheap): H200 per-CTA timeline of the board's bs128/bs256 call (tools/mktl_diag.py, DIAG
  check after 00:00Z, or Modal tools/h200.py): sum over CTAs of busy time vs 132 x call time. Build R1 only
  if idle >= 20% (>= 40 us at bs256).**

## 5. Structural candidate R2 (bs1-32): column-stationary gemm2 with the norm in registers
- Today: tk2f gemm2 writes y32 per (token, expert), then moe_norm2 (3.2-3.6 us kernel + launch gap)
  reloads it. A fused ticket tail (s9, n5s28) cost as much as norm2: store -> fence -> ticket -> spin ->
  reload -> reduce is the same latency chain.
- R2: CTA j owns the same output columns for EVERY expert in gemm2 (128 CTAs x 32 columns; balance is
  exact for any routing, since every expert has all 4096 rows). y stays in registers. Per-(token, expert)
  sum of squares goes in with one red.add. One grid barrier (all CTAs resident). Then each CTA scales its
  own y, sums over the token's experts in registers and stores bf16 out: no y32 round trip, no norm2.
- Estimate: -1.5..-3 us per call (norm2 + gap ~4.5 minus barrier ~1.5) => bs1 +6..12%, bs8 +2..4%,
  bs16/32 +1.5..3% => board +2..3%. Cost: 4 CTAs idle in gemm2 (~3% of phase-2 stream).
- Lower confidence than R1: n5s28 found the chain's latency, not the reload, is the cost.

## 6. Ranked plan
1. R1 gate measurement on the H200 (board bs128/bs256 timeline) after 00:00Z. If it passes, build R1 as
   a new kernel next to twm (own instantiation, no extra tw_item variants: C7512/spill rule), standalone
   board A/B, bitwise-deterministic test across 125 big-M routings (tools/seeds_big.py).
2. R2 only after R1, or if the gate fails.
Not proposed (on record dead): any dq/scale instruction cut (n5s39), SS/dequant-to-smem or pingpong
(s18/s19/s27), exact-N (n5s21), deeper rings (n5s15/n5s29), scale compression (n20).

## 6b. Premise re-test done this session (muse_moe_w4a16/tools/n6s1/ovl.py, NOTES n6s1)
- With SEPARATE per-warpgroup loops, ALU work in WG0 (FFMA, int, exact dq chains, LDS+int) overlaps RS or SS
  wgmma in WG1 by 45-100% of the smaller side. So "ALU and tensor never overlap SM-wide" (s18/s22/s27) is not a
  hardware fact: mb_x.py ran both WGs through one shared loop, and that structure alone destroys overlap here too.
- But twm's own pattern (each WG dequantizes, then issues RS wgmma on those fresh registers) still shows NO overlap
  in a clean kernel: dq 102 + MMA 99 cycles alone, 388 for one WG, 747 for two. The cause is still open (not
  operand values, not pipeline distance, not accumulator order).
- If the fresh-A penalty can be avoided (e.g. A via a path the tensor core reads differently, or a pingpong whose
  dq and MMA phases sit in different code regions), big-M consumer cost drops from ALU + tensor toward
  max(ALU, tensor): bs256 ~160 -> ~110 us of consumer, below the 154 us stream => bs256/bs128 stream-bound.
  That's worth more than R1 (bs256 -30%, bs128 -20%, board ~+7%). **Next session: find the mechanism of the
  fresh-A penalty in ovl.py before building R1.**

## 7. Honest bottom line
R1 + R2 together are worth ~+5% board (realistic), not the 18% to #1. The rest of the top teams' lead is
still unexplained. By the section-2 table it has to sit mostly in bs128/bs256 (142 us of excess between them)
and bs1. Either they overlap dq ALU with tensor work on Hopper (our microbenches say that can't be done, all
measured on the H100 with the same SM as the H200), or they lose far less to ramp/drain/tails than we do.
(Checked this session, section 6b: the premise is half wrong.) The original suggestion was: FA3-style pingpong microbench (WG0 FFMA or
dq, WG1 SS wgmma m64n128k16 with commit groups of 4 and wait<1>, no shared registers). If that overlaps
>= 30% while tools/s27m/mb_x.py's pattern doesn't, the non-overlap is specific to our wgmma issue pattern and
the dq/tensor serialization (~50 us at bs256) reopens.

## 6c. n6s2 follow-up (H100, tools/n6s2/, NOTES n6s2): section 6b's "fresh-A penalty" lead is CLOSED
- The DISTA "RS A reads ~3x" number was a ptxas-serialized kernel (DEPBAR 0x0 after every HGMMA). Pipelined RS wgmma with
  distinct A quads runs at tensor peak.
- In twm's real structure (ring.py: 384 threads, production dq, RT4) a stage costs 802/870/985 cyc at N8/N16/N32 against an
  ALU-pipe floor of 488. A 4-deep A ring (now free of C7512) changes nothing. mma.sync instead of wgmma: -12% at N8 only.
  dq at 7 ALU ops per 8 weights is minimal for every _POS (laysearch.py). So the ~+7% "stream-bound big-M" of 6b is not reachable
  by overlap tricks within one warp's dq->MMA chain.
- What's left in big-M: per-item fixed cost (H200 warm N8 item 34.4 us vs ~26 us of stage loop at 1.98 GHz: up to ~8 us per
  item, ~5 items per CTA at bs256) and the drain (~20 us). Both are R1's targets (section 4); measure the per-item part first.
