# NOTES-moe (long log, newest at bottom)

## Session 1 (2026-10-25 ~19:18Z-)
- Submitted baseline (SGLang Marlin path) as first correct run: job 97f135310fc24612, 1.0007x.
  SGLang per shape (H200): bs1 80.2, bs16 188.0, bs32 210.3, bs8 143.3, bs128 475.9, bs256 613.4 us.
  Marlin is 94-97% of the time.
- Own kernel v1 (commit 2b6f52a/212c0bb): see STATE. Dequant trick: each u32 holds 8 codes laid out so
  outputs are q&M, (q<<3)&M, (q<<6)&M and a 3-shift combine, M=0x81C081C0 (bf16 sign + 2 exp bits +
  1 mantissa bit); value = code*2^-126 (bf16 subnormal for 0.5 is exact), times LUT[e4m3 byte] =
  bf16(sf*2^118) via fma.rn.bf16x2 (exact: <=6 significant bits), global scale * 2^8 on the FP32 acc.
  K permuted inside each 64-chunk so each lane's B fragment = 16 contiguous activations (2x LDG.128);
  scales repacked [rt][kg][lane][16B]. Brute force showed no 4-shift/rotate/prmt partition of the
  32 bits exists, so the 4th output costs 6 ops (~11 int ops / 8 elements total).
- First check attempt rejected: "starts processes" (I compiled with subprocess nvcc + ctypes).
  Switched to torch load_inline (~35 s build). check 5d8cdaec8dd34a58 1.317x, submit 3795f52627474eca 1.311x.
- Occupancy: launch_bounds(256,4) for (1,1) cut bs16 151->136, bs32 217->190, bs8 115->104 (H100);
  must keep (2,2) at minBlocks 2 (128 regs): at 151 regs it ran 431/603 vs 333/481. bs1 best at
  minBlocks 3 (46.5 vs 51). Warp-independent full-K variant tried and dropped (see STATE dead ends).
- check d82327d852c148a0 @1567586 1.424x; submit 214fcd1319a24904 1.438x.
- Tool: /tmp/moe_regs.sh <submission.py> prints ptxas regs/spills per kernel (lost on reboot: it
  cuts CUDA_SRC before the torch include and appends a dummy DISPATCH function to instantiate).

## Session 2 (2026-10-25 19:56Z-20:35Z; session-1 log times after 20:00Z are approximate)
- Big-M kernel moe_big (see STATE). Path to it: v1 (8 warps x 2 rt, runtime `if (gi < ng)`) was SLOWER
  than the old kernel (ncu: 138M instr, predicated-off HMMAs still issue, math-pipe throttle). Making NG a
  template (switch ng -> 1/2/4/8) fixed it; then 8-warp CTAs x 2/SM beat 16-warp x 1/SM (finer items ->
  smaller tail). bs256 gemm13+gemm2 431 -> 340 us local. Multi-pass bug (n > 64): gate warps read the
  red buffer (aliases stage smem) while the next pass's prologue overwrote it: needs a barrier per pass.
  check f1fe028c5ed9461c @db182ea 1.536x; submit 13641a04ff4c4ea5 1.529x.
- moe_stream decode kernel (per-lane cp.async ring of weights+x, no barriers, across items): slower,
  dropped (STATE dead ends). ncu of the old decode kernel: 57% issue, ALU 53%, ~25 instr per weight word.
- dq search (/tmp/dqsearch.py, brute force): rotl 10 / rotl 13 / multiply by 2^3+2^8+2^14 on the 8 leftover
  bits; 8 int ops per word instead of 11. bs32 gemm13 instr 60M -> 52.5M. Local unchanged, H200 decode
  +6-7%: check 58a1676977ea4978 @6a1cda0 1.586x; submit 5706bc37ac4041ff 1.589x.

## Session 3 (2026-10-25 20:33Z-)
- Decode rewrite. Path: (1) moe_sk: register-ring stream-K over (expert, rt pair, kc) units + fp32 red.add
  into zeroed Acc (route2 zeroes, act kernel does silu*up, norm2 reads fp32): first version spilled (ring of
  x + inlined NG=2/4 paths), fixed by one-group-per-segment; ~= old kernel locally, ncu proxy -7%. Still
  ~19-21 instr/word (per-unit address/x/scale overhead ~45-65 instr per 8 words).
  (2) moe_tk: TMA (cp.async.bulk + mbarrier) ring, 1 CTA/SM, consumers read smem, x in registers (warp w owns
  kc = w mod NC). First version 171 instr/unit (segment LDS checks, i % NSTG, parity math, __syncwarp
  BRA.DIV); restructured (rp-aligned CTA ranges, incremental stage/phase, all-lane arrive) -> NC=8 NSTG=8 best;
  NC=16 worse. (3) moe_tk2: + smem cross-warp reduce per rp, epilogue writes directly (silu*up bf16 / fp32 y),
  routing computed inside every gemm13 CTA -> 3 launches, no atomics, no zeroing.
- H200 proxy: ncu durations (base clock 1.45 GHz, memory unchanged): judge ~= 0.72 x ncu total (old path
  bs16 178.1 x 0.718 = 127.8 = judge). Small kernels are over-weighted in ncu (each ~4-7 us under ncu).
- Bug caught by the judge (check 1e07623a4d694c8e, bs16 recheck FAIL 1.5e-1): consumer segment = whole expert,
  so the second 8-token group of an expert with > 8 tokens reused group 0's x/off/n. Fixed (segment = (expert,
  group)); /tmp/seeds.py runs 200 routings incl. up to 32 tokens/expert: all pass.
- check 5708ff8cb783444c @c7dce17 1.888x; submit e7115a851c834420 @c7dce17 **1.897x** (bs1 31.2, bs16 105.5,
  bs32 140.2, bs8 70.7, bs128 258.8, bs256 344.9 us). That was submit 6/6 for 2026-10-25.
- Tried 16-token groups + NG=2 path in moe_tk2 (bs32 re-reads cost +9.4% bytes on typical routings, +2.3%
  with 16-groups): proxy 3-14% worse on most seeds (spills; CTA rows balanced by count, NG=2 rows cost more),
  one routing -12%. Reverted. Scripts: /tmp/cmp.py (local graph timing over seeds 9000+, run from the
  version's dir: bench.py puts '.' first), /tmp/ncu_t.sh bs cfg [seed] with DIR=<dir of a submission.py>.

## Session 4 (2026-10-25 21:13Z-)
- Big-M rewrite. (1) moe_tb (commit ac17df8, opt-in TB_RT): moe_tk2-style TMA ring (1 CTA/SM, 8 consumer warps +
  producer warp, items = expert x 64-token pass x 8 warps x RT 16-row tiles, weights repacked per item so a stage
  is ONE bulk copy, x = one 128 B bulk copy per token row) + mma.sync: correct but slower (local bs128 299 / bs256
  402 vs old 259/332): consumers latency-bound (2 warps/SMSP, tensor 21%), 64 tiny x copies per stage.
  Gotcha: 288-thread CTAs get 168 regs max (allocation rounds to 4-warp groups); __maxnreg__(224) -> launch fails.
  384 threads + setmaxnreg (producer WG dec 40, consumers inc 232) works, BUT ptxas still budgets the whole
  function at 168 for wgmma: RT=4 (128 acc + A ring) => "C7512 wgmma serialized due to insufficient registers"
  (check ptxas -v output for C7512!).
- (2) moe_tw (wgmma): A = dq() output straight from registers (wgmma's RS A fragment = mma.sync m16n8k16 A
  layout, so the weight packing is unchanged), B = x via TMA tensor map (box 64 k x {8,16,32,64} rows, SW128,
  desc LBO 1 / SBO 1024 / layout 1<<62, +32 B per k16 step). Activations need K order matching _pack_weight's
  permutation: moe_gather writes xg[pos] (logical word (2j+h)*4+t <- physical word 8t+2j+h per 64-chunk), gemm13's
  epilogue writes inter in that order. N = 8*ceil(nv/8) template (8 variants). RT=2, one wgmma group per k16 step
  (KG=1), A ring of 2: no serialization. KG=2 serializes, KG=4 uses local memory (runtime ring select).
  Generic-pointer bug: aligning the dynamic smem base via uintptr_t made every stage read an LD.E (generic) instead
  of LDS: 290/363 -> 244/304 local after computing the offset with smem_u32.
- Check 76515c457aff4a2a FAIL (bs128 "recheck failed on repetition 2", err 1.4): the judge clones the prepared
  state ("copies": 2) and my tensor maps (CPU bytes) held the original state's xg/interp addresses. Reproduced
  with a cloned dict (tools/clone_t.py), fixed: xg/inter allocated per call, maps encoded in the launch function
  (bench.py test now also runs a cloned state).
- check f6c0c328dcb845f6 @c39a176 1.997x; submit 1561eff81c564e01 @c39a176 **1.985x** (bs1 30.8, bs16 99.7,
  bs32 137.2, bs8 77.0, bs128 238.7 (was 258.8), bs256 280.5 (was 344.9)). H200 gained more than local H100
  (bs256 -19% vs -9% local) -- ncu proxy said -9%.
- ncu (H100 base clock) of moe_tw 218 at bs256 seed 7: gemm13 211 us (DRAM 70%), gemm2 121; memory-only variant
  (consumers just release stages): 166 / 87 => compute still not hidden; 86M instr (dequant ~55%).
- moe_tw on decode M <= 32 (TK2_MAXM experiment, not committed): proxy worse at bs8/16/32 (91/123/149 vs 86/90/131).
- K-split tail (862dc7c): last min(nitems, grid) items split into 2 K halves (unit list after the whole items),
  both halves store FP32 partials to a per-call workspace, counter in meta[68..580) (zeroed by moe_route), 2nd
  finisher reduces + epilogue. ncu: SM active 89/80% -> 93/87%, durations within noise; 4 parts worse.
  check 2c41a133406a4ed9 1.972x (bs128 249.4, bs256 292.4 vs 233/278): reverted to c39a176 (6a3a955).
- A-ring depth 4 (wait_group 3) and 2-step groups: ptxas emits DEPBAR 0 / C7512 at the 168-reg budget of
  384-thread CTAs (setmaxnreg does not raise ptxas' budget). The asm "+r" register fences are not what forces it.
- Big-M split of time (ncu, bs256 seed 7, moe_tw 218): compute-only (producer arrives without copying) gemm13 180 /
  gemm2 100 us, memory-only 166 / 87, full 210 / 118, issue ~55%. => latency-bound consumers; next session: 256-thread
  variant (see STATE ideas 1).

## Session 5 (2026-10-25 22:09Z-)
- PDL (programmatic dependent launch, cudaLaunchKernelEx + cudaLaunchAttributeProgrammaticStreamSerialization,
  helper launch_k; g_pdl bits via moe_set_pdl: 1 decode gemm2, 2 decode norm2, 4 big-M gather/tw/norm).
  Decode gemm2 as a PDL secondary needs its own routing (can't read gemm13's meta before griddepcontrol.wait):
  local H100 -2.5..-5.5 us on all decode shapes, but the JUDGE (H200) lost: check 52e5359f 1.938x (bs1 34.3,
  bs8 79.7, bs16 101.6, bs32 137.3; big-M 235/279). Per-bs A/B check 9acf8e84 1.953x: bs1 PDL off but routing
  recomputed 32.9 (vs 30.8: the recompute alone costs ~2 us), bs8 norm2-only 75.9 (vs 77.0), bs16 gemm2-only 102.9
  (vs 99.7), bs32 both 139.4 (vs 137.2); big-M (bit 4) 230/275 (vs 239/281). => gemm2 back to meta + plain launch.
  tw2 L2 prefetch of its first item's weights before the wait (twpf): no local gain, off.
- Note: moe_tk2 is 116-121 regs x 288 thr and ~94 KB smem: 2 CTAs/SM do NOT fit by registers.
- tk2_map row order (TK2_BLK per bs): for experts with > 1 token group, rows cycle groups every blk rps, so the
  group-1 re-read of the same weights comes from L2. Local bs32 (10 seeds) mean 176.7 -> 173.8 with blk 2
  (skewed seed 9002 175 -> 156; seed 9001 147 -> 154?); bs16 no change. Dequant is still redone per group.
- check 715085cb @98fa831 1.910x (decode norm2-PDL only + blk 2 at bs32): bs32 151.2 (!), bs1 35.6, bs8 79.1, bs16
  100.9, big-M 236/276. blk 2 is a big loss on the H200 (x reloads per 2 rps / L2 contention?): dead.
- Reverted decode to c39a176 behaviour (1082f4e; tk2 code keeps the switchable PDL/blk paths, off by default),
  big-M PDL kept (g_pdl 4): check 53a3755c 1.9835x (bs1 30.4, bs16 101.9, bs32 136.7, bs8 77.3, bs128 231.1,
  bs256 274.7) = tie with the board 1.985x: NOT submitted. Big-M PDL ~-2..-5 us, within judge noise (~+-2%).
- Judge noise lesson: per-shape decode times move +-2-3 us between checks of near-identical code; one check
  can't resolve <2% changes.

## Session 6 (2026-10-25 22:32Z-)
- ncu bs256 gemm13 (218): 78.4M instr, issue 57%, ~170 instr per k-chunk per warp of which 96 dequant: instruction/
  overhead-bound, not TLP-bound. NW=16 x RT=1 (4 consumer WGs, cfg 1118, same packing, gate/up joined via smem):
  115M instr, local bs128 265 / bs256 336 vs 240/301: worse (kept in the file, off).
- RT=4 was serialized (C7512) because ptxas sizes the wgmma pipeline per FUNCTION: the N=64 variant (acc 128 regs)
  poisoned all. RT=4 with 32-token passes (N <= 32, TWC::PS) compiles with DEPBAR 0x1. ncu (seed 7): bs128 13+2
  323 -> 301 us, bs256 338 -> 307; instr -10..16%; local wall mixed (247/293 vs 241/301).
- check 5e70081f @a0531b2 (TW_CFG 418): 2.017x, bs128 217.2 (2.15x), bs256 269.6 (2.35x). submit 4f0ae2c7: 2.033x
  (bs1 32.8, bs16 99.8, bs32 141.5, bs8 74.5, bs128 213.0, bs256 260.4). New best.
- ncu shows big tail imbalance with RT=4: gemm2 SM active cycles min 80K / max 146K (200 items on 132 CTAs).
- moe_twf (fa39ed7): gemm13 + gemm2 in ONE persistent launch. Work counter meta[66] hands out all gemm13 items, then
  gemm2 items (same heavy-first expert order); gemm13 consumers: fence.proxy.async + bar.sync 1,256 + threadfence +
  atomicAdd(done[e]) (meta[68+e], zeroed by moe_route); gemm2 producer: ld.acquire poll done[e] == npass*16 then
  fence.proxy.async before the TMA of inter. Deadlock-free: only running CTAs take items, gemm13 items never wait.
  First version: gated + non-gated tw_item in one function -> ptxas C7512 serialized ALL wgmma (128x DEPBAR 0x0;
  355 us vs 304). Fix: one main loop (tw_item DYN=true), epilogue picked at run time (tw_epi<GATED>).
  Spin-wait cost ~0 (NEEDK 0 experiment gave the same time). Local bs128 252 -> 236, bs256 295 -> 286.
- check cb18fe27 @fa39ed7: 2.061x (bs128 198.2 2.32x, bs256 250.4 2.51x). submit 52169da1: 2.061x.
- Decode tk2 ring depth on the judge: NSTG 12 (check 21c7a150, 1.996x) and NSTG 6 (check a2b1897e, 1.991x) both
  lose ~5% on every decode shape vs 8 (bs16 1.75/1.73 vs 1.83, bs32 1.41/1.39 vs 1.49). Local H100 -1..3%. Keep 8.
- tk2 SASS: NSTG 8 constant-folds the stage index (8 stages = 8 blocks per rp): 130 instr/stage vs 142 at NSTG 12;
  likely why 6/12 lose on the judge. Overhead per stage ~22 (mbarrier ~6, LDS+arrive 4, scale LUT ~16 incl. the 4
  extract/add). Scales as bf16 would cut the LUT but add ~11% bytes (and bends "NVFP4 stays NVFP4"): not done.
- moe_twf ncu (H100, seed 7): bs128 262 us DRAM 81%, active/elapsed 0.90; bs256 301 us DRAM 74%, 0.86.

## Session 7 (2026-10-25 23:12Z - 10-26 00:30Z)
- Calibration (H100 local): torch copy 3.0 TB/s read+write; our decode kernels reach ~2.7-2.9 TB/s on the big
  phases, so locally decode is near HBM-bound; on the H200 decode runs at ~70% of 4.8 TB/s (issue-bound, see s6).
- nsys (cold L2 by a READ flush; a zero_() flush leaves dirty lines and inflates the first kernel by ~8 us):
  bs1 gemm13 23.5 + gemm2 13.7 + norm2 4.4 (incl. PDL wait) us; bs32 117 + 62.5 + 5.9.
- moe_tk2f (104bf9c): gemm13 + gemm2 of the decode path in ONE launch: same routing prologue, ring continues
  (producer streams w2 while the consumers finish gemm13), phase-2 consumers wait on done counters. Counters live in
  the state (st["cnt"], int32[160]) and moe_norm2 re-zeroes them: a cudaMemsetAsync node cost 0.8 us + 1.8 us gap
  (ate the whole gain). H100: bs1 41.8 -> 39.2, bs8 97.6 -> 94.6, bs16 122.5 -> 118.6, bs32 173.9 -> 171.1.
  check 7747817f 2.154x (bs1 29.0, bs16 96.7, bs32 134.2, bs8 72.0, bs128 200.3, bs256 247.9).
- RACE in 104bf9c (found by a 144-routing sweep, /tmp/moe7/t16c.py; bench.py test did not catch it): with more
  dynamic smem (+64 KB, same kernel code) 1-4 of 144 routings fail (rel err 1.7-4%); always rp 0 (the first row
  after an x reload) of a gemm2 segment, wrong partial sums in some k chunks; inter itself final and correct, no
  double writes. Not fixed by: every-writer __threadfence before the publish barrier, fence.acq_rel after the spin,
  holding the producer until phase 1 ends, volatile ld.global.cg, 100 us sleep after the wait. Waiting for ALL
  gemm13 rows (one counter) still failed with ld.global.cg (LDG.STRONG.GPU), even + 100 us sleep. Passing: wait for all
  + __ldg (LDG.CONSTANT) inter loads after the acquire (CCTL.IVALL). Unexplained: smells like a load-use hazard on
  the strong loads right after a segment switch. Fix = 128b11a: one global gemm13-row counter (cnt[0] == n13) + __ldg.
  576 routings at both smem sizes + bitwise equal to the 3-launch path over 30 repeated runs; same speed.
  check 68739792 2.140x (bs1 29.1, bs16 102.0, bs32 136.6, bs8 69.4, bs128 202.2, bs256 248.7).
  LESSON: run /tmp/moe7/t16c.py-style many-routing sweeps (and a +smem stress variant) before any check of a kernel
  with inter-CTA flags; twf (big-M) uses the same publish pattern (thread-0 fence after bar.sync) but reads by TMA.
- Dead: norm fused into tk2f behind a grid barrier (CTA b does tokens b, b+grid; bs1 +3 us, all decode worse);
  L2 prefetch (cp.async.bulk.prefetch.L2) of the next 1-2 rows in tk2f (bs32 +14/+56 us locally: much worse).
- Started (not working yet): 16-token segments in tk2f (2 groups share each dequantized A fragment; the 2nd group's
  x fragments in smem, lane-private, 8 KB/warp; s_red doubled). Code in /tmp/moe7/cons2.cu; it needs the race fix
  above re-applied (its first version ran into exactly that race). Expected: bs32 -5..8% (10% of bs32 rows are
  extra token groups on average, up to 40%), bs16 -2..3%.

## Session 8 (2026-10-26 00:23Z - )
- tk2f KPS template (k chunks per warp per stage; stage = NC*KPS*1152 B), cfg TK2F = KPS*100+NSTG. Local H100 ties
  (it is DRAM-bound there: ncu dram 83%, issue 47-50%); non-power-of-2 NSTG (3, 6) and the 144 KB ring (208) are
  worse locally. Instructions bs32 (ncu): 108 66.3M, 204 63.1M (-4.8%), 402 62.0M (-6.5%).
  On H200 instruction count is what counts: check 74a049c0 (204) 2.180x (bs1 28.9, bs16 96.4, bs32 129.4, bs8 67.5,
  bs128 195.6, bs256 247.0); submit 9a73c24a 2.176x (bs1 28.0, bs16 94.5, bs32 134.1, bs8 72.2, 202.1, 250.0).
- cfg 402 (KPS 4 x NSTG 2) everywhere: check 8c142202 @c59ae53 2.157x (bs1 27.7, bs16 98.2, bs32 132.0, bs8 68.5,
  bs128 202.3, bs256 248.3): tie with 204 (judge noise ~+-3% per shape).
- Timeline (tools: /tmp/moe8/mkdbg*.py inserts %globaltimer stamps; td.py prints min/median/max over CTAs), H100 bs32:
  routing prologue 1.8 us; phase-1 end 116-130 us across CTAs (34-35 rows each: the spread is per-SM speed, run-to-run
  corr ~0.6 per CTA), so every CTA waits for the slowest at the gemm2 spin; gemm2 ends 177-187. bs1: ph1 end 28-34.
- DYN tail (ff44097): S = rows/grid - D static rows, the rest from a global counter (producer pulls 1 row ahead,
  passes row ids in s_rid; phase-1 end = soft marker (-1, no stage), phase-2 end = an empty stage). H100 D=4 vs
  static: bs8 94.9 -> 91.9, bs16 119.5 -> 115.4, bs32 170.9 -> 163.5; bs1 +1.4 us (tail rows usually switch segment and
  their x load is exposed: no register room for a 2nd x set at 168 regs) -> M = 1 keeps the static kernel.
- DYN judge: check b890d69b @ff44097 (M > 1 = 5402) 2.125x (bs1 28.6, bs16 100.1, bs32 133.1, bs8 70.5, bs128 196.6,
  bs256 247.1): no gain vs static 204/402 (bs16 94.5-98.2, bs32 129.4-134.1, bs8 67.5-72.2). The H100's -3..4% comes
  from evening out per-SM DRAM speed; the H200 run is consumer-issue-bound, so SMs are already even. Defaults back to
  204 everywhere (= submitted 9a73c24a); DYN kept as cfg 5402 (TK2F_DYN).
- DYN bs1 cost root cause: consumers learned "phase over" only by waiting on the next stage's full barrier, i.e. for
  that stage's DATA (one HBM latency before the gemm13 publish). Fixed with arrive-only row-id barriers ridb[4]
  (a68b9ad): bs1 DYN 43.0 -> 40.3 (static 39.7). L1 prefetch and cp.async smem staging of the next segment's x: no gain.
- ncu (H100, base clock) 402: bs1 85 MB in 40 us (2.1 TB/s), bs32 511 MB in 181 us (2.8 TB/s): extra 8-token groups
  are L2 hits (bs32 reads exactly 18 experts x 28.3 MB). Stalls per issue bs1/bs32: long_scoreboard 1.54/0.95,
  wait 0.89/0.95, math_pipe 0.44/0.38, short_scoreboard 0.33/0.30, dispatch 0.26/0.24, barrier 0.26/0.15.

## Session 9 (2026-10-26 01:18Z - 02:35Z)
- Static tk2f timeline at bs1 (H100, read flush; tools: /tmp/moe9/mkdbg3.py + td3.py, static-path %globaltimer stamps):
  routed 2.0, first stage 4.4, 2nd row 8.0, ph1 end 20.6-25.0 (5 vs 6 rows/CTA), gemm2 go 25.5, end 30-33.6 us;
  then moe_norm2 4.4 us (nsys) + ~0.3 gap. bs32: ph1 end 116-128, gemm2 end 176-186. Phase 1 runs ~2.8-3 TB/s.
- norm2 as PDL secondary (moe_set_pdl 6): bs1 -0.4 us locally, noise elsewhere. Not pursued.
- FUSED NORM (TK2FN, moe_tk2fn_launch; still in the file, default 0): each CTA tickets cnt[4] after gemm2 (monotonic,
  generation in cnt[7] read at kernel start, last ticket zeroes cnt[0] + bumps cnt[7]: no zeroing atomics at the end),
  the last M CTAs normalize one token each (v[8][16] in registers, one pass, tw/gs in smem). Correct (240 routings).
  Lessons: (1) two-pass/__ldg reloads in the tail were rematerialized/slow (pass 2 3.2 us at bs1: 168-reg cap);
  (2) even the one-pass version's tail (ticket ~1.1 us, spin 0.3, loads+reduce 1.7, compute+store ~1.9) is about
  the same as a separate norm2 kernel. H100: bs1 39.4 -> 39.8, bs8-32 -0.5..-1.1. JUDGE check 168fa04c @39ae04f
  2.146x: bs1 30.7 (vs 27.1-28.9), rest noise. DEAD (second time: fused norm loses bs1 on the H200).
- SA (s9): e4m3 block scale -> bf16x2 by arithmetic instead of the smem LUT: __byte_perm(w, 0, 0x4k4k) * 16 +
  0x77007700 (= (e<<7|m<<4) + 238<<7, the LUT's value * 2^118), exact for positive normal scales; prepare() checks
  every local w13_sf/w2_sf byte (sign 0, exp != 0, != 0x7F) -> st["sa"], else LUT. Test weights: bytes 92..126.
  Decode tk2f (TK2SA): ncu bs32 63.1M -> 59.5M instr (-5.7%), bs8 35.1 -> 33.0M; bitwise equal to the LUT path.
  check 64c725e6 @a670c82 2.205x (bs1 27.1, bs16 94.5, bs32 128.8, bs8 71.0, bs128 203.0, bs256 246.5);
  SUBMIT 6a92803a 2.190x (bs1 28.8, bs16 93.9, bs32 128.7, bs8 67.0, bs128 197.4, bs256 248.0): new best (was 2.176).
- SA in moe_twf (TWSA, flag via moe_set_sa called in kernel_fn; moe_twf_launch signature unchanged): ncu bs128
  92.8M -> 90.1M, bs256 104.6 -> 101.5M (-3%), H100 time tie; bitwise equal. check 7c1a9ee4 @e8d47eb 2.194x
  (bs128 199.2, bs256 249.2: noise vs 197-203 / 246-250). Kept on main, not submitted.
- SASS census (tk2f 204, gemm13 row loop, per warp per row = 64 HMMA): 1391 instr; dq = 12 x 64 = 768 (LOP3 5,
  SHF 2, IMAD 1, HMUL2 4 per dq), HMMA 64, the rest ~560: LUT lookups (now gone with SA), LDS 78, ~137 extra IMAD
  (addresses), LEA 43, CS2R 34, reduce/epilogue. Tool: python script in /tmp/moe9 (loops = backward BRA with >= 8 HMMA).
- Judge routing stats (reference.generate_input seeds 1-3): bs32 grp8 = experts + 0..1 (16-token segments would save
  little: idea 0 is weak), bs128 passes32 = experts + 1..2, bs256 experts + 2..6.

## Session 10 (2026-10-26 02:27Z - )
- Target per lead (01:45Z): tk2f at the LAYER's T=16/32. Layer routing dump: /tmp/moe10/dump_routes.py (run in r-layer:
  reference.ref_front + topk -> local ids, 5 seeds x 6 shapes -> /tmp/moe10/routes.pt). tools/layer_ab.py times kernel_fn on
  those routings (cold L2 graph replay; SUB=<module> to time another copy, ENV/VS to A/B a flag); tools/ncu_layer.py = ncu driver.
  H100 baseline (main a670c82+): d1 36.8, v1 87.1, d16 89.2, d32 112.6 us (incl. norm2).
  Layer d32 routing: 9-11 experts, hot 13-20 tokens (8-token groups = experts + 2..3); d16 6-10 experts, hot 6-13.
- SASS census tk2f 204 SA gemm13 row body (per warp per row, 1122 instr): dq 768 (69%), HMMA 64, SA scale 64 (PRMT+LEA),
  overhead ~226 (LDS 40, IMAD ~42, LEA 6, VIADD 19, FADD 17, sync/branch). The "560 overhead" of s9 was mostly segment-change
  code + LUT. => only less dq per weight byte (shared dequant across tokens) can cut much more.
- GS16 (16-token segments, tk2g_row/tk2g_consume, cfg +10000; 2nd group's x in smem 8 KB/warp, s_red doubled): bitwise equal
  to GS8 (tools/gs_ab.py, 310 routings) after the race fix below. ncu instr -8..-17% on layer d16/d32, BUT time +6..18 us
  (H100 ncu) and judge bs32 135.7 vs 128.7 us (check b25182e8). Timeline (globaltimer per CTA, d32 seed 99): phase 1 ties,
  phase 2 per-row time +40% (median CTA, even single-group rows), ends +5..8 us. Weighted static split (2-group rows weigh
  (8+w)/8, tk2g_split, cfg digit) made it WORSE (w 2..6), so it is not the partition. Even no-TWO routings +2% (64 KB smem).
  When L2 cannot dedup the 8-token re-reads (16 experts x 16 tokens, T=32), GS16 wins big: DRAM 850 -> 457 MB, 307 -> 187 us.
  Off by default (TK2G = 0); code kept. Unexplained: why phase-2 rows are slower in GS16.
- RACE FOUND (the s7 "first gemm2 row after an x reload" race, root cause): consumers did mbar_arrive(empty) right after
  ISSUING the ring LDS of a stage. Under load (first row after a segment change, 16 x LDG.128 per thread in flight) the
  arrive completes before some LDS have read smem, the producer's next bulk copy overwrites the stage, and that warp gets
  wrong weights (whole-warp partials wrong in the first row of a segment; debug: per-warp partial dump + ring-read checksum
  differ between runs, x-register checksum identical). GS16's partition made it frequent (11/42 runs); the old consumer in the
  GS16 kernel failed too (2/42), so main/the layer have it latent. ld.global.cg loads for inter made it more frequent (timing),
  __ldg (s7 fix) only made it rare. FIX: arrive after the stage's dq/mma (tk2f_consume + tk2g_row): 0/180 + 0/310 + sweep 0.
  Cost: H100 +0.3 us, judge bs16 92.6 (was 93.9-94.5): free. Still early-arrive (not default paths): moe_tk, moe_tk2, tk2d (DYN).
  Lesson: never release a smem stage before its loaded registers have been USED (mbarrier.arrive does not wait for LDS).
- check b25182e8 @ea74c2b (fix + GS16 at bs32): 2.1906x (bs1 28.5, bs16 92.6, bs32 135.7, bs8 70.3, bs128 196.3, bs256 243.0).

## Session 11 (2026-10-26 03:39Z - )
- H200 judge PROFILES exist for the layer: ~/r-layer/muse_decode_layer/profiles/submission.<job>.txt (per-kernel GPU time per
  scored shape). Layer e5512e00: tk2f d1 20-27.5 us (2-3 experts; floor 13-20), d16 62-76, d32 75-104, v1 40-63; twf v16
  125-170 + moe_norm 21-45, v32 207-235 + moe_norm 34-45. moe_norm's "time" is mostly twf's TAIL: it is a PDL secondary and
  starts when the first twf CTA exits (torch profiler H100 v16: norm starts 37 us before twf ends; its own work ~4.7 us).
- Judge per-group model (standalone decode): time ~ 8 us + 6.6 us per 8-token group (= 28.3 MB at 4.3 TB/s): bs8-32 are near
  the HBM roofline per distinct expert; the layer's d16/d32 have 15-25% extra groups (hot experts).
- tk2f timeline H100 (layer d1, /tmp/moe11/mktl.py + tl.py, static path stamps): routed 1.5-1.8, first data 3.8, ph1 end
  14.6-18.2 (3 vs 4 rows/CTA), cnt ok 18.9, end 21-24.3 (2 experts). Fixed overhead ~4-5 us = routing + first-data latency +
  barrier. PDL "weights first" was already tried by layer (TK2_PDL13, tie): not redone.
- NC16 tk2f (16 consumer warps, cfg +20000; x regs halve): L2-warm proxy (tools/l2warm.py: 1 expert, warm L2 = consumer
  bound) 5-10% SLOWER than NC8 at every T (per-row reduce of 16 partials). GS16 in the same proxy: T=32 34.7 -> 27.5 us
  (pure consumer bound), so GS16's loss is specific to cold/HBM runs. Bit-tiling search: no shift+mask tiling of the
  e2m1->bf16 fields exists for 32/64/128/256-bit words, so dq's 12 instr/8 weights is minimal for this approach.
- moe_twf K-split tail (TWF_SPLIT / TWF_SPLIT_M; last gemm2 items as 2 K halves; the first finisher stores an FP32 partial
  (ticket tk[j], release flag tk[512+j]), the second adds it; self-resetting; the split starts at an expert boundary
  because moe_route orders an expert's tokens NONDETERMINISTICALLY (atomics): a token could otherwise move between a
  split and a whole pass -> 1-ulp run-to-run differences; that was the "race" I chased). Judge c1659443 (bs128 S132,
  bs256 S66): 2.179x, bs128 205.2, bs256 250.8 = tie: moe_norm (tail) 38-45 -> 27-29 us but twf +10..20 us. H100 the same
  (min CTA exit +19 us): the GPU stays throughput-bound during the tail. Kept, off.
- moe_twf RT 2 / 64-token passes for all (TWF_RT 2): H100 better everywhere (layer v16 -5, v32 -7, standalone -3/-9) but
  JUDGE 7dda218c 2.130x: bs128 224.5, bs256 267.0 (worse by ~8%). Again the H100 is not a big-M proxy.
- moe_twm (TWF_MIX = 1, ON): item shape per expert: RT 2 / 64-token passes for experts with > 32 tokens (one dequant per
  64 tokens), RT 4 / 32-token passes otherwise; prepare also packs w13b2/w2b2 (+0.85 GB, +~8 s). SASS: no spill, wgmma
  waits 0x1 in loops. Correct + deterministic (60 routings incl. split). Judge d4909e24 2.170x: bs128 206.8 (control: no
  expert > 32 tokens), bs256 247.8 (= baseline range): neutral standalone. Aimed at the LAYER's v32 (hot experts 73-98
  tokens: 34 passes of 32 -> ~24 of 64) and v16 (36-49): only a layer check can tell.

## Session 12 (2026-10-26 04:25Z - )
- NEW TOOL: H200 DIAG in prepare() (DIAG = 1 in a /tmp copy only; never submit with it): times DIAG_VARIANTS on the
  layer's routings (embedded, zlib+b64 of /tmp/moe10/routes.pt: 5 seeds x d1/v1/d16/d32/v16/v32), cold L2 (256 MB
  flush), graph replay medians, prints "diag12 ..." lines; the judge log lands in <cwd>/profiles/submission.<job>.log.
  Repeatable to ~0.5 us across checks (gs8 d16 68.5-68.6, d32 88.7-89.2): MUCH better than standalone shapes.
- H200 model of tk2f+norm2 on layer routings (3 diag checks, lstsq over 20 routings, resid <= 2.4 us):
  t = 14.1 + 6.97 * experts + 4.4 * extra 8-token groups (us). 6.97 us/expert = 28.3 MB at 4.06 TB/s (~85% of HBM
  peak). Levers: the fixed 14 us (incl. norm2 ~4.5) and 4.4 us per extra group (d32 has 2-3, d16 ~1).
- diag 0b2aed51: old GS16 +6..8 us everywhere (also d1!), DYN 5402 +3..10 (d16/d32), NC16 +1..2, TWF_MIX 1 vs 0 tie/-2.
- GS16's d1 loss = I-CACHE: ncu stall_no_instruction 923 -> 9418 at d1 (kernel 118 KB SASS vs 65 KB: 4 unrolled row
  bodies). Fix: tk2h_consume = both phases in one run-time loop (gated flag only changes x source / epilogue / RPE):
  one body per row kind. cfg +30000 = h8 (GS 8), +40000 = h16 (GS 16), flags TK2H 8/16. Bitwise = main (tools/h_ab.py).
- diag 43232617: h8 = main (88.4 vs 88.7 d32), h16 d1 fixed (31.6 vs 31.0) but d16 +4.0, d32 +7.1 (uniform over
  seeds); fit: a 2-group row costs ~6 us per 2-group segment (more than GS8's separate extra group, 4.4): static count
  split overloads the CTAs holding 2-group rows; W 2 (weight 10/8) helps a bit (d32 94.2). MIX1 v16 -3.3, v32 -4.4.
- diag 619ad6ad: deeper rings (SA builds, TK2F_RING_SA): d32 208 91.6, 206 91.1, 403 89.7, 108 93.8, 402 91.9 vs 204
  88.9; d16 all +1..3. 204 stays. Not ring-depth bound.
- PSC (TK2H_PSC 1, cfg +20000 on top): gemm2 segment j waits only for gemm13 segment j's 256 rows (cnt[64 + j]; the
  rows of a segment sit on neighbouring CTAs in both phases, so most CTAs never wait for the slowest). Bitwise = main
  (1280 runs). H100 layer: d16 89.6 -> 88.7, d32 112.1 -> 111.3, v1 87.5 -> 86.7.
- diag 6e9d6e3d (PSC + GS16 weights, MIN16 9): h8p 68.5 / 88.7 = main (68.4 / 88.5): PSC gives nothing on H200; h16p d16
  71.8 d32 94.2, W 2/4/6: 92.2/93.0/96.3 at d32, d16 72-73.5. GS16 DEAD on the H200 (4 diags, any split weight).
- TIMELINE on H200 (check 88d201b4, tools/mktl_diag.py builds /tmp/moe12/submission.py = sub_tl stamps + _diag_tl):
  stamps vs earliest CTA start (0.25 us grain), min/median/max over CTAs:
  d1: routed 1.5, first data 3.6, ph1 end 11-13.3, cnt ok 14.1, ph2 data 14.3, end 16.4-18.7; event total 28.5.
  d32 s7: routed 1.5, data 3.3, ph1 end 49.9/50.7/53.0, cnt ok 53.2, ph2 data 53.5, end 77.8/80.1/80.9; total 91.8.
  => CTAs are balanced on the H200 (ph1/ph2 end max - median 1-3 us; barrier wait ~0.5 us): splitting/balancing and
  PSC have little to win. Phase 1 streams ~4.0 TB/s (10 experts x 18.9 MB in 47 us), PHASE 2 ONLY ~3.5 TB/s (9.4 MB per
  expert, 26.5 us): same row bytes and consumer work; unexplained (next lead). ~10 us of each call is outside the tk2f
  span (graph launch + norm2 ~4.5 + gaps), 3.6 us before the first weight data (routing 1.5 + first bulk copy 2.1).
- TK2W weighted static split (extra-group rows weigh wE/8, warp-parallel tk2w_bound, cfg +70000): bitwise ok, H100
  worse (w 3: d32 207 us: on the H100 extra-group rows are NOT cheap; even w 8 = count +2 us, unexplained). Not sent to
  the judge: the H200 timeline shows no imbalance to fix. Experimental instantiations now behind #if TK2_EXP (0).

## Session 13 (2026-10-26 04:57Z - )
- tk2f PF variants (cfg 80204 + 1000 * PF; bitwise = main, tools/pf_ab.py): PF 1 = L1 prefetch (prefetch.global.L1) of the
  next segment's x one row ahead; PF 2 = gemm2's cnt acquire-spin only at the CTA's first gemm2 row (was: every segment
  start, serialized before the x loads); PF 4 = gemm2 writes y32 in ROUTE-SLOT order (row = s_srt[pos]) pre-multiplied by
  gs2 * 256, then moe_norm3 (= moe_norm2<SLOT>: no inv / gs loads, one dependent-load level less). Built: 1, 2, 3, 7.
  H100 layer routings: all within 0.5 us (DRAM-bound). H200 DIAG 1fdf6eb2 (MEAN of 5 seeds, us):
    d1  gs8 30.6/30.6(rerun)  pf1 30.9  pf2 30.8  pf3 30.9  pf7 30.4
    d16 68.4/68.3            68.7      68.5      68.9      68.1
    d32 88.3/88.4            89.4      88.6      89.8      88.9
    v1  66.3/66.6            66.7      66.7      66.6      66.4
  => prefetch LOSES (+0.3..1.1), spin-once nothing; norm3 (pf7 - pf3) -0.5..0.9 us at every decode shape. The segment
  change (x reload) is NOT what makes phase 2 slow on the H200.
- ncu L2-warm proxy (T=32, 1 expert; /tmp/moe13/warm1.py): issue active 50.6%, warp cycles per issued instr 4.40 (selected 1,
  wait 0.94, long_scoreboard 0.65, not_selected 0.47, math_pipe_throttle 0.43, barrier 0.33, dispatch 0.21, short_sb 0.16).
  ALU pipe (LOP3/SHF/PRMT/LEA, half rate) 7.0M of 13.9M instr, pipe active 51% (= issue): per mma ~8.7 ALU instr.
  SASS census (bs32 cold, per HMMA): LOP3 5.15, HMUL2 4, SHF 2.04, IMAD 1.36, BRA .89, SYNCS .76, YIELD .71 (ring spin, H100
  DRAM-bound), LEA .61, PRMT .5, LDS .49.
- DQW: dq()'s two rotations as IMAD.WIDE.U32 by 2^10/2^13 from constant memory (FMA pipe) + 3-input LOP3: warm proxy
  T=8 18.2 vs 16.8, T=32 41.4 vs 35.0 us: MUCH worse. Removed. (ALU and fmaheavy pipes are both half rate on H100:
  pipe-active % per instr equal; HMUL2 runs on fmaheavy.)
- PF & 8 = 2 accumulator chains per tile (j even/odd, summed before the reduce; FP32, not bitwise): warm proxy no change.
- H200 DIAG adfb897e (gs8 / pf4 / pf12, each twice): d1 30.4/30.4/30.3, d16 68.45/68.0/67.9, d32 88.35/87.6/87.8,
  v1 66.1/66.15/66.0. => pf4 (norm3 alone) -0.45 d16, -0.75 d32, 0 at d1/v1; 2 chains nothing.
- MAIN: TK2F_DYN = 84204 (M > 1: slot-ordered pre-scaled y32 + moe_norm3), TK2F (M = 1) stays 204 because the layer's
  T=1 n2u reimplements moe_norm2 on the old y32 layout. Bitwise = before (pf_ab), bench test 15/15.
- check adfb897e @078646e (DIAG, standalone speedup 2.177)

## Session 14 (2026-10-26 05:36Z - ~06:40Z)
- Lane = layer idle budget item (1): big-M twm at v16/v32 incl. the CTA drain. Checks: 1 left at start, used on diag
  20d0b9c5 (moe board; never submit). The sandbox expires ~19:10Z, before the 00:00Z reset: NO more moe checks ever here.
- TWM_TAIL (class 2 = gemm2 of the lightest experts at RT 2 / 32-token passes, half-size items; -nsplit encoding in
  moe_twm's last arg: tail & 4095, tail13 >> 12, order >> 24). Bitwise = main. H100 layer routings v16 -4, v32 -8 us.
  H200 diag 20d0b9c5 (MEAN 5 seeds): v16 t0 167.1 / t32 167.3 / t64 168.3 / t128 171.1 / all 171.5; v32 t0 215.3 / t32
  212.0 / t64 213.5 / t128 216.4. => -3 us at v32 only. TWM_TAIL13 (gemm13 tail): nothing (H100, H200 +2..7).
- Cost model (H100, 8 experts x T tokens, all 132 SMs): an RT 4 / N 32 expert-pass costs 8.3 us, RT 2 / N 64 12.6 us
  = ~4.0 us dq + 0.134 us per token MMA; one expert's HBM read = 8.4 us: co-bound. (Note: kernel_fn at M <= 32 goes to the
  OLD moe_tk path even with TK2_MAX = 0: don't use that to test twm at small T.)
- ncu twm v32 s7 (H100): DRAM bytes 518 MB (= bytes floor: L2 dedups repeat passes), dram 67% of peak, issue 50%,
  ALU 46%, tensor pipe 26%; stalls/issue: selected 1.0, wait .72, long_sb .56, not_sel .47, gmma .43, dispatch .40.
- TIMELINE (TWM_TL build, order bit 2, stamps in `part`; /tmp tool -> now _diag_tl in submission.py, tools/mkdiag14.sh):
  heaviest-first ordering makes the first ~40 us run at 0.7-1.5 TB/s (all CTAs on the hot experts' consumer-bound,
  L2-reuse passes), a dip at the gemm13 -> gemm2 switch, drain ~25-40 us (items ~40 us each = 1 MB at the SM's share).
  Light gemm13 items (N 8, 1.09 MB) take ~46 us = exactly the per-SM fair HBM share, also with NSTG 6 (+4%): ring depth
  is not the cap.
- TWM_ORDER bits: 1 alternate heavy/light experts (no effect: hot experts own most items), 4 pass-minor (rb-major), 8/16/24
  Bresenham merge of the "A" experts' items over the first phi*N positions of each phase (phi 1/.85/.7), 32 merge in
  gemm13 only (worse), 64 A = multi-pass (> 64 token) experts only (so v16 is untouched). First version had W = nA/phi
  (= no merge): fixed. Order 76 timeline v32: start phase 1.3-2.4 TB/s, steady 2.5-2.9, CTA end -11 us (s7).
  H100 standalone layer routings v32: o0+tail32 248.8, o76 239.9 (with tail 247: tail and merge clash), v16 = .
  H100 r-layer harness copy (/tmp/moe14/layer, tools/ab_flag.py): v32 sliding/full main(tail) 407.6/664.6, o76 401.7/655.5,
  no tail no order 414.3/668.5; o84/o92/o12 within 2-4 us of o76. MAIN = TWM_ORDER_M {256: 76}, TWM_TAIL_M {} (499bde6).
  NOT H200-verified: the layer's check is the only judge left.
- Dead: tw_item next-stage LDS prefetch (weights + scales at k16 group 3, full or x-component only): twm v32 +8 us
  (spills 194 LDL), twf (no spills) +9.5 us. The early mbar_wait inside the group pipeline hurts.
- Spill note: twm sits at the 232-register limit (22 LDL/10 STL after s14's prologue additions, all in per-item
  prologue/epilogue, timing unchanged). Any extra live value in tw_item spills.

## Session 15 (2026-10-26 06:30Z - 07:40Z) -- no new speedup; big-M characterized; tk2f FP option (off)
- No moe checks left (sandbox ends ~19:10Z before the reset). Layer took 499bde6 (o76) as its v43 (928381a).
- H200 standalone (s14 diag 20d0b9c5) v16 167 / v32 215 us vs H100 ~190 / ~240: the H200 is only ~12% faster with 1.43x
  the HBM bandwidth => big-M twm is per-SM-rate bound on the H200, not HBM bound (v16 = 509 MB at ~3.0 TB/s).
- Per-item intrinsic cost (H100, /tmp/moe15/warm2.py: 1-3 experts, 1 item per CTA, weights mostly L2): RT4 gemm13 item
  (512 rows x 4096 k, 1.09 MB): N8 36 us, N32 49 us; RT2 N64 (0.55 MB) 32.5 us. H200 fair HBM share of 1.09 MB at
  4 TB/s / 132 SMs = 36 us: so RT4 items are at/above it, RT2 items 2x above. Sum of item costs on v16 s7 = 132 us of
  all-SM time vs 127 us HBM floor: co-bound; the rest (~25 us) = start + drain (exits spread 30-45 us on H100).
- Ablations of tw_item (build flag TWX in /tmp/moe15/sub_x*.py; outputs garbage): no dq / no MMA / neither / + no wv LDS
  / + fixed weight address (L2 hits) / no x TMA, us per item (RT4 N8, RT4 N32, RT2 N64):
  full 36/49/32.5; no-dq 27.5/34.3/26.6; no-mma 34.2/39.4/23.2; no dq+mma+LDS 21.6/25.0/17.6; + same address 21.2/24.4/16.9;
  no x TMA 25.9/29.4/15.3; no wgmma fences/commit/wait 26.2/31.4/15.4. NSTG 4 or 6 = 8. => a ~17-25 us per-item floor
  from the ring itself (64 stages: ~250-400 ns per stage, ~70 GB/s per SM of TMA/bulk ingest, depth-independent);
  dq adds ~8-15, MMA ~2-15 (N32 RT4 needs ~18 us of tensor time at peak). Not one bottleneck; a bigger win needs fewer
  stages per byte (bigger/merged copies: W+S one bulk, 2 kc per stage, 3D x map) or a different item shape. Untested.
- Dead: one empty-barrier arrive per warp in tw_item (mbar count 8, lane 0 arrives, branch or predicated asm): items
  36 -> 54 us. A warp-wide mbarrier.arrive is already one op; never do this.
- tk2f FP (cfg 96204 T=1 / 97204 T>1, PF & 16, off): the producer warp routes for itself (tk2f_proute: smem-atomic counts,
  same rank/order and static split as the consumers), inits the ring, bar.arrive 3; consumers route with named barrier 4
  and bar.sync 3 before the ring. First version: producer 0.8 us earlier but consumers' x loads queued behind the
  weight burst (consumer ready 2.75 -> 4.5 us); fixed with an smem mbarrier xrdy: producer issues TK2F_EARLY = 1 stage,
  waits until all consumers issued their first x loads. Plain smem atomics instead of 8 match_any rounds: T=32 producer
  start 1.9 -> 1.3 us. Bitwise = 204/84204 (pf_ab 1240), 0/2000 repeats, sweep 0 fails, tests 15/15.
  Standalone H100 (tk2f alone, hot inputs): d1 -0.2, v1 -0.4, d16 -0.25, d32 0; absolute stamps d1 -0.4, d16 -0.8.
  IN THE LAYER HARNESS (copy of r-layer HEAD + this moe_src, tools/ab_flag.py 4 rounds): d1 +0.2/+0.85, d16 +1.5/+1.8,
  d32 +1.0/+1.1 us (sliding/full): LOSES. Not understood (layer: rmsnorm/router before tk2f, late PDL trigger patch).
  Lesson: A/B decode changes in the layer harness first (/tmp/moe15/layer: git archive of r-layer HEAD, cp moe_src,
  python build.py, ab_flag.py), not standalone.
- Dead: L2 bulk prefetch of a CTA's gemm2 rows when its producer finishes gemm13 issue (to keep HBM busy through the
  phase barrier): d1 +1.3, v1/d16 +9, d32 +14 us standalone (as s7's row prefetch). cp.async.bulk.prefetch.L2 hurts here.
- H100 tk2f stamps (hot ids/x, cold weights): routed 1.45-1.5 us after start (producer 0.5 us prologue, ids 0.16, the
  32-step rank loop 0.45 (unroll: no change), scan 0.2), consumer ready ~2.8, first weight data ~4.2-4.3 us.

## Session 16 (2026-10-26 07:33Z - ) -- H200 proxy for the expert kernels; big-M item cost re-characterized
- No moe checks left (sandbox ends ~19:10Z before the reset). All work H100-only.
- **H200 PROXY** (/tmp/moe16: sub_g.py = submission.py with `grid = getenv("MOE_GRID") ? atoi(...) : nsm` in every
  persistent launcher; layer_ab.py routings; proxy.py interleaves python-global variants): H100 with the expert kernels'
  grid capped at **88 CTAs, times x 0.73, reproduces the H200 diag numbers** on the layer routings:
  t0 v16 229.6 -> 167.6 (H200 167.1), v32 293.7 -> 214.4 (215.3); d16 93.1 -> 68.0 (68.0); v1 91.0 -> 66.4 (66.1);
  d32 127.0 -> 92.7 (87.7); d1 38.3 -> 28.0 (30.4). It also orders the H200-measured TWM_TAIL variants right
  (G88 v16 t0/t32/t64/t128 229.6/228.6/235.0/241.3 vs H200 167.1/167.3/168.3/171.1; v32 293.7/293.4/299.0/304.9 vs
  215.3/212.0/213.5/216.4), where full-grid H100 gets them backwards (v16 205.1/201.5/202.0/200.8).
  Grid sweep (main, layer routings; G132/G96/G88): d1 36.8/37.8/38.3, v1 87.1/89.4/91.0, d16 89.1/91.2/93.1,
  d32 111.6/121.5/127.0, v16 206.7/222.2/228.8, v32 238.5/277.0/298.3. Reading: the H200 runs these kernels like an
  H100 with ~88-96 SMs (same per-SM rate or ~8% slower, 1.43x the HBM): per-SM bound, as s15 suspected.
- Proxy verdicts: TWM_ORDER_M {256: 76} (main since 499bde6, layer v43) LOSES at v32: o0 293.6 vs o76 298.5 (3 rounds),
  v16 +1; o4 (pass-minor only) v32 -1 / v16 +1; o12 +4.5; o1 = o0. Told layer (COORD Requests) to drop o76.
  TWF_SPLIT (K-split tail, o0) 32: -1 both (noise), 64: +1, 128: +5.
- Dead (proxy): tw_item KG 2 (v32 +62 us) and KG 4 (5x: spills); TW_NB 4 = 4 A-fragment buffers + wait_group 3 per k16
  step (+100 us: ptxas turns every wait into DEPBAR.LE gsb0 0x0 and spills: the wgmma pipeline gets serialized).
  x tile by plain bulk copy instead of the tensor-map TMA: -0.2..2 us per item, and no x copy at all the same: the x
  copy is NOT a cost (s15's 'no x TMA 25.9' was cumulative with other ablations).
- Proxy timeline (sub_tlg = TL build + grid env, tlrun.py; G88): v32 starts at ~0.6-1.1 'TB/s' for 40-60 us (all CTAs on
  the hot experts' 64-token RT2 items, MMA-heavy), then ~2.6, drain ~30 us; o76 flattens the start but lowers the middle.
- **Per-SM item cost is intrinsic**: warm (L2-resident weights) at G16 (no L2 saturation): RT4 N8 item 34.4 us (1.09 MB:
  31.7 GB/s per SM = the H200's HBM share per SM, 4.2 TB/s over 132 SMs), N32 48.0, RT2 N64 31.7 (0.55 MB). Same at G132
  (36.5/50.3/32.5). So s15's 'ring floor 21.6 us' was aggregate L2 bandwidth (132 SMs on 3 experts), not per SM.
- Ablations at G16, warm mix (sub_y<n> = sub_tlg + mkx2.py; us per item RT4 N8 / N32 / RT2 N64): TWX 1 (no dq)
  34.6/48.7/31.6, 2 (no MMA) 34.7/48.1/31.6, 7 (no dq/MMA/weight LDS) 34.7/47.6/31.7, 15 (+ no x copy) 34.6/47.9/31.7;
  full 34.4/48.0/31.7. (TWX 32/47 builds are INVALID: mkx2's no-x patch hit moe_twf, and twm kept mkx.py's 'TWX & 32
  = bulk x' branch with a reduced expect_tx.) => at low concurrency the item time does NOT depend on the consumer at
  all: **twm's own ring delivers only ~31.5 GB/s per SM** (64 stages of W 16K + S 2K in 34.6 us = 540 ns/stage).
- **cp.async.bulk ring microbenchmark** (tools/s16/ring.py, ring3.py; 1 producer thread, 256 consumer threads that
  wait full / read 4 B / arrive empty (count 256), 1 CTA per SM): cost is per STAGE, ~270-320 ns whatever the size:
  4 KB 268 ns (15 GB/s/SM), 8 KB 275 (30), 16 KB 295 (55), 32 KB 322 (101 GB/s/SM); depth 8 = 12; L2 or HBM source
  the same at G16; +~43 ns per extra copy in a stage; empty count 8 (lane-0 arrive) = 256; producer that never waits on
  'empty' still ~256 ns/stage. twm-shaped stage (W 16K + S 2K + x 1K from 3 streams, mode 16): 297 ns/stage at G16,
  335 at G132 => ~19 us per N8 item vs twm's 34.6. THE GAP (~15 us/item, ~45%) IS UNEXPLAINED and is the lead for
  the next session: bisect microbench -> twm (384 threads + setmaxnreg, consumer arrive one stage late, per-item
  counter atomics/decode, epilogue + bar.sync, scale LDS, wgmma fence/commit/wait without MMAs, 1024-aligned stage
  stride 26 KB, tensor-map x). tk2f's H200 fit (6.97 us/expert = 30.8 GB/s/SM over 132 SMs) is suspiciously the same
  per-SM rate: check whether tk2f's ring has the same cap. If the cap is lifted, the experts can become HBM-bound on
  the H200 (v16/v32 maybe -30..50 us, decode several us/expert).
- Dead (proxy, bitwise = main on 50 big-M outputs): TWM_KPS 2 = 2 k chunks per moe_twm ring stage ([x0 x1|W0 W1|S0 S1],
  NSTG 4, 4 copies per 2 chunks instead of 6): G88 v32 294/296 vs 294.5, v16 231 vs 228; G16 warm items unchanged (34.2).
  Reverted (code not kept).

## Session 17 (2026-10-26 08:25Z - ) -- s16's "ring cap" was a measurement bug; twm is consumer-bound; moe_twq
- **s16's TWX ablation builds were all the FULL kernel**: mkx2.py inserted `#define TWX n` right before sub_tlg's own
  `#define TWX 0` (redefinition: the last wins). Every "no dq / no MMA / ring only" number in s16 = main. Fixed builds
  (/tmp/moe16/sub_z<n>.py), G16 warm items (us; RT4 N8 / RT4 N32 / RT2 N64): full 34.7/47.5/31.8, no dq 26.9/33.1/26.6,
  no MMA 34.0/39.2/23.0, no dq+MMA+LDS (ring only) 18.0/21.1/17.2, + no x 15.6/19.6/15.0. => moe_twm items are
  CONSUMER-bound; the ring alone does them in ~18 us (= the bare-ring microbenchmark). dq and MMA do not overlap
  (their costs add).
- Ring microbenchmark with consumer delay (tools/s17/ring5.py, clock64 per stage): per-stage time = consumer work +
  ~100-150 cycles (barrier wait when data is ready ~100 cyc + loop); releasing the stage one step late (as twm does)
  adds ~60 ns at G16. The ring itself is not the limit.
- ncu (G16, one launch, tools/s17/ncu_drv.py): RT4 N8 items: issue 62% busy, ALU pipe 59%, fmaheavy 35%, tensor 13%;
  2.25 warps/scheduler, stalls spread (wait .55, not_selected .51, gmma .38, long_sb .35, math throttle .34).
  RT2 N64 items: tensor 55%, issue 43%, stalls wait .95, long_sb .72 (mostly the idle producer warp), dispatch .70
  (ALU ops vs the RS wgmma reading A from the register file), gmma .45.
- dq pipe rebalancing (all bit-exact or speed-only proxies, G16 warm items): rotl via 2 IMAD (mul.lo+mul.hi) + 1 LOP3
  (bit-exact): 40.9 vs 34.7 us (IMAD.HI is costly); via one IMAD.WIDE.U32 (mul.wide): 37.3 / N32 55.5 (worse); one
  rotation wide, one SHF: 35.7 / 53.6. Speed-only (wrong values) "both shifts as IMAD.SHL": 32.6 (-6%), but no correct
  layout exists with plain left shifts (exhaustive search tools/s17/search2.py: identity + shift + (rot|mul) + mul has
  no solution; the sign bits at 15/31 block it). Dead.
- Layer v32 time split on the proxy (G88 timeline, seed 7): the 3 hot experts (97/73/50 tokens, RT2 64-token passes)
  take ~35% of all CTA time for ~17% of the bytes (8 GB/s/SM vs 31 for RT4 N8 items); v16's 2 hot experts ~2x per byte.
- **moe_twq** (new, flag TWQ / TWQ_M, off by default): moe_twm with 16 consumer warps (4 WGs) x RT 2 instead of 8 x RT 4;
  same 32 row tiles per item read from the w*b4 packing through an index remap (gated w16: gate slot w16&1, up slot
  2+(w16&1); gemm2 slots 2*(w16&1)+{0,1}; scale word = __byte_perm of the 8-warp uint2), so prepare() is unchanged.
  640 threads = 4 consumer WGs + producer WG (setmaxnreg dec 32 / inc 112). ptxas compiles the WHOLE kernel at the
  launch-bound cap (640 threads -> 96 regs; setmaxnreg does not raise ptxas's budget): N <= 32 fits (no spill, wgmma
  pipelined), N 64 does not (C7512: wgmma serialized) -> all experts run 32-token passes. 125 big-M routings OK.
  G16 warm items: N8 34.0 vs 35.0 (twm), N32 45.0 vs 47.8; issue 73% (twm 62%) but +18% instructions per byte.
  Proxy G88 (x0.73 = H200): v16 223.7 vs 230.2 (-6.5), v32 302.8 vs 298.2 (+4.6: hot experts of 90+ tokens lose
  twm's 64-token passes). => use at T=128 only (TWQ_M {128: 4}).
- Dead (s17): twq with a hot-expert class (64-token passes, 1 tile/warp + smem gate/up exchange, NSTG 7): does not fit
  96 regs (spills, serialized). 512-thread twq with an inline producer in warp 0 (to get 128 regs): first a hang
  (lane-0-only pump with __syncwarp around warpgroup MMAs: warp 0 never reached its first k chunk; fixed by running
  the producer control flow warp-uniformly), then 2x SLOWER (v16 445 vs 228, v32 600 vs 297): warp 0's polling/pump
  steals issue slots and delays WG0's MMAs, and every ring slot waits for all 16 warps. Code: /tmp/moe17/twq_inline.patch.
- Layer harness copy (/tmp/layer17 = r-layer working tree 10:51Z + tools/s17/twq_layer.patch: the s17 diff applied
  to moe_src.py (= 499bde6), "moe_twq_launch" in _moe__lib, `twq` TUNE option at T=128/256): tests 13/13. The H100
  prepare-time tuner PICKS twq at T=128 full: 469.2 default vs 454.2 twq (confirm 467.3 vs 455.5: -11.8 us full
  layer); T=256 sliding twq 426.0 vs default 433.5 but order 76 (+unpool lt) 417.7 wins. (ab_flag "_moe_TWQ = 0/4"
  shows no difference because the tuner's pick sets _moe_TWQ_M per call.) `git apply --check` OK on r-layer's tree.
- N48 single pass in twq for 33..48-token experts: C7512 too (dead).
- Proxy upper bound for one-pass hot experts (tools/s17 cap64.py: v32 routings with every expert capped at 64 tokens;
  also drops 12% of routes): 270.0 vs 296.5 proxy-us.
- **moe_twm hot class** (TWM_HOT / TWM_HOT_M, off; -nsplit bit 23; tail13 now 11 bits): experts with 65..96 tokens run
  ONE pass with one 16-row tile per warp and N 80/96 (48 acc regs: fits twm's 168-reg budget, unlike RT 2 x N 96 which
  serializes wgmma), items = half RT 2 row blocks of the w*b2 packing (contiguous 4 KB W + 512 B S per k chunk), x tile
  by a 5th tensor map (96-row box) at the stage start, W at 12 KB, S at 16 KB. Gated: up warp -> gate warp FP32 exchange
  through the item's LAST ring slot (24 KB), after `bar.sync 1, 256` (first version without it: NaNs in 33/125 big-M
  routings = the other warpgroup's wgmma still reading that slot's x tile), slot released after the exchange.
  Proxy G88 v32 3 rounds: o76 300.6 -> hot+o76 293.0; o0 295.9 -> hot 292.3; v16 unchanged (no 65..96 experts).
  Smaller than the bound because hot items stream a 12 KB x tile per 4 KB of weights (ring-bound ~300 ns/stage).
- Layer patch (tools/s17/twq_layer.patch) now carries both: TUNE twq at T=128/256, twm_hot (+ combos with order 76)
  at T=256.
- Layer copy with the combined patch: tests 13/13; H100 tuner: T=128 sliding picks twq (confirm 362.6 -> 359.6),
  full twq (469.6 -> 454.2); T=256: twq 427.9/663.0 vs default 435.3/669.0 but o76+lt:unpool 419.6/655.9 wins; twm_hot
  436.7/669.9, twm_hot+o76(+lt) 437.2/670.5 (H100 full grid: lose; the H200 proxy says -3.6..7.6 proxy-us: the judge's
  tuner decides).
- twq flag bit 3 (TWQ 12) = pass-minor order: proxy v32 307.8 vs 305.6 (TWQ 4), v16 225.3 vs 224.5. Layer patch v3
  (tools/s17/twq_layer.patch, r-moe 2d9a732) adds twq 4/12 at T=128/256 + twm_hot at T=256; layer copy tests 13/13;
  H100 tuner: T=128 sliding default 362.6 / twq4 356.1 / twq12 356.0 (confirm 363.1 -> 357.2, picks twq 12), full
  470.9 / 453.2 / 454.2 (confirm 470.7 -> 454.0, twq 4); T=256 keeps o76 (twq 429.3/663.8 vs o76 ~417/654).
- Session end 11:45Z. Next: idea -8 (hot experts > 96 tokens / x-tile streaming of the hot class), -7 (tk2f d1 fixed
  cost); no moe checks left today (sandbox ends ~19:10Z).

## Session 18 (2026-10-26 11:25Z - ~12:20Z) -- why twm's consumer is slow (microbenchmarks); no speedup; one duplicate
- DUPLICATE (dropped): moe_tk2f as a PDL secondary of the layer's pre-expert rmsnorm (my "PW": routing + producer
  before griddepcontrol.wait, consumers wait before x, x via __ldcg, rmsnorm triggers after its own wait). The layer
  built the same thing in parallel (v62 TK2FX / v63 TK2FJ) because I had not posted a Claims line: ALWAYS claim first.
  My H100 layer A/B (5 rounds) matched theirs: d1 -0.6/-1.2, v1 -0.5/-0.6, d16 -0.1/-0.9, d32 -2.1/-1.2 us. Diff kept
  in tools/s18/pw_moe_dup.diff (not on main: the layer text-patches tk2f's entry/x loads; keep their anchors).
- Also found: the layer's d16/d32 outputs are NOT bitwise reproducible across graph replays even without PDL
  (maxdiff 0.016-0.03); d1/v1 'full' rarely, 'sliding' never. Not an expert-kernel effect (told layer).
- s17's twq_layer.patch had stale layer.py hunks (would have reverted layer's v60 nu8 X); the layer caught it and
  took only moe_src + twq hunks (v61). RULE (layer's ask): post patches as a diff of moe_src.py only + a list of the
  layer.py lines to add.
- **Per-stage consumer timeline of moe_twm** (tools/s18/stl.py + stl_build.diff: clock64 per stage, consumer warp 0,
  MOE_GRID=16): consumers wait on full[] only ~120-150 cycles per stage (= the try_wait cost) and work 740-1030 cycles:
  twm items are CONSUMER-COMPUTE bound; the ring is never the limit (s16's "ring-only 17 us" ablation measured the ring
  when nothing else runs, not a bottleneck). Same with cold L2 on the layer's v16/v32 routings (wait share 0.15).
- **Microbenchmark (tools/s18/mb_wgmma.py, 1 CTA, 2 consumer WGs, a stage = 4 k16 steps x RT tiles, cycles/stage)**
  reproduces twm exactly (RS both: RT4 N8 792 / RT4 N32 1046 / RT2 N64 737 vs twm 839 / 1033 / 735):
    RT4 N8:  ALU(dq) only 750 | RS MMA only 417 | RS both 792 | SS MMA only 576 | ALU + SS 1015 | ALU + STS A + SS 1528
    RT4 N32: ALU 750 | RS 550 | RS both 1046 | SS 768 | ALU + SS 1156 | ALU + STS + SS 1812
    RT2 N64: ALU 377 | RS 512 | RS both 737 | SS 512 | ALU + SS 877 | ALU + STS + SS 1142
  (ALU-only includes 4 xor sinks per dq.) => dq ALU and wgmma DO NOT OVERLAP (sum minus ~100-250 cycles), and it is
  not RS-specific: SS (A from smem) is worse, with the STS of A much worse. Likely the accumulators' register-file
  traffic. DEAD before building: an SS / dequant-to-smem rewrite of twm.
  RT2 finer N: RS both N40 558 / N48 632 / N56 709 / N64 737 (MMA only 324/384/448/512).
- tools/s18/mb_mmasync.py: dq + legacy mma.sync m16n8k16 (B by LDS): RT4 N8 740 (vs wgmma 792), N16 874, N32 1203
  (wgmma 1046), RT2 N64 1028 (wgmma 737); mma.sync only N8 220 vs RS wgmma only 417 (tiny wgmmas ~13 cycles each).
  => mma.sync wins only N8 items, by ~7%: not worth a twm variant.
- What this means for the H200: per-SM HBM share at 4.2 TB/s = ~16 B/cycle = ~1024 cycles per RT4 16 KB stage.
  RT4 N8 (792) and N16 are under it (HBM-bound, fine), RT4 N32 (1046) at it, RT2 N64 (737 per 8 KB = 22 GB/s/SM)
  well over it: only the hot experts' 64-token passes are consumer-bound. The v16/v32 gap to the floor is
  scheduling (start phase on hot items, drain), which the layer's H200-tuned options (order 76/4, tail, twq, twm_hot)
  already cover. The levers left: fewer dq ops per weight (none found: s17 exhaustive search), or fewer MMA FLOPs
  (padding).
- DEAD (s18): TWM_NF = RT2 passes of 33-40 / 41-48 tokens at N40 / N48 in moe_twm (est. hot-pass stage -24% / -14%):
  ONE extra tw_item instantiation makes ptxas spill (LDL 15 -> 163-165), TWO serialize the whole kernel's wgmma
  pipeline (C7512: DEPBAR gsb0 0x0 136, 0x1 0). twm is at its register limit: any new N needs a separate kernel or
  replacing an existing N. Diff: tools/s18/twm_nf_dead.diff. (Untested alternative: class-1 pass size 48 instead of
  64 at T=128 only, N48 replacing N64 at compile time: est. v16 -2.6 H200 us; v32 loses.)
- Fit check, no build: tk2f (decode kernel) at v16 routings would cost ~197 H200 us (fit 14.1 + 6.97 E + 4.4 extra
  groups) vs twm's ~167: dead.

## Session 19 (2026-10-26 12:04Z - ) -- warp-specialized dequant dead; big-M loss = last-wave load balance
- **Warp-specialized dequant microbenchmark** (tools/s19/mb_ws.py, 1 CTA, cycles per twm-sized stage): 2 "T" WGs do
  LDS.128 + twm's dq + STS.128 of BF16 A into a SW128 A ring (mbarrier full/empty), 2 "M" WGs run SS wgmma on it.
  full / T only / M only: RT4 N8 2147/1031/1147, RT4 N16 2132/947/1194, RT4 N32 2206/979/1307, RT2 N32 974/636/715,
  RT2 N64 (A ring 3-4) 1092-1103/582-637/844-869 vs twm RS-both 792 / ~900 / 1046 / - / 737. Even in separate warps the
  two halves do not overlap (full ~ T + M), and SS alone is slower than RS: smem bandwidth (64 KB of A written + 64 KB
  read per RT4 stage = ~1000 cycles at 128 B/clk). With s18 this closes every "dequant to smem" design on Hopper:
  twm's RS register path stays. (NT 4 variants = 768 threads spill: ignore their numbers.)
- **Big-M per-CTA timeline on the layer's routings** (H100, sub_tl19 = main + TWM_TL + MOE_GRID; /tmp/moe19/an.py):
  items run in lockstep WAVES (inter-item gaps 0.2 us; RT4 light items ~34-37 us, RT2 hot ~34-40, N32 RT4 48-55) and
  the end is a partial last wave. Loss = last exit - (first start + sum(item time) / G):
    G88 (H200 proxy): v16 s7 20.1, s11 24.3; v32 s7 21.0, s11 17.1 us (items 480/480/600/624 = 5.45/5.45/6.82/7.09 waves)
    G132 (H100 full): v16 s7 20.8, s11 15.7; v32 s7 25.4, s11 18.3 us (3.64/3.64/4.55/4.73 waves)
  ~8-10% of twm's time, ~12-18 H200 us per v16/v32 call: the biggest single item left in the experts. The start-phase
  "low HBM rate" (hot items first) does not cost time by itself (every item is per-SM-rate bound on the H200; order
  only moves work around), so the lever is the tail: finer items in the last wave(s), sized from n_items mod grid.
  Halves only help when the last wave is <= half full (items mod G <= G/2); quarters needed otherwise.
- TWF_SPLIT_M (existing: last ns gemm2 items as 2 K halves, FP32 partial + ticket) on twm, 2 rounds x 5 seeds:
    G88 (proxy) v16 base 231.5 / s66 232.0 / s132 235.7 / s264 236.8 / all 236.5; v32 293.7 / 294.4 / 297.9 / 305.8 / 305.8
    G132 (H100) v16 base 205.8-207 / s33 204.9 / s50 205.3 / s66 202.4 / s88 202.6 / s100 203.2 / o76 206.0;
               v32 252-253.7 / s33 244.3 / s50 242.9 / s66 242.6 / s88 242.9 / s100 244.2 / o76 237.8
  Proxy and full-grid H100 disagree (the tail depends on G: the proxy's 88-CTA waves are not the H200's 132): only
  the H200 can say -> layer TUNE options. Judge s11 (twf, not twm): {128: 132, 256: 66} tie.
- New (default off, bit-identical when off): twm K-split combinable with order / hot (bit 22 of the launch code:
  bits 12-21 = split count instead of tail13; merge order maps g2 items within [0, n2f) so split items stay last).
- Bug found + fixed on the way: a K-split region that reached big/hot experts gave NaN (hot items take no K half);
  the split region now stops at the last big expert (n2f loop breaks on s_big[1][a] != 0). Old TWF_SPLIT on twm with
  large ns (s16's s264/s900 proxy runs) could hit this.
- Tests: /tmp/moe19/t.py (60 big-M routings x {main, o76+s66, hot+o76+s200, o4+s33}): 0 bad, main bitwise unchanged,
  split variants maxdiff 0.0039 (FP32 halves order); bench.py test all PASS.
- Proxy (2 rounds x 5 layer seeds; H100 us), after the change:
    G132 v16: o0 204.1 / s66 203.2 / o76 204.5 / o76s66 204.4 / o76s100 205.1 / hot+o76s66 204.7
    G132 v32: o0 250.6 / s66 246.6 / o76 238.9 / o76s66 239.1 / o76s100 242.6 / hot+o76s66 247.7
    G88  v16: o0 229.7 / s66 230.5 / o76 230.9 / o76s66 232.7 / o76s100 234.5 / hot+o76s66 232.7
    G88  v32: o0 296.7 / s66 297.7 / o76 300.4 / o76s66 294.5 / o76s100 298.0 / hot+o76s66 295.4
  => K halves move the tail by 0-4 us, sign depends on the grid (as the wave arithmetic predicts). Posted as H200 TUNE
  options for the layer's last check (twf_split 66 at T=128/256, + order 76 at T=256). Not a real fix.
- NEXT (the real fix, ~12-18 H200 us per v16/v32 call): size the tail from (items mod gridDim) in-kernel: a static
  plan in moe_twm's prologue (all CTAs agree) that turns the last r + k*G items into pieces of ~L/4 (RT 2 class-2 rows x
  K halves, or K quarters with FP32 red.global.add.v4 into an FP32 y and moe_norm reading FP32), so the last wave is
  full. Needs a per-class item cost model (H200: RT4 N8 1.0, N16 1.15, N32 1.5, RT2 N64 1.0, RT2 small 0.65 x L).
  Mind twm's register limit (any new live value spills): put the plan in smem.

## Session 20 (2026-10-26 12:45Z - ) -- grid-aware AUTO K-split tail for moe_twm (s19 NEXT, halves only)
- New (off by default, main bitwise unchanged): TWF_SPLIT(_M) = 1022 / 1023 = AUTO. moe_twm's prologue sums the item
  cost (1022: every item 1; 1023: light RT 4 items weighted by pass size N8 1 / N16 1.25 / N32 1.5), r = cost mod
  gridDim.x; if 0 < r <= G/2 it K-splits the last ceil(r) light gemm2 items (the existing s11/s19 halves; cap 66 =
  part buffer), else no split. Device-side from routing, no host sync, graph-safe. Works with order / hot (bit 22).
- Tests: /tmp/moe20/t.py (= tools/s20/t.py) 60 big-M routings x {main, a22, a23, o76+a23, hot+a22}: 0 bad, maxdiff
  0.0039 (FP32 half order). bench.py test PASS. Layer copy with 1022 forced at m > 32: bench test 13/13.
- Proxy (moe-only, 2 rounds x 5 layer seeds, H100 us; tools/s16/proxy.py with sub_g20):
    G88  v16: o0 230.1/230.8  a22 230.1/233.4  a23 233.6/233.7  o76 232.3/232.1  o76a23 233.0/232.9
    G88  v32: o0 303.9/304.1  a22 300.1/299.9  a23 302.9/303.1  o76 303.0/303.7  o76a23 302.9/302.6
    G132 v16: o0 207.7/208.5  a22 204.6/205.2  a23 205.1/205.2  o76 207.9/208.6  o76a23 208.2/208.4
    G132 v32: o0 240.9/239.9  a22 240.3/239.8  a23 242.9/242.8  o76 240.5/240.0  o76a23 243.1/242.6
  a22 is the only variant <= o0 on both grids (-0..4 us); far from the wave model's ~9 us average saving.
- Layer copy (H100 full grid, tools/s20/ab_tune.py: overrides state["tune"] twf_split at T=128/256, 3 rounds, us):
    run 1: [0] vs [1022]: v16 sl 312.2/306.2, v16 full 456.3/455.0, v32 sl 400.6/404.8, v32 full 649.6/649.5
    run 2: [66]/[1022]/[1023]: v16 sl 310.3/310.2/310.4, full 455.1/455.3/454.4, v32 sl 404.5/405.1/398.2, full 651.1/648.5/648.5
  => within the +-4 us run-to-run noise. Posted as H200 TUNE options (twf_split 1022 at T=128/256; diff
  tools/s20/twm_auto_moe_src.diff, moe_src.py only, applies to r-layer HEAD).
- Timeline of a22 (H100 G88, tools/s20/tl20.py + sub_tl20 = TWM_TL build with stamps moved past the split slots):
    v16 s7: last exit 226.0 -> 217.3, loss 22.5 -> 12.3 (split ran); s11 226.7 -> 223.2; s13 225.1 -> 228.5 (split
    ran, work/G +4 us); s17 / s99: no split (r > G/2). So halves can cut ~9 us when they run, but the split's overhead
    (work/G +1.4..4 us per call: half items cost more than half) eats about half of it, and on the H200's 132 CTAs
    the layer's v16/v32 item counts (480 / 600-624 = 3.64 / 4.55-4.73 waves) mostly have r > G/2: a22 rarely splits.
- DEAD: Q = 4 / 8 K parts (1018-1020: fill a > half-full last wave with quarters/eighths; deterministic slot-order
  sum, every part stores, last sums). Proxy (H100 us, 2 rounds): G88 v16 o0 235.1/235.7 a22 233.5/236.5 q42 241.0/241.4
  q4 241.4/240.9; G88 v32 307.0/307.0 305.7/304.2 306.6/307.0 306.8/306.8; G132 v16 207.9/208.1 206.0/206.8 207.6/207.7
  207.5/207.5; G132 v32 243.4/244.4 244.9/245.3 250.2/250.1 250.3/250.0. Quarter items cost far more than 1/4 (ring
  ramp + ticket wait + partial traffic per part). Diff tools/s20/twm_qparts_dead.diff (not on main).
  Gotcha found on the way: a runtime-Q loop with `q == hh ? acc : ldcg` in tw_item's epilogue made ptxas spill
  1636 LDL/STL in moe_twm (proxy +40 us); tools/s20/spills.sh <file.py> prints moe_twm REG/STACK/LDL+STL count of a
  build (use the .so of THAT file: S._lib().__file__; committed main = 48 LDL/STL, all in the 40-reg producer).
- Conclusion s20: the big-M tail is NOT cheaply fixable by K-splitting (halves tie, quarters lose). Remaining big-M
  levers: hot experts on their own kernel / SM subset (STATE idea -8), or making items cheaper (dq), not finer.

## Session 21 (2026-10-26 13:52Z - ) -- L2 cache hints on the expert weight streams (tk2f + twm): bitwise-free wins
- Finding (H100 ncu, tools/ncu_layer.py on the layer's routings): tk2f DRAM read bytes at d16 s7/s13 = 222.7 MB vs
  floor 198.1 (7 experts x 28.3), d32 s11 337 vs 311, s17 238 vs 226: an expert with > 8 tokens is streamed once per
  8-token group, the groups' rows (256 rows apart in R) run on other CTAs at other times, and ~half of the re-read
  came from HBM again. Routings without extra groups sit at the floor (d16 s99 283.3 vs 283.0).
- TK2F_L2H (compile-time #define, python global substituted in _lib()): tk2f_produce's weight+scale bulk copies use
  .L2::cache_hint with createpolicy evict_last / evict_first. Modes: 1 = evict_last for rows a LATER group re-reads,
  evict_first else; 2/3 = halves of that; 4 = evict_last on every row of a multi-group expert (either group's CTA may
  read a row first: mode 1 marked some first reads evict_first), evict_first else. Mode 4: ncu DRAM = floor at every
  d16/d32 routing (d16 s7 199.3, d32 s17 233.8 vs floor 226.4).
- moe-only (tools/layer_ab.py, 5 routings, cold L2, 2 rounds, H100 us): off d1 36.8/37.2, v1 87.0/87.3, d16 89.5/89.0,
  d32 111.4/112.1; mode 1 36.5, 86.1, 88.1, 108.8; mode 4 36.5, 86.0, 84.3, 109.0.
- Layer harness copy (/tmp/moe21/L = r-layer 6baba9e + tools/s21/patch_layer.py, ab_files, TUNE = {} appended, 4-8
  rounds; between-file noise ~+-2 us on d32 sliding (two copies of the same file differ by 1.7)): mode 1 / mode 4 vs off:
  v1 -3.6..-5/-3.9..-6, d16 -0.5..-2/-3..-4.3, d1 -0.3/-0.5, d32 mode 1 +2..3 / mode 4 tie, T>=128 0.
  The layer's v1 gain is bigger than the experts' own (~-1): evict_first on the expert stream keeps the layer's own
  L2-resident data (its W L2 prefetches?) alive. Posted the idea to layer/proj (Requests).
- TWM_L2H: the same in moe_twm's producer (evict_last if a later pass of the expert re-reads, else evict_first):
  layer harness v16 -0.8..-3.4 sl / -2.4..-9 full (3 runs), v32 0..+1.7 (noise). In moe_twq: v16 +7 us WORSE (tuner
  forced twq 4, 6 rounds): not kept (code reverted; twq unchanged).
- Tests: bitwise = off on the 30 layer routings (x2), bench.py test 15/15 PASS. Commits 59c312e (mode 1), 12d95a9
  (twm), b32928b (mode 4). Handoff: `python tools/s21/patch_layer.py <r-layer moe_src.py> 4 1`.
- No judge runs (moe checks 20/20 used today). H200 unverified: L2 is 50 MB on both, HBM 4.8 vs 3.35 TB/s, so the
  HBM re-read costs less time on the H200 but the same bytes; the layer's first check of the next sandbox will tell.

## Session 22 (2026-10-26 14:27Z - ~15:20Z) -- pingpong / pipe tests (dq vs wgmma overlap) and fused norm in twm: all dead
- Pingpong (FA3-style WG alternation by named barriers; tools/s22/mb_pp.py mode 7, 1 CTA, cycles per stage, s18 harness):
  first build was invalid (ptxas sank the dq into the MMA batch -> every HGMMA DEPBAR 0: check SASS); fixed by a
  2-buffer software pipeline (dq of stage s+1 after issuing stage s). RT2 N64: RS both (twm) 737, pingpong 721,
  stage-batch no-pp 695; RT4 N8: 800 / 809 / 762; RT4 N32: 1046 / 1055 / 1005. Pingpong gives nothing; stage
  batching ~-5% in the microbench but that is twm's KG=4 (dead in twm, s16).
- Pipe test (DQV env in mb_pp.py): dq without the bf16 scale multiply (int XOR only) overlaps NO better (RT2 N64 RS
  both 776, RT4 N8 873); int IMUL or f16 HMUL2 instead: same as bf16. => dq/wgmma non-overlap is not an FP16-pipe
  conflict; any ALU work in the consumer warps serializes with RS wgmma (RF/issue). Closed.
- Floor arithmetic for big-M (H200, s18 cycle counts): every expert pass costs ~377 cycles of dq ALU per 8 KB stage =
  ~5.6 whole-GPU us per pass; v32 (~20 experts, ~26 passes) dq ~146 us + MMA ~39 us (non-overlapping) = ~185 us
  compute floor vs twm 221 measured and HBM 140: twm at v16/v32 is dq-bound, not HBM-bound. Only fewer passes
  (hot-expert single pass: ~-11..16% of a hot expert, ~5-10 us at v32) or a cheaper dq (exhausted s11/s17) move it.
- TWM_FN (moe_norm fused into moe_twm's drain; tools/s22/twm_fn_dead.diff, NOT on main): per-token tickets of
  written Y rows, drain CTAs norm completed tokens with moe_norm's exact math (bitwise = main on 120 routings x 2
  replays, buffers self-reset). Moe-only layer routings, H100 full grid, 3 rounds (tools/s22/proxy.py), us:
    v1 (CTA barrier + returning atomic + queue):        v16 198.9 -> 200.9 (+2.0), v32 244.8 -> 243.3 (-1.6)
    v2 (per-warp red.release, drain in token order):     v16 +3.4, v32 -1.3
    v3 (all drain CTAs poll every ticket, claim ready):  v16 +27, v32 +88 (polling floods L2)
    v4 (per-warp atom.acq_rel + queue, s_need prologue): v16 +5.1, v32 +1.5
  moe_norm as a PDL secondary is already resident when twm ends (~3 us of real work); fusing saves at most the
  launch, and any ticket work on the items costs as much. Dead.

## Session 23 (2026-10-26 14:55Z - ~15:10Z) -- TK2F_L2H mode 4 vs mode 1 inside layer v77: tie; lane status
- Layer v77 (r-layer 39e9919) carries TK2F_L2H 1 (s21 handoff line); main has mode 4. A/B in a /tmp copy
  (/tmp/moe23/L, mode 4 = 2-line sed on moe_src.py, TUNE = {} appended to both, tools/ab_files.py, 6 rounds, A/B/A2):
  d1 88.21/88.50/88.43 sl, 108.29/108.67/108.55 full; d16 204.73/205.35/204.62, 305.82/306.70/307.46;
  d32 200.30/201.07/200.79, 430.24/430.95/431.01; v1 148.96/149.18/149.45, 161.60/161.43/161.94 (H100 us).
  => mode 4 = mode 1 within A-vs-A2 noise. No request to layer (keep mode 1).
- Budget arithmetic re-done for big-M (H200, s16/s18 numbers): light RT4 N8 items run at 31.7 GB/s/SM consumer-bound =
  the H200's per-SM HBM share; hot RT2 N64 items are at wgmma peak (512 of 737 cycles MMA). So twm ~= sum of item
  times / 132 + wave tail (~20 us, s19/s20). Without a cheaper dq (exhausted s11/s17/s22) big-M has no lever left;
  a separate N128 hot kernel saves ~5% of hot-pass time only (dq 377/2 + MMA 512 vs 737 cycles per 64 tokens).
- No judge runs (moe checks 20/20 used today; sandbox ends ~19:10Z before the reset).

## Session 24 (2026-10-26 15:14Z - 15:45Z) -- two consumer-latency ideas for tk2f, both dead (no judge runs, 0 checks left)
- Premise (s16 proxy): on the H200 tk2f/twm run like an H100 with ~88 SMs (per-SM rate bound at d32/big-M; d1/d16 much
  less: G88 vs G132 d1 +2%, d16 +4%, d32 +14%). Consumers issue ~50% with 2 warps/SMSP, so tried more latency hiding.
- PF & 32 = stage-ahead LDS: tk2f_consume loads stage q+1's weights/scales into registers (after its full wait)
  before stage q's dq/mma, within a row (cfg 95204 = 84204 + it, 98204 = 204 + it). H200 proxy (G88, layer routings,
  2 rounds, H100 us): d1 37.7 -> 40.1, d16 90.3 -> 94.2, d32 122.7 -> 126.7, v1 89.6 -> 93.9. DEAD (the earlier
  full-wait serializes the consumer behind the producer; same verdict as s14's tw_item LDS prefetch).
- TK2F_OCC 2 = 2 CTAs/SM (launch_bounds(288, 2), grid 2 x nsm, ring 74 KB per CTA): warm-L2 1 expert T=8/16/32
  19.6/22.9/34.1 -> 26.7/32.6/45.7; proxy d1 37.5 -> 46.6, d16 90.1 -> 107.5, d32 121.7 -> 145.0. DEAD (113-register cap:
  xf alone is 64 registers; s4 found moe_tk2 at 116-121 regs).
- Paper checks (no code): (a) phase-2's "12% slower per byte" (s12 timeline) is mostly accounting: phase 2 is timed to the
  median CTA end incl. the last row's unoverlapped epilogue and the 13-vs-14-row quantization (~+8% of it); the rest is
  ~1.5 us at d32 = ~0.1% layer score: not worth a check. (b) The phase barrier is already covered by the producer's
  1-row w2 prefetch (~2.4 us of HBM per CTA = the measured ~2.5 us skew). (c) A barrier-free d1 (gemm2 split-K over
  each CTA's own inter slice, FP32 partials/atomics) could save ~1.5 us at d1 (~0.3% score) at the cost of
  nondeterministic sums and a tk2f rewrite: parked. (d) f16-format dq (byte-local mask 0x8E008E00) has no 4-translate
  tiling either (same proof as the bf16 search), and would need x in f16: dead.

## Session 25 (2026-10-26 15:26Z - 15:40Z) -- paper only
- No checks left, sandbox ends ~19:10Z; H100-measurable levers exhausted (s17-s24). Valued the remaining ideas against the
  layer's score weights (STATE s25): hot-expert single-pass kernel at v16/v32 (+0.15-0.3%) is the only one worth a build;
  deterministic barrier-free d1 via per-CTA K-slices + FP32 partials is <0.15% net (partial reduce eats most of the barrier gain).

## Session 26 (2026-10-26 15:44Z - 15:55Z) -- UPF4: unpool W L2 prefetch spread over moe_twm's main phase: DEAD (H100)
- Idea: at v16/v32 twm is dq/MMA-bound (HBM ~60% busy), so let its producer lanes L2-prefetch (cp.async.bulk.prefetch.L2,
  evict_last) the first SEG bytes of every unpool row, spread over the items numbered f0..f1 % of the work counter (not in
  the drain, as layer s6 tried). Patch on a clone of r-layer f970e18: tools/s26/upf4_layer_patch.py <layer dir> (adds
  UPF4 = {T: (seg, f0, f1)} + TUNE key "upf4"); tests 13/13.
- H100 tools/ab_files.py verify, 3 rounds, TUNE = {} (A off / B 2048,10,70 = 16 MB / C 4096,10,70 = 32 MB / D 3072,20,90):
  v16 sl 311.0 / 312.4 / 319.4 / 316.4, full 453.7 / 457.3 / 462.1 / 461.0; v32 sl 407.2 / 412.5 / 417.5 / 414.0,
  full 662.2 / 665.9 / 671.7 / 668.5; v1 controls flat. Monotone loss in prefetch size.
- bench.py prof v32 sliding A vs C: twm 254.8 -> 263.9 us, the unpool-side GEMMs only -0.4 / -1.1 us: the prefetched lines
  do not survive ~500 MB of expert stream through L2 even with evict_last (twm's own evict_first hints notwithstanding).
  Gain bound ~1 us even if twm lost nothing => dead on any GPU. Not sent to layer.
- No judge runs (moe checks 20/20 used today; sandbox ends ~19:10Z).

## Session 27 (2026-10-26 15:53Z - 17:05Z) -- RFLAG (tk2f launched during pr1, spins on a routing flag): marginal, parked; RSPLIT dead
- H100 d1 sliding layer timeline (copy of r-layer 00ce840 = v84, /tmp/moe27/tl.py: per-kernel median start/end over 15 cold
  profiled replays): pr1 22.0-46.0, rmsnorm 30.8-47.0 (PDL-waiting), tk2f 46.2-80.5, n2u 76.7-86.1, boundary -> 88.0.
  pr1's 26 router blocks finish (and the last one routes) long before its pooler blocks: ids are ready ~15 us before pr1 ends,
  but tk2f launched only at pr1's end (rmsnorm TRIG2 after its wait).
- RFLAG (tools/s27/rflag_patch.py <layer dir>, edits layer.py + pr1_src.cu): pr1's routing warp release-stores a flag at moe
  st["cnt"][97] (its ticket moves to cnt[96]: call-time view of the cloned state; n2u already zeroes cnt[0..127] after tk2f);
  rmsnorm triggers at its START at T=1, so tk2f launches during pr1 (CTAs land on freed SMs) and thread 0 spins
  (ld.acquire.gpu) on the flag before the routing; the flag pointer reaches tk2f through its unused non-FN `tw` arg
  (moe_set_rflag, one-shot). Tests 13/13. Timeline: tk2f starts 32.0 (was 46.2), pr1 not slowed (ends 45.5), tk2f ends 79.0.
  ab_files TUNE {} (A = v84, B = RFLAG), 8 rounds A/B/A/B: d1 sliding 87.42/87.18/87.19/87.08, full 107.94/107.40/107.53/106.52;
  earlier 6-round A/B/A: 87.68/87.31/87.53, 108.24/107.62/107.94. => -0.1..-0.4 us (~0.05% layer score on the H200 at best).
  NOT handed off: the gain is about the launch + routing prologue only (the early CTAs' 74 KB rings hold ~2 MB in total),
  and a cross-kernel spin that hangs if the flag protocol ever breaks is not worth 0.05%. Matches layer s38's oracle
  (L2 prefetch of the experts during pr1: -1.2 us at best): HBM is saturated during pr1.
- RSPLIT (same patch, `RSPLIT = Q*1000 + W`): the first Q = 26 tk2f CTAs (on pr1's early-free router SMs) get W/100 of a
  normal CTA's gemm13 rows. d1 sliding: W 150 +3.0, 200 +6.9, 300 +10.5, 50 +3.2, 80 +1.4 us vs uniform. DEAD: any
  imbalance costs; tk2f is HBM-bound globally and the phase barrier waits for the slowest CTA.
- No judge runs (moe checks 20/20 used on 10-26; sandbox ends ~19:10Z).

## Session 28 (2026-10-26 16:12Z - 16:45Z real UTC) -- hot-expert single pass: feasibility microbenchmarks (H100, no judge runs)
- Question: STATE idea a / -8 (hot experts at v32 in ONE pass instead of 64-token RT2 passes). s17's twm_hot (1 tile/warp,
  N80/96) tied/lost in the layer's H200 tuner (v70: T=256 sliding 343.9 vs 336.6 default, full 521.2 vs 520.5): its x tile
  (N x 128 B per k chunk) streamed from L2 at 3x the weight bytes. RT2 N128 halves that ratio but needs 128 acc regs.
- tools/s28/mb_n.py (s18 harness, no memory traffic, cycles per RT-stage): dq hides UNDER the wgmma once N >= ~96:
  RT2 N64 737 | N96 821 | N104 867 | N128 1024 (= MMA only 1024 = H100 peak BF16 per SM); RT1 N128 561 per 4 KB.
- tools/s28/mb_mc.py (real streams: 8 KB W + 1 KB S per stage from HBM, distinct per CTA; N x 128 B x tile from a 2 MB
  L2-resident buffer; producer lane + 2 consumer WGs with twm's dq + RS wgmma; NSTG 6; best of 5): cycles per 8 KB stage
    G132: N8 702 (HBM-bound: ring floor 388 ns = 3.1 TB/s) N16 705 N32 718 N40 731 N64 823 N72 896 N96 997 N104 1017 N128 1143
    G88 (H200 proxy grid): N8 593 N16 633 N32 695 N40 720 N64 821 N72 895 N96 997 N104 1017 N128 1142
  => per 8 KB of a hot expert (G88): 97 tokens now N64 + N40 = 1541 -> one N128 pass 1142 (-26%); 73 tokens N64 + N16 =
  1454 -> 1142 (-21%); 128 tokens 1642 -> 1142 (-30%). The x tile from L2 at N128 (2:1 vs weights) is NOT a limit at G132.
  Estimate, layer v32 (s17 seed 7: hot experts 97/73): (399 + 312) cyc x 3072 stages/expert / 132 SMs / 1.98 GHz = ~8.4
  H200 us per v32 call before the extra launch / balance cost; net est. 5-7 us = ~0.25% layer score (v32 sl+full only;
  v16's experts are <= ~49 tokens = already one pass).
- Cluster-2 TMA multicast of x (each CTA copies half, mapa + mbarrier.arrive.release.cluster to the peer's empty barrier):
  SLOWER (N128 2179 vs 1143 cyc; ring-only 1650 vs 760): the remote arrive inside the wgmma loop makes ptxas serialize every
  HGMMA (DEPBAR gsb0 0x0 on all 8) and costs ~900 cycles per stage. Not needed anyway (x from L2 is fine). Also: mbarrier
  try_wait with .acquire.cluster + release.cluster arrives in the MC=1 build halved its ring rate (1461 vs 761 cyc): use
  plain cta-scope barriers wherever no peer CTA is involved.
- tools/s28/mb_mix.py: TWO wgmma N variants (N72 / N128 chosen per stage at run time) in one kernel serialize every HGMMA
  (1342 cyc vs 1142 for N128 alone, gsb0 0x0 on all 16). => a hot kernel is its OWN launch with ONE N (128): experts with
  65..128 tokens in a single pass (MMA pays for N128 even at 73 tokens, numbers above already assume that).
- All MC=1 builds: 0 LDL/STL at __launch_bounds__(384, 1) (ptxas budget 168 regs: RT2 N128 = 128 acc fits).
- DESIGN for the next sandbox (2-3 sessions; verify with H200 DIAG checks, never submit DIAGs):
  moe_hot<N=128> (384 thr: producer WG setmaxnreg 40 / 2 consumer WGs, RT2, NSTG ~6 of 25 KB) launched BEFORE moe_twm at
  T=256 only (shape-based): its CTAs take hot items (expert with 65..128 tokens) from their own counter in meta, gemm13 rows
  then gemm2 rows with twm's done[e] counters; moe_twm becomes its PDL secondary (hot CTAs trigger at their start or when the
  hot counter runs dry, so twm CTAs fill SMs as hot CTAs exit) and skips hot experts (route/bucket marks them); twm CTAs
  griddepcontrol.wait before exiting so moe_norm (twm's secondary) sees hot's Y. x tile = a 128-row tensor-map box of xg
  (twm has 8/16/32/64/96 boxes: add 128). Weights: reuse twm's w*b2 RT2 packing (same item rows as the RT2 class).
  With no hot expert (most T=128 routings) moe_hot exits at once: keep it off at T=128.

## Session 29 (2026-10-26 16:25Z - 16:45Z) -- moe_hot BUILT: hot experts in one N128 pass, layer v32 sliding -8 H100 us
- moe_hot<NSTG 8, SA> (submission.py, after tw_hot/moe_twm): 384 thr (producer warp + 2 consumer WGs, setmaxnreg 40/232),
  experts with 65..128 tokens (ballot over meta's active list), items = all gemm13 RT2 row blocks (HG::NRB 32 per expert),
  then gemm2 (HN::NRB 16; producer spins on done[e] == 32 like twm), own counter meta[67] (zeroed by moe_route). Stage =
  x 128-row TMA box (new TMaps m[5]) 16 KB + w*b2 weights 8 KB (+ scales 1 KB, evict_first), 25 KB x 8 stages. Consumer =
  tw_item<2,1,128,...,STRIDE,XBO=16384> (new XBO template param = weights' offset in the stage; default unchanged).
  SASS: 8 HGMMA 64x128x16 in the loop, gsb0 0x1 waits, 168 regs, spills only in the epilogue (8 LDL/STL).
- moe_twm gets `hx` (= g_hot, set by moe_set_hot from kernel_fn): skips experts with 65..128 tokens (nt = 0), does NOT
  griddepcontrol.wait at entry (moe_hot, its PDL primary, waited for moe_gather and triggers at start), thread 0 waits at
  the end so moe_norm (twm's secondary) sees hot's Y. Python: HOT_M = {M: grid} (default {}: the board's bs256 has no
  expert > 49 tokens; the layer's v32 has 2 of 73..98).
- Outputs bitwise identical to HOT off (per-column FP32 accumulation order is the same at any N): hot_t.py on the
  layer routings, hot_stress.py 48 skewed T=128/256 routings (up to 9 hot + >128-token experts): 0 bad, 0 non-bitwise.
  bench.py test 15/15. Checked that twm really skips (dropping the hot launch -> reference check fails).
- H100 standalone (layer routings, verify 32, full grid): off 243.3 | G16 336 | G24 253.6 | G28 241 | G32 235.6 | G36 238
  | G40 238 | G48 239.8 | G64 242.3 us. G16 too few CTAs (96 items x ~37 us); G32 = one item round per phase.
- H200 proxy: the old MOE_GRID=88 cap is unfair here (hot's CTAs come on top). New emulation HOT_EMU (compile-time, tests
  only): twm grid 88, its last G CTAs spin until G hot CTAs have exited (exit count in meta[67] >> 20; hot's producer masks
  & 0xFFFFF). Proxy-us: off 307.0 | g20 288.8 | g24 289.2 | g28 291.7 | g32 287.1 => -18..20 proxy-us = ~-13..15 H200 us
  per v32 call (x 0.73). Pessimistic variant (twm grid 88-G, no refill): g20 294.5.
- Layer harness copy (r-layer 8a24599 + diff, tuner on (picked twm_order 76 + lt:unpool at T=256), tools/ab_flag.py, 4
  rounds): v32 sliding off 395.2 | G32 387.4 | G24 386.8 | G40 388.9; full 646.3 | 644.0 | 644.5 | 645.5 us.
  => recommend _moe_HOT_M = {256: 32} in the layer (est. H200 -8..13 us sliding, -2..5 full = ~+0.3% layer score).
- Not done: hot at T=128 (layer v16 experts <= ~49 tokens: no hot experts, launch would cost ~2 us); weighting the hot
  grid by the number of hot experts (shape-only rule: grid can't depend on routing).
- Layer copy with _moe_HOT_M = {256: 32}: bench test 13/13, tools/scored_test.py "verify 32" 6 seeds 0 bad,
  replay_test 4 inputs x rs 3 0/8 fails. The layer tuner (skewed synthetic T=256 loads, H100) with hot on: default full
  619.6 vs 655.4 us with hot off (s29 first A/B run), and it then picks {'twf_split': 66, 'twm_order': 76}.
- COORD: Handoffs + Requests (layer: patch + 2 names + _moe_HOT_M = {256: 32}) posted 16:43Z.

## Session 30 (2026-10-26 16:44Z - 16:52Z) -- moe_hot follow-ups (no judge runs: moe checks 20/20)
- H200 proxy (HOT_EMU 32 build, MOE_GRID 88, 3 rounds x 5 layer v32 routings, H100 us): off 302.0 | hot G32 285.2 |
  + o76 285.9 | + o4 285.8 | + split66 285.6 | + split66 o76 286.4 | + tail32 287.8. With hot on, the other twm options
  tie (+0.3..1) or lose (tail32 +2.6): moe_hot already removes what o76/split were trimming (the hot experts' 2nd passes).
- DIAG_VARIANTS on main now = the moe_hot H200 A/B queue (off / h24 / h32 / h40 / h32o76 / o76 / offb). Local H100 run
  of exactly that DIAG (/tmp copy, DIAG = 1, full grid, 2 runs): v32 off 253.3 | h24 251.5 | h32 236.7 | h40 237.2 |
  h32o76 237.0 | o76 242.9 | offb 251.5 us; v16 all 200-201 (no hot experts at T=128: = off). 180/180 outputs bitwise = off.
  Next sandbox, FIRST moe check: `sed 's/^DIAG = 0$/DIAG = 1/' submission.py > /tmp/x/submission.py; okc check
  /tmp/x/submission.py --json`, then grep "diag12 MEAN" in profiles/submission.<job>.log. It settles hot G (layer v87
  runs hot 32 at T=256 with 0/24/40 as tuner candidates) on the H200 for the moe part alone.
- layer s44 (v87, r-layer 1b9743e) took moe_hot: H100 tuner hot0 vs hot32 v32 sliding +14, full +11 us.
- Re-read s12-s29 for anything H100-testable with size: nothing left (big-M tail: K-split halves tie / quarters lose /
  RT2 tail loses; dq has no cheaper decode on sm_90 (no e2m1 cvt), scale cannot leave the per-stage path; tk2f phase-2 gap
  is accounting; barrier-free d1 <0.15%). The remaining lever classes need the H200 (hot grid, order) or a design change
  of the layer (overlap expert streams with non-HBM phases), not moe-local tuning.

## Session 31 (2026-10-26 16:50Z - 17:10Z) -- housekeeping before the sandbox expires (no judge runs: moe checks 20/20)
- /tmp dies with the sandbox (~19:10Z): copied the layer routings /tmp/moe10/routes.pt (77 KB, 30 loads x 5 seeds) and
  dump_routes.py to tools/data/; tools/layer_ab.py reads tools/data/routes.pt; mktl_diag.py makes /tmp/moe12 itself.
  Sanity: layer_ab "verify 16" = 200.7 us (H100, = s30's 200-201). The DIAG path is self-contained (_DIAG_R embedded).
- Token histogram of the layer routings (local experts): v16 17-19 experts, max 36-49 tokens, 33-48-token experts 1-2
  per call, 49-64 0-1; v32 18-21 experts, 2 hot (73-98), 33-48: 0-2. => an N40/N48 pass class (separate launch, s18 said
  twm itself cannot hold another N) would touch ~1 expert per call: < 0.5 us. Dead.
- Nothing else is H100-testable with size (s30 review). Next sandbox: first moe check = the queued moe_hot DIAG.

## Session 32 (2026-10-26 17:07Z - 17:20Z) -- review only (no judge runs: moe checks 20/20; sandbox ends ~19:10Z)
- Re-read the lead's "think big" item (overlap the expert weight stream with latency-bound phases) against r-layer's notes:
  layer s38 measured the upper bound (oracle side-stream L2 prefetch of the KNOWN routed experts during the d1 pooler):
  dead, HBM already saturated there; layer s41's v1 H100 timeline shows no idle-HBM interval > 4 us before the experts;
  moe s27 RFLAG (tk2f starting 14 us earlier) gave -0.1..0.4 us. A speculative (routing-blind) prefetch of all 32 local
  experts' first rows would waste ~50% of its bytes at d16 (13-19 of 32 experts touched) on top of that: not built.
- Judge layer profile 703d3a7e (v70): tk2f d1 sl 28.7 (incl. PDL wait) / full 21.2, d16 sl 69.7 / full 44.1 us;
  moe_norm2 d16 7.9 us = mostly its PDL wait on tk2f's tail.
- Nothing changed in code. Next sandbox: first moe check = the queued moe_hot DIAG (session 30), unchanged.

## Session 33 (2026-10-26 17:24Z - 18:15Z) -- RG: moe_route + moe_gather fused (no judge runs: moe checks 20/20)
- moe_routegather (submission.py, before the torch includes; RG = 1, M > 32 only): one 256-thread CTA per token. Each CTA
  counts all routes per local expert (+ those of earlier tokens) itself with warp-aggregated smem atomics
  (__match_any_sync), so sorted position = base[e] + #earlier routes to e: STABLE (expert, route index) order, no
  cross-CTA sync. CTA 0 writes meta (same fields as moe_route); every CTA writes sroutes for its routes and copies its x
  row (moe_gather's K permutation) to each of them. PDL: wait at entry, trigger after (like moe_gather). -1 launch.
- Bug found on the way: the count loop must have a uniform trip count (round nroutes up to 256), else
  __match_any_sync(full mask) hangs the GPU when T*8 % 32 != 0 (T=33). Fixed; also handles repeated ids in a token.
- Outputs bitwise = RG 0 (tools/s33/rg_bitwise.py: layer v16/v32 routings + 24 skewed T=33..256, HOT_M {} and {256: 32}:
  0 of 68 differ; per-column FP32 order does not depend on route order). bench.py test 15/15.
- H100 layer_ab (cold L2, 5 routings, 2 rounds ABBA): verify 16 199.0/200.2 vs 200.6/200.8 (-1.1 us), verify 32
  242.6/242.9 vs 242.7/242.8 (tie: at T=256 each CTA scans 2048 ids, eating the saved launch). Layer profile 703d3a7e
  had moe_route 2.1 us serial before moe_gathern on the H200: expected H200 gain ~1-2 us at T=128, ~0..1 at T=256.
- Layer would need its moe_gathern (pre-expert rmsnorm, strided FP32 pooler rows) merged with this routing prologue:
  handoff posted as a T=128 tuner candidate (small: ~0.3% on 2 of 12 shapes).

## Session 34 (2026-10-26 18:21Z-18:27Z): notes + DIAG queue only
- Budget spent (moe checks 20/20), sandbox ends ~19:10Z, lead: push before 18:30Z. No GPU work.
- Added ("h32rg0", RG 0) / ("h32rg1", RG 1) (both HOT_M {256: 32}) to DIAG_VARIANTS["big"]: RG is a module global read
  in kernel_fn at call time, so the DIAG override works. py_compile OK; DIAG = 0 on main (board code = s33).
- Next sandbox: first moe check = DIAG run (STATE s30/s34 sections); grep "diag12 MEAN" in the job log.

## Night4 session 1 (2026-10-26 19:15Z - 20:10Z) -- NORM_EARLY: post-expert norm inside twm's drain (no judge runs)
- New sandbox okc-night4: bench test 15/15; DIAG copy (DIAG = 1) runs locally, v32 off 247.7 | h32 231.0 | h32rg1 230.5 us.
- Measured the serial cost of moe_norm after moe_twm (H100, graph with vs without the norm launch, 5 seeds):
  T=128 239.3 vs 235.9 (3.4 us), T=256 274.4 vs 267.1 (7.3 us). ncu (base clock) moe_norm alone 7.0 / 10.6 us.
- NORM_EARLY (submission.py, #define substituted in _lib(); python NORM_EARLY picks moe_norme_launch for the twm path):
  * moe_twm: s_nd2[a] = gemm2 item ends of slot a = (cum1[a+1]-cum1[a]) + max(0, cum1[a+1]-max(cum1[a], n2f)) (K-split
    halves count twice: the 2nd half writes Y before its count). gemm2 item end: bar.sync 1,256; thread 0 threadfence,
    atomicAdd(meta[100+e]) and + 1 << 20 when it was the last. moe_hot: same with HN::NRB items per expert.
  * moe_route / moe_routegather zero meta[100..131] (meta now 132 ints).
  * moe_norm got a `rdy` param (nullptr = the old griddepcontrol.wait). rdy set: threads 0-7 spin (ld.acquire.gpu,
    nanosleep 256) on the token's local experts' flags, __syncthreads, Y loads via __ldcg. Safe vs PDL: the norm grid
    launches only after every twm CTA has run its trigger (after its own wait / moe_hot's), so ids/tw are visible
    and no norm CTA can take an SM a not-yet-resident twm CTA needs.
- H100 layer routings (tools/layer_ab.py, same build, ENV=NORM_EARLY): hot off v16 195.1/195.1, v32 238.7 -> 233.6;
  hot 32 (2 runs) v32 229.8/230.5 -> 226.4/227.0, v16 tie. vs HEAD build (hot 32, ABAB): v32 231.2/230.5 -> 227.0/228.0,
  v16 195.8/199.4 -> 193.8/196.4 (noisy). => v32 -3.3, v16 0..-2 us.
- Correctness: tools/n4s1/ne_bitwise.py 0 of 640 differ (layer routings + 30 skewed T=33..257, HOT_M off/on,
  TWF_SPLIT_M off/on, 20 back-to-back graph replays each); seeds_big 0 bad (hot off and on); bench test 15/15.
- DIAG_VARIANTS big += ("h32ne0", hot 32 NE 0), ("offne0", NE 0): the 00:00Z DIAG check A/Bs it on the H200 too.
- Why v16 gains little: probably most tokens have a route to one of the last-finishing experts; not investigated.

## Night4 session 2 (19:26Z, no judge runs: moe checks 20/20 until 00:00Z) -- PDL .nc-hoist fix
- layer s52 request: moe_routegather read ids (and X) via const __restrict__ after griddepcontrol.wait = .nc loads
  ptxas may hoist above the wait. Now __ldcg for ids and X. SASS audit (cuobjdump of the built .so, every kernel with
  ACQBULK, LDG.*CONSTANT before the first ACQBULK): moe_norm also had `ids` hoisted above its wait (LDG.E.CONSTANT
  from param ids before ACQBULK) -> ids and tw via __ldcg there too. Remaining pre-wait .nc loads: w1 (moe_norm),
  lut (twm/hot/twf/tw/twq) = prepared state, safe. moe_route / r-moe tk2f have no wait (not PDL secondaries here;
  layer owns its patched tk2f). Standalone outputs unchanged (cache op only); bench test 15/15.
  Layer: its moe_norm copy (and moe_norme) needs the same __ldcg on ids/tw if it runs it as a PDL secondary.

## Night4 session 3 (2026-10-26 19:31Z - ~20:05Z, no judge runs: moe checks 20/20 until 00:00Z) -- L2 hints on the H200 DIAG
- Found while reading: the s21 L2 cache hints (TK2F_L2H / TWM_L2H; layer v73/v76) have NEVER run on the H200: the last
  moe check was 06:05Z (s14) and the layer's board run 703d3a7e is v70. The layer's first check tonight carries them blind.
- r-moe 1f3b71b: tk2f hint mode selectable at run time: python TK2F_L2H_RT (None = the #define) -> launch cfg +
  1000000 * (mode + 1) -> host static g_tk2f_l2h -> new last kernel arg l2h. Only PF & 64 instantiations read it
  (cfg 204 -> PF 64, 84204 -> PF 68, SA only; non-SA falls back to the default kernel); every other instantiation uses
  the #define as before. First try (l2h read by every instantiation) cost d32 +0.4..0.8 / d16 +0..1 H100 us vs HEAD
  (layout; 3 ABAB rounds), so default kernels stay compile-time: their SASS is now byte-identical to HEAD's (cuobjdump diff).
- H100 layer routings (tools/layer_ab.py, cold L2), mode 4 / 0 / 1: d1 36.2/37.4/36.8, d16 82.1-82.9/87.5-88.1/86.2-87.2,
  d32 105.6-105.7/110.0-110.7/105.8-106.2, v1 84.1-84.5/85.9-86.1/84.1-84.5 us. Outputs bitwise equal across all modes
  (/tmp/n4s3_eq.py: 20 decode routings x modes None/0-4). Tests 15/15. No spills (regs = HEAD +-1).
- DIAG_VARIANTS["dec"] = gs8 (mode 4) / l2h0 / l2h1: the 00:00Z DIAG check now also answers "do the hints help on the
  H200" for d1/d16/d32/v1. Local DIAG dry run OK. If l2h0 wins on the H200: tell layer to set its #define TK2F_L2H 0.
- Looked at and left (paper, all covered by earlier notes): tk2f is at ~4.06 TB/s per marginal expert on the H200 and its
  in-layer fixed cost is mostly hidden by TK2FX/TK2FJ; twm is dq-bound (s18/s22) with the drain the only lever (s19/s20).

## Night4 session 4 (2026-10-26 19:49Z - ~20:25Z, no judge runs: moe checks 20/20 until 00:00Z) -- paper only, two dead ends
- Judge weights (nvfp4.random_weights): E2M1 codes uniform over 16 patterns -> 4.0 bits, incompressible. Block scales
  (e4m3) all positive, exponent field 11..15 (2^4..2^8): counts over all 256 experts w13 103 / 2.4M / 242M / 288M / 4.6M
  (w2 alike). Scales are 1/9 of the expert byte stream.
- DEAD (paper) lossless scale compression: 5 bits (2-bit exp + 3-bit mantissa) does not fit (exp 11 appears in ~1/3 of
  the experts' w13; escapes would need a pre-SiLU fixup inside tk2f). 6 bits (3-bit exp offset): 4 fields do not pack
  into a word with byte-free extraction; decode = SHF+LOP3+IMAD (3 instr) vs today's PRMT+IMAD (2) per scale, i.e.
  ~+4% consumer instructions (104 per 64 weights today) for -2.2..2.8% bytes. Decode shapes are at ~85% HBM peak and
  partly issue-bound: net ~0, and real checkpoints' wider scale ranges would hit the fallback. Not built.
- DEAD (paper) e4m3-convert dequant (E2M1 nibble -> s00eem00 e4m3 byte = value * 2^-6, then cvt.rn.f16x2.e4m3x2):
  {2,3,4,7} has no shifted copy disjoint from itself mod 8, so the second register costs SHF+SHF+LOP3; with 4 cvt (F2FP,
  slower pipe) + 4 HMUL2 that is >= 12 instr per 8 weights (= today), needs FP16 x (BF16 -> FP16 not lossless below
  2^-14): no. FP8 wgmma: needs FP8 x (rule) and k32 > the 16-wide scale block: no.
- Budget view (layer score weights per us: d1 .16, v1 .10, d16 .07, d32 .05, v16 .04, v32 .03 %): tk2f d1/v1 are at
  the HBM rate (4.06 TB/s per marginal expert vs a ~3.6-4.3 TB/s bare stream), big-M is dq-issue + last-wave bound
  (s17-s20). No H100-testable moe item with size left; the 00:00Z DIAG check (STATE n4s3) is the next real step.

## Night4 session 5 (2026-10-26 19:57Z - ~20:35Z, no judge runs: moe checks 20/20 until 00:00Z) -- GS16 loss traced, split fixed
- Question: why GS16 (tk2h TK2H 16: one dequant shared by 16 tokens, -13% instructions, fewer DRAM bytes) loses on the
  H200 at d16/d32 (s12: +4.0/+7.1 us). The G88 proxy (tools/s16) reproduces it: h16w0 vs h8 d16 +5.5, d32 +6.6 proxy-us.
- ncu (G88, d32 seed 7, tools/n4s5/ncu_h.py + top_sb.py): h16 34.6M instr vs h8 39.8M, DRAM 286 vs 301 MB, but 153.8 vs
  141.9 us; long_scoreboard 0.90 vs 0.35 per issue: 10% of all warp samples sit on the gemm2 counter spin (LDG.STRONG.GPU
  + CCTL.IVALL, 532k executions). => phase-1 IMBALANCE: count split puts the 2-group rows' extra work on a few CTAs.
- Per-CTA timeline (tools/n4s5/mk_t.py + tl.py, G88 d32 s7; us min/med/max): h8 ph1end 71/74/76, end 114; h16w0 ph1end
  62/62/81, wait med 19, end 114; h16w2 ph1end 64/66/70, end 104. But at d16 s99 (no expert > 8 tokens: same work) h16w2
  is +7 us in the graph: the weighted split itself, tk2g_split, ran 64-bit loops + divisions in EVERY thread: consumers
  started 1.8 (median) / 4.1 (max) us late.
- FIX (r-moe 1378d3e): tk2g_bound = warp 0, one lane per active expert, 32-bit shuffles (like tk2w_bound), #define
  TK2G_PAR 1. Same bounds; h_ab-style bitwise vs main 0/1600 (320 routings x W 0/2/3/4/9), bench test 15/15.
  G88 proxy (2 rounds, 5 seeds): d16 h8 90.5 / h16w2 89.75 / h16w3 89.6 / main 89.3; d32 h8 122.1 / h16w2 115.4 /
  h16w3 114.6 / main 120.6 (was h16w2 91.5 / 118.9 with the old split). => h16w3 ~ -4.4 H200 us at d32 IF the proxy holds.
- CAUTION: the proxy got the s12 W effect wrong: H200 h16w0 95.5 -> h16w2 94.2 at d32 (-1.3), proxy -7 H200-us. Full-grid
  H100 (DRAM-bound) is balanced at w0 (h16w0 beats h8 by 1-7 us at d32 on every seed; w2/w3 overshoot: heavy CTAs end
  first). The right 2-group weight depends on how DRAM- vs issue-bound the GPU is: H200 must decide.
- DIAG (00:00Z check) "dec" += h16w0/2/3/4 at M >= 9 (TK2H_MIN 9: d1/v1 stay main), DIAG builds compile TK2_EXP 1 (local
  dry run 95 s, prints all variants). NOTE the h variants lack main's PF 4 (slot-ordered y32 + norm3: -0.5..0.9 us),
  so a tie with gs8 = a ~0.7 us real win after a port.
- tools/n4s5/mktl_h.py -> /tmp/moe5tl/submission.py: H200 tk2h timeline DIAG (diagth lines: c0/ph1end/wait/end per
  variant per d16/d32 seed). Second check if the DIAG shows h16 within ~2 us of gs8.
- If h16 wins on the H200: layer port is big (layer's tk2f has TK2FX/TK2FJ weights-before-wait, late PDL trigger,
  x via __ldcg after griddepcontrol.wait): first port PF 4/norm3 into tk2h here, then hand off the GS16 consumer.

## Night4 session 6 (2026-10-26 20:18Z - ~21:00Z, no judge runs: moe checks 20/20 until 00:00Z) -- tk2h PF4 + layer port
- Lead 20:16Z asked for "the hot-expert single pass": already built in s29 (moe_hot) and in layer v87; did the queued
  GS16 follow-up instead (STATE n4s5: port PF 4 into tk2h so the H200 DIAG compares like with like, then hand off).
- tk2g_row<..., SLOT> / tk2h_consume<..., SLOT>: gemm2 writes Y[s_srt[off + tok]] = v * gsc (gsc = gs2 * 256), as
  tk2f_consume PF & 4; moe_tk2f passes (PF & 4). Dispatch: tk2h cfg % 1000 == 4 -> PF 4 instantiations (kinds 3/4).
  Python: TK2H_PF4 (cfg + 4), TK2H_M = {M: W} (GS16 PF4, overrides TK2H/TK2W), kernel_fn picks moe_norm3 for them.
- Bitwise: tools/n6/h4_ab.py (TK2_EXP 1 copy): TK2H 8/16, W 0/3, PF4 0/1 vs main, layer decode routings + 120 random
  T=1..32, each twice: 0 of 1680 differ. TK2H_P4 build (P4ONLY): 0/480. TK2H_M W 0/3/4 on layer routings: 0/60.
- G88 H200 proxy (proxy_s.py, 2 rounds, proxy-us): d16 main 89.3/89.5, h8p4 89.8/90.2, h16w0p4 95.6, h16w3p4 89.6,
  h16w3 89.4; d32 main 121.0/120.7, h8p4 121.6, h16w0p4 128.0, h16w3p4 114.7/115.0, h16w3 114.6. PF4 itself ~0 here (as s13).
- #define TK2H_P4 (default 0): builds only tk2f_go<..., SA, GS 8/16, MRG 1, PF 4> without TK2_EXP; compile time of the
  layer's moe lib 83.9 -> 81.8 s (noise: no cost). Without TK2_EXP a tk2h cfg now raises (TORCH_CHECK) instead of
  silently falling into the NC16 branch.
- tools/n6/patch_layer_h16.py <moe_src.py>: layer port (r-layer 69ab934's moe_src): tk2g_bound split (n4s5), SLOT,
  griddepcontrol.wait before tk2h_consume (the layer's TK2FX/TK2FJ wait sits before tk2f_consume only) + __ldcg for the
  gated x loads, TK2H_P4 1, python TK2H_MIN / TK2H_PF4 / TK2H_M. In a /tmp copy of r-layer (built): bench test 13/13
  with TK2H 16 W3 PF4 forced at T>=9; layer-namespace stress (T 8/9/16/24/32 x 60 skewed, W 3/2 + GS8, twice) 0/1800.
- H100 layer harness (tools/ab_flag.py, tuner on, 3 rounds, us sliding/full): d16 main 197.4/294.1, h16w3p4 201.3/298.1,
  h16w0p4 199.6/301.7; d32 main 194.1/411.9, h16w3p4 201.2/413.9 (2nd run 196.0..200.9), h16w0p4 201.8/423.4, h8p4
  195.9/415.9. Same gap with TK2FJ off (d32 +4.3/+1.9), so not the PDL overlap. Moe-only H100 (DIAG dry run, full grid)
  had h16w3p4 d32 104.6 vs gs8 105.4, d16 84.5 vs 82.0. => H100 says GS16 loses in the layer; the H100 is DRAM-bound
  for these shapes and the proxy says issue-bound H200 gains: only the H200 tuner can decide (TUNE candidate, never default).
- Layer s58 was already wiring it (uncommitted r-layer work seen in a /tmp copy): TUNE option tk2h at T=16/32; told them
  under Requests that their value goes in as W directly (no -1).

## Night4 session 7 (2026-10-26 20:50Z - ~21:50Z, no judge runs: moe checks 20/20 until 00:00Z) -- tk2f phase barrier: ILV dead, BAL queued
- Lead 20:16Z item (hot-expert single pass) is DONE since s29 (layer v87 runs it). Took the tk2f phase-1 -> phase-2 transition.
- G88 proxy per-row timeline of main (tools/n7/tl_rows.py, tk2h GS8 = main; stamps after each CTA's 1st/2nd row per phase):
  steady row (R = 16-row tile pair, 72 KB) = 2.0-2.3 us in BOTH phases (the s12 "phase 2 is 12% slower per byte" is
  NOT per-row cost); what costs is the transition: ph1end median -> counter ok 0.8-2.3 us (wait for the slowest CTA)
  + the first gemm2 row 2.5-2.9 us (inter x load latency with the ring already full). d32 s7: ph1end 73.7/76.5/78.6,
  cnt 78.8, row1 81.7, row2 84.0, end 118. d1 s7: ph1end 13.3/15.1/15.4, cnt 15.9, row1 18.4, end 22.5.
- DEAD: ILV (CTA b takes rows b, b+G, ... of every segment, per-segment counters cnt[64 + j], next-segment x prefetch):
  barrier gone (spin 0.3 us) but H100 d32 105.9 -> 130 (L1 prefetch) / 154 (cp.async smem prefetch), d16 +13..19.
  ncu d32: +5.3M instr, L2 hit 10 -> 39%, long_scoreboard x2: every CTA reloads each segment's x (8 tok x 4096 x 2 B =
  64 KB, ~0.9 R of weights) = +200 MB L2->SM. Any scheme that multiplies segment visits per CTA pays 64 KB per visit.
  Diff kept in tools/n7/ilv_dead.diff (timeline tools tools/n7/mk_i.py + tli.py).
- BAL (cfg 98204 = 84204 + PF 32; 99204 = 204 + PF 32 for M = 1): per-segment counters (cnt[64 + j], zeroed by
  moe_norm2/3 as before) + the split puts gemm13's n13 % G extra rows on the LAST CTAs and gemm2's n2 % G on the FIRST,
  so no CTA has an extra row in both phases (max rows ceil((n13 + n2) / G) instead of ceil + ceil). Bitwise = main
  (tools/pf_ab.py 100 98204,99204: 0/240), bench test 15/15.
  H100 full grid (layer_ab, 3 rounds): d16 83.0 -> 81.8, d32 105.7 -> 108.3, v1 84.0 -> 83.5, d1 (99204) 36.0 -> 36.4.
  G88 proxy (2 rounds): d16 88.6/89.2 -> 88.0/88.2, d32 119.7/120.7 -> 119.5/120.7, v1 87.0/86.8 -> 85.6/85.5,
  d1 36.8 -> 38.2. => if the H200 agrees: v1 ~-1.0, d16 ~-0.6 H200 us (~0.14% layer), d1 must stay 204.
  Queued in DIAG "dec": bal (M > 1) and bal1 (also M = 1). Local DIAG dry run: d16 bal 81.7 vs gs8 83.1, d32 107.0 vs 105.5.
- If bal wins on the H200 at v1/d16: the layer port is small (moe_src: the PF & 32 lines in tk2f_consume / moe_tk2f +
  dispatch; layer's other kernels after tk2f must zero cnt[64..127] too, as moe_norm2/3 do: check n2u).

## Night4 session 8 (2026-10-26 21:17Z - ~22:20Z, no judge runs: moe checks 20/20 until 00:00Z) -- BAL ported into the layer
- tools/n8/patch_layer_bal.py <r-layer muse_decode_layer dir>: n7 BAL (tk2f PF & 32: per-segment counters cnt[64 + j],
  gemm13 rounding rows on the last CTAs, gemm2's on the first) into the layer's moe_src.py + layer.py, in place (asserts
  every anchor). Launcher: cfg + 1000000 = BAL, SA only: pf 4 (84204) -> PF 36; pf 5 (NU8 85204) -> PF 52 only with
  #define MOE_BAL_NU8 (not built by default); every other cfg (204 at T=1, tk2h, ...) ignores the bit. Python:
  moe_src TK2F_BAL_M (set of M) -> _tk2f_cfg adds 1000000; layer TUNE option "bal" (bool), default TK2F_BAL_DEFAULT = False;
  T=8 nu8 path reads it too. cnt[64..127] is re-zeroed by every tk2f follower in the layer (moe_norm2/3, n2u, nu8 zero 0..127).
- Tests (/tmp copy of r-layer cef0979 + patch): tools/n8/bal_stress.py (namespaced _moe_kernel_fn, BAL vs not, T 2/5/8/9/
  16/24/32 x 40 skewed routings x 2 reps): 0/560 differ (bitwise). bench.py test with BAL forced on: 13/13 PASS.
  (The full layer output is NOT bitwise reproducible even with BAL off: 58/96 off-vs-off differ on scored inputs.)
- H100 layer harness (tools/ab_flag.py, tuner on, 4 rounds, ABA, 5 seeds, us): decode 16 sliding s1-s5 167.1/166.1/150.7/
  168.5/166.2 -> 162.6/163.1/150.6/166.9/162.6 (A again 166.5/165.9/150.1/168.3/166.3) = mean -2.4; full 277.5/271.1/297.5/
  253.3/276.9 -> 274.0/271.6/296.7/255.0/272.9 = mean -1.2. v1 (T=8) tie (+-0.3; the H100 tuner picks nu8d/nu8x False at
  T=8 -> PF 20 path, BAL not built there). d32 (tools/n8/ab_st.py: tk2h removed from the tuned state vs BAL vs tk2h 4):
  BAL LOSES, sliding +1.5..11.6 vs plain tk2f (noisy, shared GPU), full mixed. => BAL at T=16 only.
- Handed to layer (COORD Requests): apply the patch, add {"bal": True} to the T=16 TUNE candidates (the H200 tuner decides).

## Night4 session 9 (2026-10-26 21:27Z - ~22:15Z, no judge runs: moe checks 20/20 until 00:00Z) -- H200 HBM-ceiling probe in the DIAG
- Added moe_stream (r-moe 804b9f5): a bulk-copy ring kernel (1 producer lane + 1 warp that only waits full / arrives
  empty, 64 threads, NSTG 3/4/8) streaming a host-built copy list; `_diag_stream` (DIAG_STREAM = True, DIAG builds
  only) times it on the DIAG's 30 layer routings, cold L2, graph replay, 20 reps; prints `diag9s <load> s<seed> <v>`
  and `diag9s MEAN <load> <v>: t us, TB/s over UNIQUE bytes` (distinct experts x 28.3 MB). Variants:
  tkp = tk2f's exact copy pattern (rows of 4 stages x {W 8K, W 8K, S 2K} from 3 regions, one pass per 8-token group,
  gemm13 rows then gemm2 rows, count split, NSTG 4); uq / uq8 = each expert once (NSTG 4 / 8); uqk4 = KPS 4 shape
  (16K/16K/4K); sq16/sq18/sq18x8/sq32/sq36/sq64 = contiguous single-copy stages of that size. Default build unchanged
  (the kernel + launcher are compiled in but only the DIAG calls them); bench test PASS. DIAG dry run: seconds.
- H100 (full grid, MEAN of 5 seeds, us; tk2f gs8 = main from diag12 on the same routings):
           gs8    tkp    uq     uqk4   sq18   sq32   sq64
    d1     36.1   29.5   29.4   27.8   29.2   27.7   27.3
    d16    82.6   80.6   76.3   71.5   75.1   71.3   70.2
    d32   105.1  104.1   97.0   91.3   94.8   90.6   89.4
    v1     84.3   80.4   80.0   75.0   78.3   74.5   73.4
    v16  196.9*  271.8  172.2  166.4  169.8  165.7  163.4   (* twm off; tkp is meaningless at big M)
    v32  243.5*  421.0  190.7  184.8  186.8  183.4  181.0
  Reading (H100): tk2f runs AT its own copy pattern's rate (d16 82.6 vs 80.6, d32 105 vs 104): DRAM-bound as known.
  The stream rate depends on bytes per bulk copy: 8 KB copies (KPS 2) ~6% slower than 16 KB+ (uqk4 ~ sq32); one 18 KB
  copy ~ three copies 8/8/2 (sq18 vs uq, 1.5%). Multi-group re-reads (tkp vs uq) cost d16 +4, d32 +7 us without hints.
- DEAD (H100, bitwise = main): tk2f KPS 4 in the real kernel (cfg 84404 / 84403 / 404 at M = 1, PF 4): d1 +2.2 / 0,
  d16 +5.8 / +2.9, d32 +4.2 / +1.8, v1 +4.3 / +2.0 us (NSTG 4 / 3). With the older H200 result (diag 619ad6ad: 403
  +0.8, 402 +3 at d32) KPS 4 is closed: the stream's 16 KB preference does not carry into tk2f. Code reverted.
- 00:00Z: the DIAG check now also prints diag9s. What to read: on the H200, (a) tkp vs gs8 per decode load = how far
  tk2f is from its own copy pattern (consumer cost; the H100 shows ~2 us = fixed costs only); (b) sq64 TB/s = the
  achievable HBM read rate on these routings (the real floor for the layer's budget, instead of 4.0 TB/s);
  (c) uq vs sq32/uqk4 = what a repack to bigger copies could give if (a) is small. If tkp ~= gs8 and sq32 >> uq on
  the H200, a single-copy 36 KB-stage repack of w13/w2 with KPS 2 consumers (stage = 2 rows' chunks) is the next build.

## Night4 session 10 (2026-10-26 21:44Z - ~22:15Z, no judge runs: moe checks 20/20 until 00:00Z) -- KPS 4 re-closed; er + nm DIAG probes
- Paper first: (a) phase-interleaved tk2f (gemm2 of expert e while gemm13 of later experts streams, to drop the grid-wide
  gemm13 -> gemm2 wait): NOT built: the H200 timeline (s12) shows the 72 KB ring already absorbs the wait at d1 (ph1end
  median ~12.3, ring full ~13.8, cnt ok 14.1); and d1's 1152 rows / 132 CTAs = 8.73 -> 9 rows max whatever the order.
  (b) row interleave of multi-group segments for L2 reuse: moot, TK2F_L2H 4 already gives DRAM bytes = floor (ncu, s21).
- Stream probe (moe_stream, H100, 5 seeds MEAN, us; tools: /tmp/moe-n10/ds.py, monkeypatched _diag_stream variants;
  moe_stream_launch now also takes NSTG 2):
             tkp_n4  tkp_n8  tkp4_n2  tkp4_n3  tkp4_n4      (tkp = tk2f copy pattern 18 KB stages; tkp4 = 36 KB stages)
    d1        30.3    29.6    27.7     28.0     27.9
    d16       83.8    80.3    74.8     77.8     78.1
    d32      104.5   103.4    95.1     98.4     98.6
    v1        79.4    78.8    74.4     74.9     75.0
  => at the SAME 72 KB of smem, 36 KB copies x 2 looked 5-9 us faster. BUT the stream has no L2 hints (tk2f has mode 4).
- Real kernel, moe-only DIAG on the layer routings (H100, /tmp/moe-n10/dd.py <name>=<TK2F_DYN>:<TK2F>, us):
    cfg 402 (KPS 4 NSTG 2, existing SA build): d1 +0.5..1, d16 +1, d32 +1, v1 +0.7 vs gs8 (bitwise).
    ER (new, PF & 128): consumers release the ring slot right after the stage LDS, ordered by
      `fence.proxy.async.shared::cta` before mbar_arrive (instead of after the dq/mma, s10): er204 tie (-0.5..+0.9),
      er402 +0.5..1.3, er403 +1.5..2.7. Bitwise 0/1680 (tools/pf_ab.py 400 94204,91204,94402,94403). cfgs 94xxx = ER + PF4
      (norm3), 91xxx = ER only (M = 1 / norm2); xxx = 204/402/403; SA only; L2 hint mode at run time (+1000000 * (mode+1)).
    L2 hints off (mode 0) for er204/er402: +1.5..2.7 worse; mode 1 for er402: mixed.
    NO-MATH probe (PF & 256, TK2_EXP builds only, cfg 91214 / 91412: LDS + arrive, no dq/mma, wrong output):
      nm204 = nm402: d1 35.1, d16 79.5/79.3, d32 101.2/101.1, v1 81.5/81.4 vs gs8 36.1 / 82.6 / 104.9 / 84.3.
  => On the H100 the whole consumer math costs only 1.0 / 3.1 / 3.7 / 2.8 us (d1/d16/d32/v1), and with no math the copy
     size does not matter (nm402 = nm204): the stream probe's KPS 4 gain was its missing L2 hints. KPS 4 CLOSED (3rd time).
- 00:00Z DIAG now also has `er` and `nm` (dec list) + diag9s `tkp4`. READ: gs8 - nm on the H200 = the upper bound of ANY
  consumer-side work (GS16/tk2h, cheaper dq, ER). If it is ~1-4 us like the H100, stop consumer work on tk2f; the
  remaining tk2f lever is then the stream itself (fixed costs, tail). If it is >= 8 us at d16/d32, tk2h GS16 / dq work is
  worth it on the H200 (compare h16w*p4 there).
- r-moe 77fa9b6. Default build unchanged (bench test PASS); the DIAG build dry-runs.

## Night4 session 11 (2026-10-26 22:06Z - ~22:30Z, no judge runs: moe checks 20/20 until 00:00Z) -- 00:00Z DIAG readiness
- Full dry run of the queued DIAG check exactly as the judge builds it (/tmp copy with DIAG = 1, FRESH
  TORCH_EXTENSIONS_DIR, bench.py test): compile + prepare + all DIAG variants = 104.6 s, whole run 107 s (judge limit
  10-15 min: safe); 420 diag12 lines + diag9s; all 15 tests PASS (cloned state too). H100 numbers = n10's (gs8 - nm:
  d1 1.1, d16 3.5, d32 4.2, v1 2.8 us; bal d16 -1.5, d32 +1.4; h16w0 d32 -2.0; h32 v32 228.5 vs off 243.4).
- tools/n11/diag_read.py <profiles/submission.<job>.log>: per load every variant's MEAN, delta vs the base (gs8 / off),
  delta x layer score weight (pct), per-seed values, the diag9s table, and the n9/n10 reading rules (gs8 - nm, gs8 - tkp).
- Paper, not built: per-slice fused gemm13 -> gemm2 for d1 (each CTA does an intermediate slice's gate/up then the W2
  K-slice partial, FP32 partials reduced in norm2: no phase barrier). n10 already showed the H200 ring absorbs the d1
  barrier, and the finer granularity only removes 9 -> 8.73 row rounding (~3%, ~0.6 us). Not worth a 2 h build blind.

## Night4 session 12 (2026-10-26 22:25Z - , no judge runs: moe checks 20/20 until 00:00Z) -- TWM_PB_M 48 (N48 big-class passes)
- Layer routing histogram (tools/data/routes.pt): v16 has 17-18 local experts, 10-13 of them <= 8 tokens, 2-6 with 9-16,
  0-2 with 17-32, 1-2 with 33-49, none > 64; v32 21/18/20 experts, 9-12 <= 8 tokens, 2 hot (73-97). twm already runs
  <= 8-token passes at N8 (RT 4, HBM-bound), so the only consumer-bound passes at v16 are the 33-49-token experts (RT 2 N64).
- Built s18's untested alternative: moe_twm<NSTG, SA, PB> (PB = big-class pass size; 48 = 48-token passes at wgmma N48 in
  place of N64, NOT an extra instantiation: SASS of <.., 48> = <.., 64>: LDL 9 / STL 7, gsb0 0x0 10 / 0x1 40, 104 HGMMA).
  Host: moe_set_hot bit 1; python TWM_PB_M = {M: 48}. Bitwise = PB 64 on the 10 layer v16/v32 routings + 30 skewed
  synthetic (tools/n12/pb_t.py, HOT_M {} and {256: 32}): 0 bad, 0 non-bitwise.
- H100 full grid (proxy.py, 3 rounds, us): v16 o 195.0 | pb 198.4; v32 o 244 | pb 249.6; with hot 32 at v32: 228.6 vs 228.3.
  Per seed v16 (o/pb): s7 [39, 49 tok] 194.9/206.3 (49 = 48 + 1: a whole 2nd pass for one token), s11 189.6/189.4,
  s13 205.0/206.2, s17 196.1/196.7, s99 196.7/198.4.
- G88 H200 proxy (sub_g MOE_GRID=88, 3 rounds): v16 o 231.6 | pb 230.4 (-1.2 proxy-us = ~-0.9 H200 us), v32 +5.
  Per seed v16: s7 -1.5, s11 -4.0, s13 -2.6, s17 +2.0, s99 +0.2. => at best ~0.05% layer score at v16 (weight .055/us).
- Compile cost of the 2 extra instantiations: +7 s (fresh _lib(): 89 -> 96 s). So: #define TWM_PB48 0 (not compiled;
  TWM_PB_M then TORCH_CHECKs), _lib() sets it to 1 in DIAG builds only. 00:00Z DIAG: + pb48 (v16 {128: 48}) and h32pb48
  (v32 {256: 48} with hot 32). Rule: pb48 < off by >= 1.5 us at v16 on the H200 -> offer the layer TUNE pb48 at T=128
  (TWM_PB48 1 there); else close it.

## Night4 session 13 (2026-10-26 22:48Z - , moe checks 0/20 until 00:00Z) -- mixed N48/N64 big passes DEAD; then the 00:00Z DIAG
- Tried (not kept): moe_twm<.., PB 0> = 64-token big-class passes, wgmma N48 when the pass has <= 48 tokens, else N64
  (no 48 + 1 split like pb48). Bitwise = off, 0 bad (tools/n13/pb_t.py, layer v16/v32 + 30 skewed synthetic, SUB=sub_g13 in
  /tmp/moe16 = TWM_PB48 1 + MOE_GRID build); REG 168 / STACK 40 like the others.
- Proxy (tools/s16/proxy.py, 3 rounds x 5 layer seeds, HOT_M {256: 32}, H100 us):
    G88  v16: o 230.8 | mx 230.8 | pb48 229.4      v32: o 241.0 | mx 243.3 | pb48 241.7
    G132 v16: o 196.9 | mx 197.3 | pb48 199.7      v32: o 228.3 | mx 228.5 | pb48 228.8
  => N48 vs N64 for a <= 48-token pass is worth nothing (twm's big passes are not consumer-bound by N there); pb48's
  small G88 v16 gain must come from its extra items (49..64-token experts -> 2 items = finer load balance), and it
  flips sign at G132. Reverted; pb48 stays only as the queued DIAG probe.
- 00:00Z DIAG check 4a591c7d (de63ab1 + DIAG = 1; 2.197x as a test run, NOT for submit): every diag12 line LOST: the judge
  keeps only result.log_tail (~207 lines) and the n9 stream probe printed ~330 diag9s lines after them. Fixed in 36bcb76
  (one compact `diag12 MEAN <load> <v>: X us | seeds .. | maxdiff ..` line per variant = 82 lines; DIAG_STREAM False).
- H200 stream probe (diag9s MEAN, us / TB/s over unique bytes; moe_stream = bulk-copy ring, consumers only wait/arrive, no hints):
            tkp    uq     uq8    uqk4   tkp4   sq16   sq18   sq18x8 sq32   sq36   sq64
    d1      24.8   24.8   24.1   22.0   22.0   24.0   23.7   23.6   21.9   21.9   21.5
    d16     65.6   61.0   59.2   53.5   57.4   59.1   57.5   57.2   53.1   52.8   52.1
    d32     86.8   77.5   75.3   68.0   74.5   74.6   73.0   72.4   67.4   67.1   66.1
    v1      63.9   64.0   62.0   56.4   56.4   61.8   60.5   60.0   55.8   55.6   54.7
    v16    213.1  134.4  130.7  122.8  190.4  129.3  127.8  127.0  121.3  121.1  119.2
    v32    343.6  147.4  142.7  135.7  302.8  142.0  140.8  139.9  134.6  133.8  132.1
  (tkp = tk2f's copy pattern 18 KB stages x NSTG 4; uq/uq8 = each expert once NSTG 4/8; uqk4/tkp4 = 36 KB stages NSTG 4;
  sqNN = contiguous single-copy NN-KB stages.) Readings: on the H200 copy SIZE matters: at the same 144 KB in flight
  36 KB stages (16 KB copies) beat 18 KB (8 KB copies) by 5.7 / 7.3 / 5.6 us (d16/d32/v1: uq8 vs uqk4); single
  contiguous 32-64 KB copies reach 3.9-4.0 TB/s at d16/d32, 4.2 TB/s at v16/v32. tk2f's own pattern (tkp) is 3.06 TB/s
  at d16/d32 incl. its multi-group re-reads (tkp - uq = 4.6 / 9.3 us there; no L2 hints in the probe). But the real
  kernel at KPS 4 did NOT gain on the H200 in s12 (619ad6ad, no hints then: 403 89.7 / 402 91.9 vs 204 88.9 at d32), so
  the probe's copy-size gain is not available to tk2f as is; the rerun's nm / er402 / er403 lines decide what is.
- Rerun ab707ffc (36bcb76 + DIAG = 1, 2.219x test run): all 82 compact lines arrived. H200 MEANs (us, delta vs base):
    d1  gs8 30.5 | l2h0 +1.2 l2h1 +0.8 | h16w0 0 h8p4 -0.2 h16w2/3/4p4 -0.3 | bal -0.3 bal1 +0.5 | er +0.8 | nm -1.8 | er402 +0.9 er403 +1.3
    d16 gs8 66.0 | l2h0 +1.9 l2h1 +1.6 | h16w0 +4.6 h8p4 +0.5 h16w2p4 +3.2 w3 +4.0 w4 +4.2 | bal 0 bal1 +0.2 | er +1.9 | nm -4.2 | er402 +2.2 er403 +2.3
    d32 gs8 87.6 | l2h0 +1.8 l2h1 +0.6 | h16w0 +5.6 h8p4 +0.8 h16w2p4 -1.0 w3 -1.9 w4 -0.2 | bal +1.7 bal1 +1.7 | er +2.7 | nm -6.5 | er402 +3.2 er403 +3.1
    v1  gs8 65.0 | l2h0 +2.1 l2h1 +0.9 | tk2h -0.5..+0 | bal 0 | er +0.7 | nm -3.6 | er402 +1.4 er403 +0.7
    v16 off 172.4 | hot any -0.3..-0.6 (no hot experts) | o76 -0.7 | h32rg0 -2.5 (vs h32 -0.6: RG 0 better by 1.9) | ne0 +1.1/+2.0 | pb48 -2.4
    v32 off 218.0 | h24 +5.4 h32 -5.4 h40 -2.8 | h32o76 -5.5 | o76 +1.7 | h32rg0 -6.3 (RG 0 better by 0.9) | h32ne0 -2.1 (NE worth 3.3) offne0 +5.6 | h32pb48 -5.0
  Decisions: tk2f consumer work closed (gs8 - nm <= 6.5 us anywhere; every consumer variant <= -1.9); KPS 4 / ER / BAL
  dead on the H200; L2H 4, NE 1, hot 32 @256, no o76: confirmed; RG 1 loses (layer already runs RG off); pb48 @128 -2.4
  offered to layer as TUNE. Posted under COORD Requests (moe -> layer n13).
- The remaining tk2f lever = its copy stream: gs8 87.6 = tkp 86.8 at d32 (tk2f runs exactly at its copy pattern's rate).
  Probe pieces at d32 / d16 / v1 / d1: multi-group re-reads (tkp - uq) 9.3 / 4.6 / 0 / 0; 3 copies per 18 KB stage vs ONE
  contiguous 18 KB copy (uq - sq18) 4.5 / 3.5 / 3.5 / 1.1 us -> NEXT BUILD (below).

## Night4 session 14 (2026-10-27 00:32Z - 01:15Z; moe checks 3/20 used today) -- TK2F_SQ single-copy tk2f stages
- Built n13's NEXT: prepare `_sq_pack` repacks w13q/w2q + w13s2/w2s2 into the tk2f stage images
  [E][RPE][NQ][SB] = {tile lr kc q*16.., tile lr + RPE same kc, scales} (int32 [E, 256, 4, 4608] for w13, [E, 128, 4, 4608]
  for w2; lossless, +~0.86 GB); tk2f_produce<.., SQ> issues ONE cp.async.bulk of SB = 18 KB per stage (same L2 hint per row);
  consumer unchanged. Kernel PF bit 512 (PF 512 = M 1 cfg 204, PF 516 = M > 1 cfg 84204), host cfg + 10000000, python
  TK2F_SQ = 1 (now default; kernel_fn passes st["w13sq"]/["w2sq"]). KPS 4 SQ (36 KB stages, _sq_pack nq 2, cfg 402/403 SQ,
  #define TK2F_SQ4) compiled in DIAG builds only.
- Bitwise vs off: 228/228 (tools/n14/sq_t.py: 20 layer routings + 56 random M 1..32, 3 reps each). bench test PASS.
  H100 layer routings: d1 36.1/36.0, d16 82.7/82.2, d32 104.8/104.7, v1 83.9/83.8 (DRAM-bound locally, as expected).
- H200 DIAG check df253e4c (01326b4 + DIAG = 1; 2.214x as a test run, not for submit). MEAN us (repeat in brackets):
            gs8 (off)     sq            sq402   sq403
    d1      30.3 [30.2]   30.2 [30.2]   31.3    31.3
    d16     66.1 [66.1]   65.5 [65.6]   66.0    66.5
    d32     87.6 [87.6]   86.1 [86.0]   85.8    87.3
    v1      64.7 [64.8]   64.8 [64.7]   65.6    65.7
  => SQ: d32 -1.5, d16 -0.55, d1/v1 0 (repeat agrees to 0.1 us). Only ~1/3 of the stream probe's uq -> sq18 gain
  (probe: d32 -4.5, d16 -3.5, v1 -3.5): the probe's per-copy cost is not tk2f's limiter at d1/v1. KPS 4 SQ: d32 -0.3
  more but d1/v1 +1: dead (4th time KPS 4 fails in the real kernel). Layer weight: ~-1.5 x .056 + -0.55 x .072 = ~0.12%.
- Handoff to layer posted (COORD): port = _sq_pack in prepare + SQ template bit in tk2f_produce + PF 512 at the producer
  call site + the w13sq/w2sq pointers at its 3 moe_tk2f_launch call sites (T = 1, 8, 16/32). tools/n14/sq_layer.md.

## Night4 session 15 (2026-10-27 01:06Z - ; moe checks 6/20 used today) -- first H200 big-M timeline; AUTO-split clamp bug
- DIAG timeline tool made generic (submission.py `_diag_tl` + `DIAG_TLV`; TWM_TL builds store a per-item descriptor
  g2 | class << 1 | Khalf << 3 | pass << 4 | nv << 8 | e << 16 after decode, stamps now at part + 66*32768 floats, past the
  K-split partials). Build: `sed 's/^DIAG = 0$/DIAG = 1/; s/^#define TWM_TL 0$/#define TWM_TL 1/; s/^DIAG_TL = () /DIAG_TL = (1,) /'`.
  Summaries: /tmp/n15/an.py <log> (copy in tools/n15/an.py). ~110 log lines: fits the judge's log_tail.
- H200 check 5bbb03ca (a2b0634 + DIAG, 2.215x test run, never submit). moe-only, layer routings, cold L2, MEAN 5 seeds:
    v16 twm (off) 175.7 | twq4 177.9 (twq LOSES on the H200 moe-only, +2.2) | v32 off 223.7 | hot32 216.7 (-7.0)
  Timelines (means over 5 seeds, us from first CTA start):
    v16 twm:  exit min/med/max 120.5/140.2/157.9, busy/G 135.7, loss (= last exit - start - busy/G) 21.9, last gemm13 end 117.9
    v16 twq4: 125.0/144.3/162.2, busy/G 139.3, loss 22.5
    v32 twm:  165.9/186.7/203.2, busy/G 181.2, loss 21.7
    v32 hot32 (twm CTAs only; 32 SMs on moe_hot until ~155-160, then the last twm CTAs start): 158.8/172.2/198.8
  HBM rate (twm first-pass bytes only), TB/s per 20 us:
    v16 twm   2.5 2.9 3.7 3.8 3.6 3.6 2.6 1.3       (first wave = heaviest experts: big RT2 N40-56 + N16/N32 items, consumer-bound)
    v32 twm   0.9 1.4 3.2 3.4 3.8 3.8 2.4 2.9 3.4 1.5 0.2   (first ~35 us: ALL CTAs on the hot experts' 64-token RT2 passes)
    v32 hot32 1.7 2.1 2.5 2.8 2.9 2.9 2.7 3.1 2.3 0.9  (+ ~0.37 TB/s of moe_hot's own bytes: twm on 100 SMs is consumer-limited)
  Item times (us, H200, steady): gemm13 RT4 N8 37.6, N16 44, N24 47, N32 50; gemm2 N8 35.7, N16 39.5, N32 48; big RT2 N40-64
  34-37 (0.56 MB: ~16 GB/s/SM, consumer-bound); 2nd (L2) passes 27-31. Peak sustained 3.8-4.0 TB/s mid-kernel.
  Readings: (1) v16 is co-bound (busy/G 136 vs bytes ~127 us at 4.0); its losses are the drain (~22 us: items 36-50 us, 3.3-3.6
  waves) and the first wave at ~2.7 TB/s (~10 us). (2) v32 is consumer-throughput-bound in aggregate (N16-N32 + hot passes);
  hot32 helps 7 us but leaves twm on 100 SMs at ~2.7 TB/s. (3) the drain is ~20 us on EVERY variant.
- HOT_TWG (new python option, off): with moe_hot, moe_twm grid = SMs - k (no late-starting twm CTAs on freed hot SMs).
  H100 layer v32 (tools/n15/twg_ab.py, 2 rounds, bitwise): off 241.7 | h32 227.4 | h32g32 231.0 | h32g16 227.7 | h24g24 247.6 |
  h40g40 232.4 => dead (late CTAs help).
- BUG FOUND (twm launch): `ns = min(ns, part.numel() / 32768)` clamped TWF_SPLIT 1022/1023 (AUTO) to 66 whenever it was NOT
  combined with order/hot bits (the negative encoding bypasses the clamp). So every plain "a22/a23" / twf_split 1022 number
  since s20 (moe proxy, layer tuner) was a fixed 66-item K split. Fixed in twm only (twf has no AUTO). The layer's moe_src.py
  has the same clamp: its {'twf_split': 1022} = {'twf_split': 66}; {'twf_split': 1022, 'twm_order': 76} = real AUTO.
- H100 with the fix (layer routings, hot32 at 256, NE on, us): v16 h32 197.6 | s66 193.8 | a22 195.8 | a23 196.9 | t32 195.1;
  v32 231.4 | 221.5 | 225.2 | 228.2 | 226.5. => the fixed 66 split beats the real AUTO; s66 + hot32 at v32 -10 H100 us.
  Timelines (H100): v16 loss 26.6 -> 16.9 (s66), v32 67.4 -> 56.8. (The layer's H200 tuner already picks 66 at T=256:
  v103 log sliding 344.3 -> 340.6, full 535.8 -> 529.5.)
- Split-count sweep (H100, tools/n15/split_ab.py, hot32 at 256, 2 rounds): v32 h32 227.7 | s33 222.6 | s66 219.3 | s100 220.7 |
  s132 223.8 | s200 224.3; v16 195.2 | 193.8 | 193.0 | 194.1 | 193.7 | 194.0 => 66 is the optimum (v32 -8.4, v16 -2.2).
- H200 check 609ab981 (6147b67 + DIAG, 2.190x test run, never submit): the diag12 MEAN lines were LOST again: log_tail is
  BYTE-limited (only the last 95 lines survived: the long "TB/s per 5us" / "kinds" lines). Next DIAG: <= ~60 short lines, or
  print MEANs LAST. The diagtl summaries survived (5 seeds, twm CTAs, makespan = last exit, us):
    v16 h32 160.1 | h32s66 156.6 (-3.5) | h32a22 158.0 (-2.1); loss 23.1 / 16.9 / 20.7
    v32 h32 198.2 | h32s66 190.4 (-7.8) | h32a22 191.0 (-7.2); loss 61.7 / 51.7 / 53.9 (twm-only view, hot SMs counted idle)
  => the fixed 66-item K split is the drain fix on the H200 too, at BOTH T=128 and T=256 (moe-only). Told layer.
- Big-M levers left after this (all need new work): first-wave HBM under-use (v16 ~10 us at 2.7 TB/s; v32 without hot ~25 us
  at ~1 TB/s), v32 aggregate consumer bound (N16-N32 items 44-50 us vs N8 37.6 for the same bytes; dq + MMA do not overlap,
  s18), and the remaining ~17 us drain at v16 (halves only shorten the last items to ~half).

## Night4 session 16 (2026-10-27 01:50Z - ~02:50Z; moe checks 6/20 used today) -- tk2h on SQ stages: d32 -1.7 H200 us
- The layer's H200 profile (b9685e03) shows it runs tk2h (moe_tk2f<.., 16, 1, 4> = GS16 + PF4) at d32, which had no SQ
  (n14's single-copy stages were gs8-only). The SQ stage image is byte-identical to the 3-copy stage and tk2h shares
  tk2f_produce, so tk2h + SQ = PF 516 on w13sq/w2sq: new dispatch in the `if (sq)` block (TK2H_P4 || TK2_EXP builds),
  python TK2H_SQ = 1 (cfg 14x004 -> + 10000000). Bitwise vs 3-copy 348/348 (tools/n16/hsq_t.py, SUB = TK2H_P4 1 copy).
- DIAG builds now compile TK2H_P4 (not TK2F_SQ4; KPS 4 SQ images behind TK2F_SQ4_DIAG = False: dead, saves ~0.9 GB).
- H200 check 8b79cc25 (03b61af + DIAG, 2.221x test run, never submit), MEAN us over 5 layer seeds:
            sq (gs8)  h16w2  h16w2sq  h16w3  h16w3sq  sqb
    d1      30.0      30.2   30.2     30.1   30.1     30.1   (tk2h not used at M = 1: same kernel)
    d16     65.3      69.6   68.8     71.0   69.4     65.5
    d32     86.3      86.9   86.7     86.2   84.5     86.3
    v1      64.7      64.6   64.8     64.5   64.8     64.4   (same kernel)
    v16 s66 171.3, v32 s66 207.0
  => at d32 tk2h W3 + SQ -1.7 us vs tk2h W3 (the layer's pick) and vs gs8 SQ; d16 keeps gs8 SQ. Handoff + Request posted
  (port note tools/n16/hsq_layer.md). Layer weight ~1.7 x .056 = ~0.1%.

## Night4 session 17 (2026-10-27 02:22Z - ~03:20Z; moe checks 7/20 used today) -- TK2G_PAR4: tk2h weighted split ~2 us faster to start
- Re-read the layer's H200 profile (acce7f04, v105): decode tk2f is at its HBM floor at d1/v1 (d1 2-3 experts 20.6 us,
  v1 ~7.6 experts 49 us); d16 65 us vs ~49 floor, d32 75-90 vs ~63 (extra 8-token groups); twm v16 170-177, v32 ~200
  (dq-bound, s18/s22). Layer v106 now runs tk2h W3 (GS16 + PF4 + SQ) at T=32.
- New tool tools/n17/mktl_d.py: DIAG copy with kernel-level globaltimer stamps in moe_tk2f (entry, consumer start c0,
  phase-1 end p1, gemm2 spin end, CTA end, rows per CTA), variants sq (main gs8) / h0 (tk2h W 0) / h3 (tk2h W 3) on the
  layer's d16/d32 routings; 30 short "diagtd" lines (fit the judge's log_tail). Local dry run 95 s + tests 15/15.
- H100 timeline FOUND: h3 consumers start at 4.4 us vs 2.3 (sq, h0): tk2g_bound's 4 bounds ran serially on warp 0
  (smem loads, shuffles, 64-bit division: a latency chain) and every warp, the producer too, waited at the next
  __syncthreads. FIX TK2G_PAR4 (#define, default 1): bound w on warp w < 4, p in 32-bit (W * b < 2^28). Same bounds:
  bitwise 138/138 (tools/n17/par4_t.py: W 2/3/4 on layer d16/d32 + 36 random T 9..32, vs a TK2G_PAR4 0 build).
  H100 h3: c0 4.4 -> 2.6 us; moe-only totals d16 -0.4..2.4, d32 -0.8..1.5 us.
- H200 check 0d0e6c8d (bdfc090 + mktl_d DIAG, 2.213x test run, never submit), tot = graph replay us (5 seeds, mean):
    d16: sq 66.1 | h3 67.6 | h0 71.4      d32: sq 87.2 | h3 83.8 | h0 91.0
  (n16 without PAR4: d32 h16w3sq 84.5 vs sq 86.3 = -1.8; now -3.4: PAR4 ~ -1..1.6 us at d32 on the H200.)
  Timelines (H200 us, min/med/max over CTAs): h3 is BALANCED on the H200 (d32 s7 p1 45.8/48.9/51.7, spin med 3.3;
  sq 48.6/49.7/52.2) unlike the H100 (p1 36-52: W3 overshoots when DRAM-bound). h0 (count split) is badly imbalanced
  (d32 s11 p1 51/53/70, spin med 18): W must stay > 0.
  d16: h3 ties sq on seeds with a 9-16-token expert (s7 66.1/66.0, s13 66.7/66.7) and loses 2-3 us without one
  (s11 +2.0, s17 +2.4, s99 +3.0). s99 (no expert > 8 tokens: identical work) shows WHERE: phase 1 p1 median 47.1 vs
  45.1 (+4% per gated row), phase 2 equal (e - p1 25.6 vs 25.8), c0 +0.3. => tk2h's merged-body gated rows are ~4%
  slower than tk2f_consume's on the H200. If that is fixed, h3 wins d16 seeds with a hot expert by ~2 us and
  ties the rest, and d32 gains ~1-2 us more. Suspects (not tested): run-time gated/RPE in tk2g_row (GATED_ vs RTG),
  tk2f's PF bits the tk2h path lacks (the layer's d16 tk2f is PF 548 = SQ + BAL + slot y32), i-cache of one body.
- Layer port: tools/n17/par4_layer.md + par4.diff (3 hunks in moe_src.py, device code only). Posted under Handoffs.

## Night4 session 18 (2026-10-27 02:50Z - ~03:50Z; moe checks 8/20 used today) -- layer is on L2H mode 1 (moe: 4): port note
- PF2 (L2 bulk prefetch of each CTA's first gemm2 stages when its producer reaches gemm2, to use the gemm13->gemm2 spin):
  bitwise, but H100 full grid +1..8 us and G88 proxy +0.5..2 at every decode load, monotonic in the prefetch depth. DEAD
  (reverted; the critical path is the slowest CTA's gemm13, and the extra traffic slows it).
- tk2h 1-group-row cost (n17's 4%): ncu source counters (d16 s99, no expert > 8 tokens): instructions +0.9%, time +5%.
  Stalls are in the gemm2 counter spin (CCTL.IVALL of ld.acquire, +210 samples) and an LDC of the run-time X/inter param at
  every segment start (157 samples). TK2H_REG (asm "+l" on Xp/Out/gs, #define 1, bitwise) removes the LDC; H200 h3 is unchanged:
  d16 66.2 vs sq 64.9 (+1.3; n17 +1.5), d32 83.0 vs 86.3 (-3.3; n17 -3.4). Warm-L2 consumer proxy (tools/n18/warm.py):
  a 2-group tk2h row costs 1.10-1.15x a 1-group row on the consumer side, and 1-group rows cost the same as tk2f's (+1 us fixed
  prologue). So the H200's W=3 (1.375x) optimum is set on the memory side, not by consumer cost.
- moe_norm3 alone on the H100 graph costs ~3.3-3.9 us per launch even at T=1 (tools/n18/tail2.py: a latency chain; the
  layer's PDL hides the launch part).
- Per-phase L2 hint modes (TK2F_L2H2_RT + moe_set_l2h2; SQ run-time instantiations PF 576/580 behind #define TK2F_SQRT, compiled
  in DIAG builds only). Hypothesis: mode-4 evict_last w13 lines pile up in L2 and slow gemm2. H100 full grid, layer routings:
  d32 m34 (gemm13 3 / gemm2 4) -3.1 us, m33 -2.4. H200 check 1b1d0fbd (a8841ee + DIAG, test run, never submit), MEAN us:
            sq(4/4) m33   m43   m34   m11   m40   m44(RT ctl) h3
    d1      29.8    30.4  30.4  30.3  30.3  30.4  29.9        29.9
    d16     64.9    65.5  66.1  65.1  66.5  66.5  64.7        66.2
    d32     86.3    87.4  87.6  87.2  87.4  88.0  86.3        83.0
    v1      64.2    64.5  64.8  64.5  64.7  65.6  64.3        64.2
  => mode 4 in both phases stays best on the H200 (the H100 result was another DRAM-bound artefact). Per-phase modes: DEAD.
- FOUND: the layer's moe_src.py is still on `#define TK2F_L2H 1` and has no mode-4 code (n13 assumed it ran 4). m11 vs sq above =
  what the layer loses: d1 +0.5, d16 +1.6, d32 +1.1, v1 +0.5 H200 us. H100 IN-LAYER A/B (copy of r-layer 77c0f17, run-time mode
  switch g_l2m, ab_flag, 3 seeds, A B C A): mode 4 = d16 sl -2.1/-4.5/-2.6, d16 full -2.4/-2.0/0, d32 s11 -1.9 sl / -3.2 full,
  others tie; 3/4 is worse than 4/4 at d16. Port note tools/n18/l2h4_layer.md (3 lines, bitwise). Est. ~+0.25% layer score.

## Night4 session 19 (2026-10-27 03:27Z - 04:45Z; moe checks 9/20 used today) -- moe_twq weights-before-wait (WBW): DEAD on H100 + H200
- Context: layer profile 71f451bb (v106): v16 = moe_twq 170 + moe_norm 42 (twq's tail), v32 = twm 215 || moe_hot 160; d1/v1
  tk2f near its copy floor (s12 timeline: d1 kernel span 18.7 us for 2-3 experts). Big-M consumer work is closed (s17-s23:
  dq/wgmma never overlap, no cheaper dq), so tried the one untested fixed cost: twq's prologue behind its PDL primary.
- WBW = moe_twq flags & 16 (python TWQ / TWQ_M value + 16; kernel_fn strips it when RG = 1): no griddepcontrol.wait at entry;
  meta / s_off read with __ldcg; the producer claims its first item and issues weight + scale bulk copies for the first NSTG
  stages, then griddepcontrol.wait, then those stages' x TMAs (flush() before any blocking wait: ring full, gemm2 readiness
  spin, end). Valid ONLY when the primary is moe_gather / the layer's moe_gathern (they trigger after their own wait, so
  moe_route's meta is complete). With RG = 1 (moe_routegather writes meta itself after triggering) it HANGS in graph
  replay (pre-wait meta reads see stale counts -> gemm2 spin forever); eager runs passed by timing luck. Lesson: before any
  pre-wait read, check WHICH kernel is the PDL primary and whether it writes what you read after its trigger.
- Bitwise vs plain twq (tools/n19/wbw_t.py, RG 0): 0/26 bad (random bs 33/128/256/257 + skewed + layer v16/v32, 3 replays each).
- H100 moe-only layer routings (tools/s22/proxy.py, 3 rounds): v16 q4 200.7 | q20 (WBW) 202.7 | q52 (WBW, at most 2 stages
  before the wait) 202.6 vs 202.1; v32 q4 259.6 | q20 261.0. Monotonic in the prefetch depth, like n18 PF2.
- H200 check 9da19e71 (DIAG copy, 2.224x test run, never submit), MEAN 5 seeds, maxdiff 0:
    v16 q4 186.8 / q4b 186.3 | q20 189.0 / q20b 189.1 | q52 187.5;   v32 q4 246.5 / 246.8 | q20 248.7 / 248.1 | q52 246.8
  => +2.5 us (v16), +2 (v32): early weight traffic slows the gather -> x critical path on the H200 too. DEAD (code kept, off).
  Third confirmation (RFLAG s27, PF2 n18, WBW n19): HBM is never idle enough before the experts for prefetch to pay.

## Night4 session 20 (2026-10-27 04:33Z - ~05:35Z; moe checks 11/20 used today) -- S6: 6-bit block-scale codes, lossless: DEAD on the H200
- Idea: e2m1 codes are uniform random (4.0 bits, incompressible) but block scales are log2 ~ N(7, 0.4) clamped to [8, 448]:
  measured over 32 local experts, every e4m3 scale byte is in 93..126 (clamp floor 8.0 = byte 80). Scales are 128 of the
  1152 B per (warp, k chunk) of a tk2f stage (11%). 5 bits does not fit globally (34 distinct codes); per-tile-pair windows
  would (max range 31) but need escapes. 6 bits with base 80 fits ALL possible scales of this generator, and base 80 makes
  (c + 80) * 16 + 0x7700 = c * 16 | 0x7C00 (bits 4-9 of 0x7C00 are 0), so decode = PRMT + LOP3 (was PRMT + IMAD).
- S6 (PF & 1024, cfg + 20000000, prepare _s6_pack, st w13s6/w2s6; fallback to 8-bit SQ if any scale byte outside
  [80, 0x7E]): per lane per SQ stage (KPS 2) one nibble word (8 codes, bits 4k) + one u16 of high bits (bits 2k); stage
  17920 vs 18432 B (-2.8%). Producer/consumer (tk2f_consume, tk2g_row/tk2h) stage stride SB6. s6dec() in the source.
  Bitwise = SQ (tools/n20/s6_t.py 104/104 incl. tk2h W3; fb_t.py fallback ok); bench test 15/15; ptxas 162 -> 165 regs, no spill.
- H100 moe-only, layer routings (tools/n20/s6_ab.py, DRAM-bound): tk2f d16 -1.7, d32 -2.4, v1 -2.0, d1 -0.1; tk2h W3 d32 -1.7 us.
- H200 check 13b528a8 (4a87466 + DIAG, 2.208x test run, never submit), MEAN 5 seeds, us (b = repeat):
    d1 sq 30.2/30.1 s6 30.3/30.4 | d16 sq 65.7/65.4 s6 66.4/66.4 | d32 sq 86.1/86.0 s6 89.0/89.0, tk2h hsq 82.4 h6 84.0 |
    v1 sq 64.7/64.7 s6 63.9/63.9
- S6P (DIAG only, PF & 2048, cfg + 40000000): S6 images at the 18432-B stride (aligned starts, 17920-B copies) to test
  "misaligned bulk copies": H200 check c6b296a3 (6247aa1 + DIAG, 2.211x): s6p = s6 everywhere (d32 88.7 vs sq 86.0,
  d16 66.7 vs 65.9, v1 64.0 vs 64.6). Not alignment.
- Reading: S6 WINS where rows are single-group/byte-bound (d16 seed 99 with no expert > 8 tokens: 80.4 vs 82.7; v1 -0.5..0.8)
  and LOSES wherever multi-group experts make the consumer the limit (d16 other seeds +1.4..2, d32 +2.7). Hypothesis: the
  H200 decode consumer is integer-ALU-pipe-bound (dq = LOP3/SHF); S6 moves the scale decode from IMAD (FMA pipe) to LOP3
  (ALU pipe) and adds 5 SHF per stage. => any byte saving that adds ALU work in the consumer loses at d16/d32.
  Corollary worth a test (not done): move dq work FROM the ALU pipe TO the FMA pipe (IMAD.SHL / IMAD-based shifts for
  rotl(q,10/13), as o3 already does with *16648) -- the s17 search minimized instruction COUNT, not ALU-pipe load.
- Main: TK2F_S6 = 0 (code kept; kernels compiled only when TK2F_S6 or DIAG via #define TK2F_S6K). Layer port script
  tools/n20/patch_layer_s6.py (applies to r-layer 9937f2e moe_src.py, layer test 13/13) NOT handed off: it loses.

## Night4 session 21 (2026-10-27 05:13Z - ; moe checks 12/20 used today) -- moe_twq K-split tail (TWQ_SPLIT)
- Why: the layer runs moe_twq at v16 sliding (H200 tuner: twq -11 us vs twm), and twq had NO K-split tail; the H200
  timeline (n15) shows ~22 us drain loss at v16 for twq (items 36-50 us, 3.3-3.6 waves). twm's split (s66) was -3.5 H200 us.
- Built (18dfa4c + flags form): twq flags | items << 8 | log2(parts) << 16 (python TWQ_SPLIT / TWQ_SPLIT_M = (items, log2 parts),
  or directly in the TWQ_M / layer "twq" value). The last ~items gemm2 items, from an expert boundary (token results must not
  depend on their pass), run as 2^k K slices (64 >> k k-chunks). New combine: every part __stcg's its FP32 partial, threadfence,
  ticket (atomicAdd tick + j, st["tk"], reset by the last); the LAST sums the parts in part order 0..P-1 (deterministic, no spin)
  and runs the epilogue. Partials 16384 floats per part (python allocates items << k * 16384). C++ twq_flags() clamps items to
  the buffers. tq_item takes nkc / part / tick / s_flag / h / np.
- Correctness: tools/n21/ks_t.py 0/78 bad (random + skewed bs33-257, layer v16/v32 routings; (66,1) (33,2) (66,2); eager +
  3 replays bitwise equal = deterministic; vs unsplit <= 1 bf16 ulp). bench test 15/15. Flags form == TWQ_SPLIT form bitwise.
- H100 moe-only, layer routings (tools/s22/proxy.py, 3 rounds, us): v16 q4 220.6 | halves s16 215.5 s24 213.3 s33 212.9 s50 213.7
  s66 213.3 s80 214.2 s100 216.6 s132 224.3 | quarters s16 217.0 s24 217.3 s33 217.3 s66 219.7 s100 223.8 (quarters lose to
  halves: per-item fixed cost). v32 (twq, not what the layer runs there) q4 285.0 | s50h 282.4, others ~tie.
- H100 IN-LAYER (copy of r-layer aa08453 + tools/n21/patch_layer_twq.py, TUNE {}, twq forced at T=128, 4 rounds x seeds 7/11/99):
  sliding twq4 309.0 | s24h 304.8 | s33h 303.9 | s66h 309.0 (per seed s33h +1.2/-7.3/-9.1); full 427.1 | 427.2 | 432.8 | 428.8
  (per seed +2.6/+9.9/+4.7). Correct (scored verify 16 x 3 seeds both layer types, 0 bad). Full loses on the H100: unexplained
  (same routings; the full layer's context differs). => TUNE candidate per (T, sliding), never a default without the H200.
- H200 check 1ebfc1e3 (18dfa4c + DIAG, 2.225x test run, never submit), moe-only layer routings, MEAN 5 seeds, us (b = repeat):
    v16 q4 210.1/209.6 | s24h 200.6 | s33h 199.3/199.3 | s66h 203.2 | s33q 202.3   (s33h every seed better: -3.5..-14)
    v32 q4 273.0/272.6 | s24h 270.1 | s33h 271.4/270.2 | s66h 269.9 | s33q 270.5   (layer runs twm + hot at v32: n/a)
  => **twq + 33-item half split = -10.6 H200 us at v16** (~5% of the experts; ~+0.3..0.4% layer if sliding takes it). Quarters
  lose to halves on the H200 too. Handed to layer as TUNE candidates at T=128 (twq 73988 = 4|33<<8|1<<16, 73996 = 12|...).
  Decode shapes unchanged (dec = sq only this run: d1 29.8, d16 65.2, d32 86.4, v1 64.4).

## Night4 session 22 (2026-10-27 05:35Z - 05:50Z; moe checks 12/20 used today, none this session) -- DQW: dq() on the FMA pipe DEAD; consumer is latency-bound
- Tested n20's corollary ("H200 decode consumer is ALU-pipe-bound: move dq work to the FMA pipe"): DQW = 1 computes dq()'s two
  rotations as mul.wide.u32 q, 2^10 / 2^13 (IMAD.WIDE.U32) + one LOP3 (lo | hi) & M (ptxas keeps it: 5 ALU-pipe ops per dq
  instead of 7). Bitwise = DQW 0 (tools/n22/dqw_ab.py, all layer routings), bench test 15/15.
- H100 moe-only, layer routings (cold, 2 rounds): d1 +0.6, d16 +0.2, d32 +1.0, v1 +1.1, v16 +3.5, v32 +15.7 us.
  Consumer-bound proxy (tools/n22/dqw_warm.py: 1 expert, L2-resident, 5 calls/graph): T8 +0.2, T16 +1.9, T32 +3.9, T64 +15.3.
  The SM is the same on the H200, so no judge check spent. DEAD (code kept: #define DQW / python DQW = 0; _lib() caches one
  library per DQW value, kernel_fn uses _lib() in DIAG builds so a DIAG variant {"DQW": 1} would work, at +~95 s compile).
- WHY (ncu, tools/n22/ncu_pipe.py, moe_tk2f gs8 SQ, 1 expert T=32 warm): the consumer is NOT pipe-bound:
    DQW 0: 39.7 us, ALU pipe 50.0%, FMA 28.8%, FMA-heavy 33.9%, tensor 17.0%, issue active 49.9%
    DQW 1: 44.2 us, ALU 34.8%, FMA 35.1%, FMA-heavy 49.3%, issue active 43.9%  (same instruction count 13.75M)
  9 warps/SM (8 consumers + producer, 160 regs, occupancy 14%), 4.49 cycles per issued instruction per warp. Stall reasons per
  issue: selected 1.0, wait 0.92 (fixed-latency chains dq -> HMMA), long_scoreboard 0.79, not_selected 0.44, math_pipe_throttle
  0.40, barrier 0.29, dispatch 0.20, short_scoreboard 0.16. SASS stall samples spread thin: top = gemm2 readiness spin (BRA 4.9%
  + CCTL.IVALL 4.0% from the ld.acquire), HMMA 3.8%. => the n20 "ALU-bound" reading was wrong; S6 lost through added
  dependent instructions (latency), not pipe load. Levers for the decode consumer are ILP/latency (more independent chains
  per warp, fewer dependent ops per dq), not pipe balancing; the longer IMAD.WIDE latency is what made DQW lose.
  (Consistent with s24: issue ~50% with 2 warps/SMSP; s24's latency fixes -- stage-ahead LDS, 2 CTAs/SM -- and s13's 2 acc
  chains (PF & 8) are dead, so the decode consumer has no cheap lever left; the gemm2 readiness spin is ~9% of stall samples
  in the 1-expert proxy only, and per-segment counters were 0 on the H200 (s12).)

## Night4 session 23 (2026-10-27 05:45Z - ; moe checks 13/20 after this session's DIAG) -- HOTQ: moe_hot<NH 64> beside moe_twq at v16
- Skipped s21's NEXT (store-then-ticket for twm): twm's s11 combine takes the ticket only AFTER its main loop, so the
  "spin" on the other half is just the other half's store (~0); the twq gain came from twq having no split at all.
- Why HOTQ: layer v16 routings (5 seeds) have 17-19 experts, the top 2 with 36..49 tokens. twq (what the layer runs at v16)
  has only 32-token passes (64 do not fit its registers), so those experts are dequantized twice (~10% extra dq work).
- Built: moe_hot<NSTG, SA, NH> (NH 64: experts with 33..64 tokens, one N 64 pass, x box m[3]; NH 128 = s29 unchanged), chosen
  by moe_set_hot bit 4. moe_twq flags bit 20 = hx (as twm's hx): skip 33..64-token experts, no griddepcontrol.wait at entry,
  wait at the end (moe_norm, twq's PDL secondary, must see moe_hot's Y). Python HOTQ_M = {M: grid} or the twq flags
  q | grid << 24 (bits 24..30; the layer's "twq" TUNE value). Off by default.
- Correctness: tools/s23/hq_t.py 0/26 bad, BITWISE equal to plain twq (random/skewed bs33-257 + layer v16/v32, eager + 3
  replays); flags form == HOTQ_M form bitwise; bench test 15/15.
- H100 moe-only, layer v16 routings (tools/s22/proxy.py, 3 rounds, us): q4 220.5 | h8 318 h12 252 h16 225.5 h24 214.8 h32 215.3
  h40 216.2 h48 217.3; s33h (n21 split) 214.5 | s33h32 210.0 s33h40 209.8. (H100 twm 195: the H100 misranks twq/twm, DRAM-bound.)
- Layer port: tools/s23/patch_layer_hotq.py <moe_src.py> (applies to r-layer moe_src.py @ s75), then TUNE at T=128
  {"twq": 73988 | 32 << 24} (= 536944900).
- H200 check 80af8c44 (8561d6c + DIAG, test run), MEAN 5 seeds, us: v16 s33h 201.4/201.7 | s33h24 194.9 | s33h32 194.8/195.0 |
  s33h40 195.6 | q4h32 202.5 (seeds s33h -> s33h32: 7 -3.5, 11 -2.4, 13 -0.7, 17 -13.2, 99 -13.0). v32 (twq 4, no hot) 274.
- H200 check 3187c4e7 (9d8e201 + DIAG): s33h32 196.5/196.6 | s24h32 197.3 | s50h32 195.2 | **s66h32 191.8** (seeds 7/11 203 -> 190) |
  s33h16 196.1 | s33h20 195.6 | q4h24 204.1. With the heavy experts gone from twq, a longer split tail pays (n21 alone: s66 lost to s33).
- H200 check ecb491a2 (caf38cb + DIAG): s66h32 191.1/191.4 | s80h32 193.2 | s100h32 195.7 | s66h24 194.9 | s66h40 193.5 | s66q32 194.5.
  => layer TUNE[128] {"twq": 536953348} (= 4 | 66 << 8 | 1 << 16 | 32 << 24): -10.4 us vs the n21 candidate 73988 (201.5), -19 vs
  plain twq. Split variants differ from s33 by <= 1 bf16 ulp (split sums differ; each is deterministic). Seed 13 (one 36-token hot
  expert) stays ~204: unexplained (hot pass = 4 extra tokens' dq saved only).
- H100 does NOT track this (per seed s66h32 vs s33h: 7 -9.5, 11 -1, 13 +12, 17 -7, 99 -6): use the H200 diag only.
- NEXT ideas: a 16-warp hot pass (twq structure, RT 1 x N 64) if the 8-warp hot items limit seeds 7/11/13; hot64 at v32 would
  need hx in moe_hot as well (twm single-passes 33..64 already, so little to gain there).

## Night4 session 24 (2026-10-27 06:27Z - ; moe checks 16/20 after this session's DIAG) -- TWQ_NE: NORM_EARLY for moe_twq
- On paper first (routings decoded from _DIAG_R): a v32 "hot chain" (moe_hot<128> -> moe_hot<64> -> twq) saves NOTHING: twm + hot128
  already does exactly 1 dequant pass per expert on every layer v32 seed (passes: twq 24-27, twm 20-23, twm+h128 = chain = #experts).
  v16 twq + HOTQ is also 1 pass/expert. Remaining big-M gap = per-pass consumer efficiency + fixed/drain, not extra passes.
- Layer profile 2f7349ee (v16 sliding): moe_twq 183.4 + moe_norm 32.3 us (norm = twq's PDL secondary, waits for ALL of twq);
  twm already runs NORM_EARLY (moe_norme: per-token waits on per-expert gemm2 flags meta[100 + e]). Built the same for twq:
  s_nd2[a] = gemm2 item ends of slot a (K-split items count once per part: + (npt - 1) * split items), after each gemm2 item
  bar.sync 1, 512 + threadfence + atomicAdd(done + 32 + e); the one reaching s_nd2 adds 1 << 20. A split's last part writes Y in
  tq_item before the count. HOTQ's moe_hot<64> already flags its experts. Python TWQ_NE = 1 or twq flags | 1 << 21 -> moe_norme.
  (The count always runs in NORM_EARLY builds; TWQ_NE only picks moe_norme.)
- Correctness: tools/s24/ne_t.py 0/78 bad, BITWISE equal to TWQ_NE 0 (q4 / s33h / s66h32, random + skewed bs33-257 + layer
  v16/v32, eager + 3 replays); flag form == TWQ_NE form (5/5 bitwise); bench test pass.
- H100 (tools/s22/proxy.py, 3 rounds): v16 s66h32 210.3 -> 208.7, s33h 213.6 -> 209.5.
- H200 check 11f0161e (bdde3b7 + DIAG, test run, never submit), MEAN 5 seeds: v16 s66h32 192.6/192.1 -> ne 191.3/190.9 (every
  seed -0.4..-1.8); s33h 200.6 -> 198.5; v32 (twq, not the layer's v32 path) 276.7/275.9 -> 269.7/269.6 (-6.5).
  => layer TUNE[128] {"twq": 539050500} (= 536953348 | 1 << 21): ~-1.3 us at v16 vs the s23 candidate. Port tools/s24/patch_layer_ne.py
  (applies to r-layer moe_src.py @ a90cadb, with or without tools/s23/patch_layer_hotq.py; layer's halves-only n21 copy handled).
- H200 check fccff52d (tools/s24/mktl_twq.py copy: TWM_TL 1 + DIAG_TL, test run, never submit): twq per-CTA timelines on the
  layer's v16 routings (us; items = 1 MB weights + scales each, same bytes for gemm13 and gemm2 items):
    kinds (g2 class reuse N: n, mean item us), s66ne seed 7: g13 N8 160 x 42.6 | N16 96 x 48.7 | N32 32 x 62.3 | g2 whole-K N8 41.5 (q4)
    | g2 K-half N8 144 x 26.3 | N16 49 | N32 60. Same on every seed (N8 43, N16 49-50, N24 58, N32 62).
  => N8 items run at ~3.45 TB/s aggregate (near the H200 stream rate); N16 +15%, N24 +35%, N32 +45% for the SAME bytes = the
  consumer (dq + wgmma not overlapping; at N32 per stage ~1770 clk vs ~1230 at N8). K halves are ~20% slower per byte than whole items.
  s66h32ne: hot SMs' twq CTAs start at 70-100 us (start max), exit min/med/max 141-161 / 154-176 / 171-188, busy/G 135-156.
  s66ne (no hot): loss 17-23 us. q4 (no split): loss 30-33, last CTA +20 us after the median.
  diag12 same check: v16 s66h32ne 189.8 | s66ne 200.9; v32 twq 268.5.
  Next lever for v16 (and v32 twm): the N16-N32 consumer (per-step wgmma wait / dq latency chain), not bytes or passes.
- TQ_DEEP (#define, default 0): tq_item with 4 A-fragment sets / wgmma.wait_group 3 (stage release moved to j0 == 2) instead of
  2 sets / wait 1. Bitwise = main (tools/s24/deep_t.py 0/36; the TQ_DEEP 0 refactor = previous commit 0/36). H100 layer v16/v32
  cold: tie (+-0.3 us); 1-expert L2-resident proxy (tools/s24/deep_warm.py, T40 with 8/16/24/32 tokens): tie. DEAD: the wgmma
  wait is not what limits the twq consumer. (Proxy also shows one twq CTA at N8 needs ~50 us per 1 MB item from L2 = consumer-bound
  even at N8; N32 +20%.)
- SASS of moe_twq<8, SA> N8 k-chunk loop: 166 instructions per warp per k chunk (stage): ~100 dequant (43 LOP3, 32 HMUL2, 18 SHF,
  IMAD), 8 HGMMA + 8 WARPGROUP, the rest (~50) = mbarrier wait/arrive, 3 LDS, descriptor/address math (LEA/R2UR/UIADD3),
  branch. 16 warps -> 664 issue-clk per stage per SM vs H200 measured ~1330 clk (43 us / 64 stages @ ~1.98 GHz) = ~50% issue
  efficiency (same as tk2f, ncu s22). Stream rate needs ~1170 clk/stage. Levers: 2 k-chunks per stage (KPS 2, as tk2f: -3..4% there)
  to halve the per-stage ~50 instr, and for N16-N32 the HGMMA cost itself (N32 items +45%).

## Night4 session 25 (2026-10-27 07:01Z - ; moe checks 18/20 after this session) -- DIAGNOSIS: twq never beat twm; n21 regressed twq
- Lead 06:41Z asked why the n21 K-split (moe-only H200 v16 209.9 -> 199.3) made the layer's v16 sliding SLOWER (tuner 298 vs
  287, forced scored 282.2). Answer: it never won. Every twq DIAG since s17 was compared to a twq baseline (q4), never to twm,
  the layer's T=128 default. Same H200 diag on the layer's v16 routings across checks (decode controls stable to 0.4 us):
  twm 172 (x13/x14) | q4 177.9 (n13) -> 186.8 (n19) -> 210.1 (n21). n21's split code made EVERY twq item slower.
- Same-check H200 confirmation, check b14b8be4 (9c4808e + DIAG, test run): v16 twm 171.6/169.4 (repeat) | q4 214.5/214.3 |
  s66h32ne (HOTQ+split+NE, the s24 "best") 191.9 | twm + TWF_SPLIT 66 169.9 (tie); v32 twm 218.5/217.7 | q4 278.1 | s66h32ne 269.9.
  H100 dry run of the same DIAG agrees (twm 193.8, q4 223.1, s66h32ne 208.8, twms66 191.7).
- Layer scored data (16 judge profiles in r-layer/profiles, v16 sliding by tuner pick): twm-picked runs 272.3 mean (268-276,
  n=9), twq-picked 277.0 (268-285, n=6). The layer tuner's synthetic routing is realistic (SYNTH_P); its twm/twq picks are
  within its noise. The judge profiles' per-kernel times are single runs and too noisy to compare (twm 147..199 us).
- Regression bisect (H100 moe-only proxy, layer v16 routings, tools/s22/proxy.py with SUB=<module> from /tmp/s25):
  pre-n21 4c4cd4e q4 200.4/202.3, twm 194; n21 18dfa4c q4 216.0/218.7; main q4 219.4/221.5; main with tq_item's
  `if (part)` split-combine block compiled out: 204.6/206.3. So ~15 of ~19 us = that block (SASS: moe_twq<8,1> 4.3k -> 17k
  instructions, LDL 9 -> 57; the k-loops themselves only 164 -> 166 instr, still pipelined).
- Fix attempt (NOT kept, /tmp/s25/sub_spl_attempt.py): split as a template flag tq_item<..., SPL> (constant nkc 32/64, offsets
  from kernel params after the loop, gemm2-only epilogue): code back to 7.8k instr, q4 207.5, s33h 202.9 (H100) BUT ptxas C7512
  "wgmma serialized due to insufficient register resources" for moe_twq (every loop: DEPBAR 0x0 after each HGMMA). Tools:
  tools/s25/ptx_c7512.sh (rebuilds the lib, prints C7512 count + loop shapes), tools/s25/sass_loops.py (HGMMA loop sizes).
  Not pursued: twq is ~40 us behind twm even pre-n21 at v32 and ~15 at v16; HOTQ's best case after a full fix ~tie with twm.
- Posted the diagnosis to layer (COORD Requests 07:25Z): drop the twq options at T=128 (incl. s76 HOTQ 536953348/536870916);
  my HOTQ / NE requests withdrawn. DIAG_VARIANTS now always carry a twm control (+ repeat).
- LESSON: every DIAG must contain the layer's default path as a control; any kernel change must be re-timed against an
  older build of the same variant (proxy SUB=<old module>), not only against sibling variants in the same build.

## Night4 session 26 (2026-10-27 07:34Z - ~08:05Z; moe checks 18/20, no judge runs) -- audit + idea triage, no build
- PDL load-order audit of every moe kernel (lead 06:50Z/10-26 ask; SASS of the main build, /tmp/moe26.sass): loads before
  the first ACQBULK (griddepcontrol.wait) are only prepared constants: the e4m3 LUT copy in twm/twq/twf/tw/moe_hot
  (LDG.CONSTANT at +0x600/+0xc00 = 384-thread LUT fill) and moe_norm's w1 (norm weight). moe_gather, moe_norm2,
  moe_routegather: no pre-wait loads (routegather reads ids/X with __ldcg, layer s52 request done). moe_norm's
  NORM_EARLY path never runs griddepcontrol.wait: ids/tw by __ldcg, Y by __ldcg after an ld.acquire of the expert flag,
  and twm triggers its dependents only after its own wait (route/gather/router complete): OK. moe_tk2 (not a default
  path) has 33 pre-wait LDG.CONSTANT: re-audit before ever using it.
- Ranked idea -9 (warp-specialized twm consumer, written in s25) was already dead: s19 mb_ws.py measured full ~= T + M
  with dequant warps and SS-MMA warps separated, and SS slower than RS. STATE corrected. mma.sync for N8 items: no H200
  gain (N8 items already at the per-SM HBM share, s23).
- Judge layer profile baae0e41 (v121) for reference: d32 sl tk2f 68.8 us (9-11 experts x 6.97 = HBM fit), v16 sl twm 180 +
  moe_norm 38 (PDL wait), v32 sl twm 230 || moe_hot 211 (grid 32). The expert kernels sit at their fits; what remains in
  moe's lane is the big-M drain (~20 us/call) and tk2f's fixed ~14 us (mostly hidden by layer TK2FX/TK2FJ).

## Night4 session 27 (2026-10-27 07:54Z - ; moe checks 19/20 after this session's DIAG) -- SCB scale decode (kept), overlap closed
- Lead 07:42Z asked to cut twm's consumer instruction count: (a) byte_perm LUT e2m1->bf16, (b) one HMUL2 per pair for the block
  scale, (c) warp-specialized dequant re-checked on twm.
  (a) no gain: the 8-op extraction (o0 = q & M, rotl 10/13 & M, IMAD for the rest) is minimal. Exhaustive search
      (tools/s27m/dq_tiling_search.py): no layout reaches 7 ops with rotations / shifts / PRMT, single-word or 2-word funnel
      windows, or rotation + one IMAD.WIDE pair. A PRMT LUT needs 8 selector bits per bf16 value (we have 4) and nibble bit 3
      is PRMT's sign-replicate flag. (b) already the case (4 HMUL2 per 8 weights). (c) see below: no overlap is possible.
- NEW microbench (tools/s27m/mb_x.py, from s22/mb_pp.py; 1 CTA, cycles per twm stage, each WG times itself):
    RT4 N32: WG0 dq alone 522 | WG1 RS-MMA alone 359 | both at once in DIFFERENT warpgroups 862/861 (sum 881)
             WG1 SS-MMA alone 623 | WG0 dq || WG1 SS 979/981.  RT2 N64: 337 | 357 | 602/555 | SS 372 | 608/551.
    Same with pure FFMA chains instead of dq (DQV=4): 516 | 354 | 873/880. And int-XOR only (DQV=1): same.
  => on this H100, tensor-core work and CUDA-core work by OTHER warps do not overlap either (RS or SS A). The s18/s22
  "dq and wgmma don't overlap" is SM-wide, not RS / register / same-warp: every warp-specialized or pingpong design is
  closed, and consumer time ~= ALU(dq) + tensor-active(FLOPs incl. N padding). (N24 class already exists in twm.)
  Padding bound (tools/s27m/pad_bound.py, H200 item model): exact N per expert would save only 0-2.6 us per v16/v32 call.
- twm in-kernel vs microbench (tools/s27m/twm_itl.py: TWM_TL build, ONE L2-warm expert, per-item durations; H100):
  items N8 36.6 | N16 40.3 | N24 41.6 | N32 46.2 us vs the bare dq+MMA microbench 28.6 (N8) / 37.7 (N32) us per 64 stages
  = a constant ~8-9 us (~230 cycles/stage) of in-kernel overhead. Ablations (timing-only /tmp copies):
    full-barrier wait only every 2nd stage: -0.9 us/item; scale decode removed (constant scales): N8 -2.7, N16 -2.9, N24 -0.7.
- KEPT (40de385) SCB: scp<P>(w) = __byte_perm(w, 0, 0x4140 / 0x4342) * 16 + 0x77007700 decodes TWO e4m3 scales (one per
  bf16 half); blo/bhi (PTX mov.b32 {l,h} -> {l,l}/{h,h}) fold into HMUL2's .H0_H0/.H1_H1 operand selectors (SASS checked:
  64/64 HMUL2 with selectors, PRMT 8 -> 4 per twm k chunk, loop 277 -> 269 instr). Applied at every SA scale site (tw_item,
  twf, twq, tk2f/tk2h). Bitwise = before on all layer routings (tools/s27m/scb_ab.py, 0 diffs, old-vs-old 0); bench test 0 FAIL.
  H100 cold layer routings (old -> new us): d1 36.2/36.2, d16 82.6/82.3, d32 104.7/104.2, v1 84.0/83.5, v16 195.9/194.0,
  v32 235.1/232.1; warm items N8 -1.9 (-5%), N16 -1.8, N24 -1.5, N32 -0.2. Layer port: tools/s27m/patch_layer_scb.py
  <moe_src.py> (applies to r-layer main's copy: 5 sites, 0 left).
- DEAD TWM_PAIR (tools/s27m/twm_pair_dead.diff): producer puts each (even, odd) stage pair on full[odd], consumers wait
  once per 2 stages. Bitwise ok, but v32 233.2 -> 236.3, v16 195.2 -> 195.7 cold (one stage less lookahead per wait).
- Judge: check 5ad98078 (40de385 + DIAG, test run) 2.2576x; H200 diag12 vs b14b8be4 (same DIAG_VARIANTS): d1 30.0/30.0,
  d16 65.0 -> 64.2, d32 85.8 -> 84.5, v1 64.3 -> 64.5, v16 twm 171.6 -> 166.2 (every seed -3.7..-7.9; repeat 169.4 -> 165.0),
  v32 twm 218.5 -> 213.7 (repeat 217.7 -> 213.4), q4 / s66h32ne also -2..-3. Plain main check ed55a5c0 2.2635x
  (bs1 26.8, bs8 63.6, bs16 89.7, bs32 124.8, bs128 206.4, bs256 245.6 us) -> submit febfcb1f 2.2469x (new board best, was 2.190x).
  Checks 20/20 used on 10-27; 1/10 submits.

## Night4 session 28 (2026-10-27 09:10Z - ~09:50Z; moe checks 20/20 used on 10-27, no judge runs) -- twm per-stage overhead bisected: no lever
- Lane per lead 08:55Z: twm/twf at v16/v32 only. Goal this session: find the ~230 cyc/stage of in-kernel overhead (s27) with
  timing-only ablations on the 1-expert warm proxy (tools/s27m/twm_itl.py; 16 gemm13 + 8 gemm2 items, so 16 CTAs: no L2 limit).
  Ablation builds: tools/s28/mk.py <v> (from /tmp/s28/sub_nc.py = TWM_TL build + NOCOPY producer: after the first ring fill the
  producer only arrives on full[], no copies). gemm13 item medians, us, H100 (GPU shared, +-0.5..1):
                     N8    N16   N24   N32
    TL (copies)     35.1  39.0  39.8  47.6
    NOCOPY          32.7  36.9  40.3  45.8   (feed = only ~2 us of the overhead)
    + tid in regs   32.6  37.8  40.6  44.1   (S2R SR_TID.X is rematerialized at every stage top; fence_r keeps offsets)
    + 2x mbar_wait  35.1  38.2  42.0  46.6   (the full wait costs ~1-2 us/item)
    + no weight LDS 33.9  36.9  37.3  43.5   (LDS->dq latency ~2-3 us at N24/N32)
  vs bare dq+MMA microbench (mb_x MODE 3, same session): N8 28.9, N32 38.1 us/64 stages. So the overhead is several ~1-2 us
  latency items (wait, LDS latency, tid remat, prologue/epilogue), no single big one.
- ncu (tools/s28/drv.py + stall.py/loop.py, NTOK 32): loop = 280 SASS (LOP3 83, HMUL2 64, SHF 34, IMAD 17, HGMMA 16, ~40
  uniform/control ops). Stall samples in the loop: selected 21%, wait 20%, dispatch 16%, not_selected 8%, short_sb 5%
  (PRMT after the scale LDS.64: 86% short_sb); first HGMMA of each group holds ~27% of loop samples (tensor-pipe wait).
- tid fix on the real kernel (sub_tidm: wl/sl offsets kept in registers with fence_r, all tw_item users): bitwise, H100 cold
  layer routings (/tmp/s28/ab.py = scb_ab.py on v16/v32, 3 rounds): v16 193.5 -> 195.2, v32 232.7 -> 231.6 = noise. DEAD.
- Microbench tools/s28/mb16.py (mb_x + 512-thread variants): same 512-row stage as 16 warps x RT2 vs 8 x RT4, cycles/stage:
    N8  ALU/MMA/both 750/421/792 -> 691/416/769;  N16 750/445/902 -> 691/429/877;  N32 750/532/1046 -> 691/516/985
  i.e. 4 warps/SMSP buy only 3-6% even in the microbench: the ALU pipe is near its limit (7 ALU-pipe ops per 8 weights at
  16 lanes/SMSP = 448 cyc/stage + the MMA ~512 at N32, which sit near peak FLOPs: 32 m64n32k16 per stage = 16 cyc each).
  This design is moe_twq (s17), which lost to twm on the H200, consistent with that. Don't rebuild it.
- Readings: twm consumer ~= ALU-pipe time + tensor time + ~150-250 cyc/stage of latency bits that each cost ~1-2 us/item and
  sit under the noise when fixed alone. The N32 tensor time is real FLOPs (no FP8 path: activations stay BF16). The kernel-level
  lane for twm is exhausted; what is left is scheduling (v16 drain ~17 us after s66, first wave ~10 us at ~2.7 TB/s, n15).
- In-layer check (layer profile 19163546, H200): v16 sl twm 157.8 us (moe-only diag 166 before SCB), so twm runs in-layer
  as fast as standalone. The full-layer row (193.7) includes overlap/wait next to _gw_verify, not slower expert work.

## Night4 session 29 (2026-10-27 09:30Z - ~10:00Z; moe checks 20/20 used on 10-27, no judge runs) -- big-M lane bound; no build
- Goal (s28 NEXT): twm scheduling at v16/v32 (drain, first wave). Re-read s15/s19/s20/s25 + layer NOTES; nothing new to build:
  drain fixes tried = K halves (s66: the only winner, kept by layer), AUTO split, quarters/eighths (lose: part overhead),
  row-split tails (TWM_TAIL/TAIL13), orders (o76/o4), HOT_TWG grids; drain-time L2 prefetch of the unpool W = layer s6
  (twf counter-out, H100 +1..5), UPF3 (H200 DIAG b1ee542c -4% d32), UPF4 (s26). First-wave low HBM rate is not a loss by
  itself (s19: order only moves per-SM-bound work around).
- Bound (H200 profile 19163546 + s15 item times): N8 items run AT the per-SM HBM share (37.6 us/1.13 MB at 4.0 TB/s); only
  N16/N24/N32 items are consumer-bound (44/47/50 us). v16 sl: twm 157.8 = bytes ~127 + consumer excess ~9 + drain ~17 + start.
  v32 sl: twm 205.7 || moe_hot 154.5 (32 SMs) vs ~624 MB / 4.0 = 156: ~30 us N16-N32 consumer excess (ALU+tensor, s27/s28:
  dq at the 8-op minimum, MMA at N32 = tensor peak for real FLOPs) + ~20 drain/hot imbalance. Every listed lever on these
  is measured dead; what remains are 1-2 us latency bits per item (s28) that each sit under the judge noise.
- Ideas rejected on paper this session: FP8 wgmma (activations must stay BF16; a 3-term FP8 split of x costs 1.5x tensor
  time); scale after the MMA (needs per-k16 temp accumulators: same FFMA count at N8, more at N16+, kills the wgmma ring);
  per-block scaled PRMT LUT (LUT built per 4 weights per thread: dead, as s27); exponent-add scale fold (breaks zeros);
  unpool started on finished tokens (64 MB W > L2: streams W twice).
- Asked the lead (COORD Requests) to close moe's big-M lane or name a new one.

## Night4 session 30 (2026-10-27 09:36Z - 09:45Z; moe checks 20/20 used on 10-27, no judge runs) -- waiting for a lane; no build
- No lead answer yet to the s29 Requests line (close big-M lane / reassign); last Human decision 09:28Z. layer s81 closed tk2f
  and also asks for a lane; proj s76 is idle in lane (b). No new work was started, so the shared GPU stayed free.
- Paper checks only: (1) twm N8 items are near consumer-bound too: ~792 cyc dq+MMA + ~230 overhead per ~17 KB stage
  = ~33 GB/s per SM at 1.98 GHz, = the 4.0 TB/s / 132 share, so slowing N16+ items does not leave HBM time for N8 items.
  (2) smem byte-LUT dequant (PRMT byte -> LEA addr -> LDS bf16x2 pair) = 2 ALU ops per 2 weights = today's 8/8: no gain, plus MIO load.
  (3) cvt e2m1x2 is sm_100+ only.
- NEXT: whatever the lead assigns (see COORD Requests moe n4s29).

## Night4 session 31 (2026-10-27 09:53Z - ~10:30Z) -- NEW LANE muse_attention_decode (lead 09:45Z LANE DECISION 2); 2 attention checks
- Seeded muse_attention_decode/ from ~/attn 82866c6 (code = main 410dcb6). Its first check was a duplicate of 06456b7e
  (2.570x, node aa266yoh): per-shape table in STATE. Scored median_ns = raw replay - ~4.28 us.
- Merged agent2's never-H200-checked branch agent2-exp/halfpfkr (3-way merge on base 4c00ee9, clean): verify 32 sliding
  nvs `cl` (S 2 pairs as 2-CTA clusters, 9 KB cp.async.bulk DSMEM exchange, in-kernel merge/norm/gate, no _combine_hb)
  + split start barrier, csl/nds page-table L2 prefetch (batch > 1), nds K-early barrier. Check **362ff219 2.5447x**
  (node ojkcfmco, slower: unchanged shapes +0.3..0.8): v32 sl **14.07 vs 15.51** (real), d32 sl -0.10, d16 sl +0.17,
  but v16 sl +0.65: the cl build carries 18 KB more smem + gate registers into the S 4 path.
- Fix: verify 16 runs a second nvs build = 410dcb6's exact _NVS_SRC (`_NVS0`, name muse_nvs_a2s13h); the cl build only at
  batch >= 32. Plus csl/vtk owner merge fast divides (`__fdividef` for vscale/den and the sigmoid gate; H100 paired
  d1 sl -0.19 9/9, v1 sl -0.10 8/9, outputs bitwise). Check **18cf3bfb 2.5510x** (same node ojkcfmco): v16 sl 11.69
  (fixed), v32 sl 14.39, d1 sl 4.71 (+0.07 vs 362ff219: fast divides invisible on the H200), v1 sl 5.69 (-0.13).
  Unchanged d1 full moved 12.81 -> 12.07 between the two same-node runs: per-shape noise up to ~0.7 us; batch-1 wins of
  0.1-0.2 us can't be seen in one check. Not submitted (board 2.566x; rule: check must beat it).
- csl timeline (tools/s31/cslstamp.py, H100, clock64 per CTA; globaltimer is 0.256 us-granular here, useless): cold,
  from first CTA entry: K/V landed 1.58 (warm 0.96), P@V done +0.45, premerge sync +0.05, cluster wait +0.16, owner
  mbarrier +0.37 (= slowest of 16 CTAs + DSMEM push), owner merge+store +0.47 (fast div: +0.33; of that 16-partial loop
  0.22, norm/gate/store 0.11). The page-table and K stamps coincide (shared load scoreboard), so pt vs K is not split.
  FP8 mma.sync is emulated on sm_90 (F2FP.F16.E4M3.UNPACK_B + HMMA.16816): a native F16 QK/PV would save only ~16 instrs.
- DEAD: W4 owner merge (4 warps x 4 partials + bar.sync; stamps -0.08) whole-kernel d1 sl -0.03 5/9, v1 sl +0.13 0/9
  (vtk is 2 CTAs/SM); csl new-token reciprocal instead of / kscale, / vscale: +0.03 (1/9).
- Handoff: v32 sliding cl (+ the separate pre-cl nvs build for S 4) to layer (r-layer attn_src.py has neither).

## Night4 session 32 (2026-10-27 10:24Z - 12:40Z) -- attention lane: H200 DIAG of batch-1 full (d1/v1 full)
- Reasoning: d1 full 12.07 / v1 full 16.75 us (floor 4.3) are not in layer's claim (layer = d16/d32/v16/v32 full) and are
  worth 0.67 / 0.51 %/us. ~/attn exp/hdiag5 (H200 batch-1 random/dense gather floors + plan variants) was queued for 2 days
  and never run. Ported it onto main as tools/s32/mkdiag.py (writes a DIAG copy; NEVER submit): d1 full variants d1a/s3/s4/
  p1023/p255/b16s4/b(repeat) + hfloor R2/R4/R8/dense; v1 full a/s10/s12/p255/p255s5/p191/b128s4/b32s12/b(repeat). Each
  variant runs after the real output with an L2 flush (512 MB read), kernels renamed by variant for the judge profile.
- Check 7df7c251 (DIAG v1, renamed kernels in the judge profile): useless, the profile lists only the top 12 kernels.
  Tool v2+: on the first EAGER kernel_fn call of a diag shape, each variant is graph-captured and replayed 40x round-robin
  after a 512 MB read (cold L2); prints `diag32 <shape> <variant> med/p25/min` minus an empty graph (grep the run log).
  Timed calls are main's code (diag runs once, before timing), so such a check's per-shape times = main's on that node.
- **Check 4fd8b5d5 (DIAG v2), H200 us (median, cold, graph replay - empty):**
    d1 full: main 12.77 | stages 3 13.76, 4 14.45 | splits 1023 13.36, 255 16.08 | BN16 s4 14.99 | floors: random
             gather all-in-flight (hfloor R2/R4/R8) 8.29/8.34/8.70, dense rows 7.74
    v1 full: main 17.09 | stages 10 17.58, 12 18.40 | splits 255 20.58, 255 s5 19.97, 191 19.07 | BN128 s4 19.38,
             BN32 s12 19.65 | floors R2/R4 7.89/7.94, dense 7.42
  => the H200 batch-1 full floor is ~8 us for both (pt trip + 20.5 MB random 128-B rows incl. launch). Headroom: d1 ~4.5,
  v1 ~9 us (v1 full on the H100 ~= its floor + consumer: the H100 hid this; s97's "<= 1 us H200 ceiling" was an estimate).
  Every plan knob loses on the H200 too.
- **Check 6b5011ed (DIAG v3)**: v1 full main 17.34 | Gluon gw (s94 _GW1, 127 splits) 17.82 | gw gfuse 127/255/263
  52.7/188/197 (the in-grid merge spins: dead at batch 1) | warps 8 23.8 | pre 0 17.68 | pf off 17.50 | fix 0 17.82 |
  splits 63 20.96 | stages 6 17.73 | qk FP16 19.10. (Locally gw255 stages 4 deadlocks: gfuse assumes 2 CTAs/SM.)
  Its timed shapes are main's code: **2.6193x** (fast node; d16/d32/v32 full 84.6/160.8/192.4, v32 sl 14.03).
- Submit **90b7ce85** = main (s31 code, never submitted; its timed code had just checked 2.619x): **2.5653x** (1468.1 us)
  = tie with board 2.566x (node noise). Scored 1/10 today.
- **Check b0d361bc (DIAG v4, data-level ablation: page table remapped so all gathers hit L2 (`L2`) or read consecutive
  rows (`seq`); timing only)**, H200 us: d1 full main 12.22 | all-L2 11.04 | seq 11.84 | floor 7.97 | dense 7.46;
  v1 full main 17.14 | all-L2 16.66 | seq 16.93 | all-L2 + stages 4 20.35 | all-L2 + splits 255 20.50 | floor 7.95.
  **=> batch-1 full is NOT memory-bound on the H200**: free DRAM saves only 1.2 (d1) / 0.5 (v1) us. The ~9 us v1 gap
  over the gather floor is per-program fixed latency + consumer chain + merge. Timed shapes = main: 2.6391x (fast node).
- **Check 0c60ebfc (DIAG v5)**, v1 full H200 us: main 17.02 | Gluon gw 127 17.42, 191 20.83, 255 22.88, 255 lag 2 23.94,
  255 stages 2 23.47, 255 all-L2 21.81 | gbal (264 programs, _combine_bal) hb2 cw4 21.63, cw8 18.59, hb1 19.81, all-L2
  20.82 | main all-L2 16.93. More programs per SM (2 CTAs/SM) LOSE even with free DRAM: the extra programs' fixed cost +
  wider merge outweigh the consumer parallelism. Timed shapes = main: 2.5891x.
- Readings: v1 full is bound by a serial per-program chain (~10 tiles x ~1 us in _attn_split, profile 12.6 us) + the
  _combine hop (2.7 us). d1 full likewise (_gd_full 8.2 + _red_fin 1.4; all-L2 only -1.2). Neither more programs nor
  deeper stages help. Plan knobs for batch-1 full are closed on the H200 (all measured here).
- NEXT (s33): per-program globaltimer timeline of _attn_split at v1 full ON THE H200 through the diag hook (stamps: program
  start, q ready, first tile landed, each tile's end, partial store, _combine start/end; print percentiles in the log):
  find whether the ~1 us/tile is the page-table->row dependent chain per tile (Triton does not pipeline the dependent
  pt load: then prefetch the whole split's page-table slice into smem/registers up front), the softmax/dot chain, or
  the launch/ramp. Same for _gd_full at d1. Only build once the H200 timeline names the cost.
- Budget 10-27 (attention): checks 7/20 used (5 this session: 7df7c251, 4fd8b5d5, 6b5011ed, b0d361bc, 0c60ebfc; all
  DIAG, never submit), scored 1/10 (90b7ce85).

## Night4 session 33 (2026-10-27 11:52Z - ~12:10Z) -- attention lane: v1 full per-program timeline (H100, no checks spent)
- Tool: muse_attention_decode/tools/s33/mktl.py (+ tl_tail.py): DIAG copy (NEVER submit) with clock64/globaltimer stamps
  in _attn_body (start, q ready, after _new, after the FIX pre-loop, per tile (TS 2), loop end, after _finish) and in
  _combine (start, after gdc_wait, loads done, end); on the first eager v1-full call it graph-replays TS 0/1/2 (+ all-L2
  page table) cold and prints 'tl33' lines. Local driver: /tmp/s33d/run2.py style (OV='{"splits":255}' env = plan override).
  Stamp cost: TS1 +0.9 us, TS2 +2.4 us (the per-tile l_i reduction); phase numbers below are from TS1.
- **H100 v1 full (main plan: 127 splits x 2 heads = 254 programs + 2 tsep, 10 tiles of 64 keys each), us:**
  q phase (start -> fp8 q in smem: seq/qkv/cos-sin/pt trips + PRE prefetch) 1.9-2.0 | _new 0.6 | FIX pre-tile 1.1 |
  loop 9.0-9.2 | _finish 0.5 | program total 13.1-13.6 | _combine: launched at ~1 us (PDL), released 0.48 after the last
  _attn_split end, partial loads 1.47 (4 MB FP32 from L2), done 2.0 us after the last program. Whole 16.0-16.4.
- Per tile (TS2): first 1.5, then a flat **0.94 us/tile**, the SAME with every gather in L2 (0.93-0.96). Not memory.
- **Per-SM throughput saturates**: splits 63 (1 program/SM, 20 tiles) 0.66 us/tile; 127 (2/SM) 0.92; 255 (4/SM, 5 tiles)
  ~1.67 incl. fill. SM tile rate 1.5 -> 2.2 -> 2.4 tiles/us. Splits 255 also doubles the merge (c_load 4.2-5.9 us, ends
  4.9 us after the last program): 21.5 us total. These H100 totals (63: 20.6, 255: 21.5) match the s32 H200 diag (20.96,
  20.50): for this shape the H100 IS a valid proxy (not DRAM-bound on either).
- ncu (H100): 123 regs, 41.7 KB smem, issue active 34%, warps active 12%; stalls spread (long_sb 21%, wait 14%, barrier
  14%, short_sb 12%). Loop SASS = 308 instr/warp/tile: MUFU 32, FFMA 32, FSEL 32 + ISETP 16 (tail masks on every tile),
  F2FP 32 (P and V cvt), LDS 16 + STS 4 (V fp8 -> fp16 via smem), 2 QGMMA + 4 HGMMA each followed by DEPBAR 0,
  3 BAR.SYNC + 2 MEMBAR.ALL.CTA + 2 FENCE.VIEW.ASYNC, 5 LDGSTS.
- DEAD (local, 2 rounds each): peel (unmasked full tiles) +0.3 us, mif (masks only when needed) -0.3 (noise). Removing
  20% of the loop's instructions does nothing: neither issue- nor memory-bound; it is the serial wgmma->exp->STS V->
  fence->bar->wgmma chain with 2 warpgroups per SM contending. ~/attn already lost with native CUDA (s71/s74/s92/s111),
  hand-pipelined FPIPE (s113), FA3 ping-pong (s101), FP8 PV (s90), more programs (s32 here). Verify 1 full is closed for me.
- Not tried / left: an intra-tile stamp breakdown (QK wait / exp / V convert+bar / PV wait) to say WHICH link of the 0.66 us
  single-program chain dominates; the merge's 1.47 us L2 load of 4 MB FP32 partials (BF16 partials are banned).
- **spec at verify 1 full** (seq = table width - 8, loop-form redo if the real length differs; the plan flag existed,
  off since s86's +0.35 before fix 1): H100 bench sweep 16.85-16.90 vs 17.44-17.50 (3/3); tests 16/16; redo path
  tools/s33/spec_redo.py (table padded by 5/64/300, fill -1/0) PASS; scored output bitwise = before.
  **H200 DIAG check 4fe17fa9** (tools/s33/mkab.py, round-robin cold): v1spec 16.22/16.24 vs nospec 16.83/16.90 = **-0.63 us**.
  Its timed shapes (= main with spec): 2.6136x. Same check, d1 full (Gluon _gd_full, spec on in main): d1a 13.14 vs
  d1nospec 12.58 (one pair; H100 3 pairs: spec -0.1..-0.36) -> second DIAG check with 3 pairs.
- **DIAG check 28658e59** (3 d1 pairs + 2 v1 pairs): d1 full spec 13.07/13.23/13.18 vs nospec 14.16/13.25/13.12 = tie
  (4fe17fa9's d1 pair was noise; d1 keeps spec). v1 full spec 16.34/16.45 vs 17.10/17.14: -0.7 us again (4/4 H200 pairs).
  Timed shapes 2.5809x (node). Handoff to layer: one line in attn_src.py's verify-1-full plan (spec=True).
- **Submit 957cabd7** (main ac5b67e + mkab tool only = spec at v1 full): **2.5962x** (1450.7 us), NEW attention board best (was 2.566x 06456b7e). v1 full 15.98 us (board run 16.60). Budget 10-27 attention: checks 9/20, scored 2/10.

## Night4 session 34 (2026-10-27 12:28Z - ~13:20Z) -- attention lane: v16 sliding in-kernel merge (closed by measurement/arithmetic); 0 checks
- Target: v16 sl (judge 11.54 cold; warm profile nvs<5,2> 7.6 + _combine_hb 2.0), STATE idea "drop _combine_hb via DSMEM
  cluster merge". nvs CTA = (request, KV head, split): v16 = 16 x 2 x S4 = 128 CTAs, 384 thr, 160-reg consumers.
- H100 (tools/s34_cl_v16.py, 3 rounds, NVS_CL_BATCH 32 vs 16 = cl build at S 2 with 2-CTA DSMEM merge, no combine):
  v16 sl 12.45/12.78/12.70 (S4 + combine) vs 14.13/14.59/14.37 (cl S2): **+1.7 us**; v1/v32 unchanged (control).
- 4-CTA clusters at S4: 2 CTAs/SM impossible (2 x 46K regs > 64K), at 1 CTA/SM only 30 4-clusters co-reside (need 32)
  -> second wave. S3 with 3-CTA clusters (96 CTAs, chunk 768/768/512 by the 128-key half rounding): +50% keys on the
  critical CTA, S2's +1.7 us says that loses more than the ~1-1.5 us combine exposure. => v16 sl merge CLOSED.
- Same arithmetic for d32 sl (nds S8, 256 CTAs at 2/SM, combine 1.6 warm): an 8-CTA cluster per request needs 32
  clusters x 4 SMs; at 0.24 %/us the whole combine is worth <= 0.3%: not built.
- Re-read: d1 full judge profile 8.2 + 1.4 warm vs 12.19 cold; s32 all-L2 only -1.2 -> the rest of the cold penalty is
  the per-program page-table trip (pt 320 KB cold), no overlap available at kernel start (nothing earlier in the graph).
- Conclusion for the lead: every attention item in my STATE list is now measured dead or <= 0.3%. Batch-1 sliding
  (csl/vtk) structure was mined in ~/attn s72-s126; I have no new >= 1% lever. Asking for a lane (COORD Requests).

## Night4 session 35 (2026-10-27 12:48Z - ) -- attention lane: decode 1 full (_gd_full RED) timeline + H200 DIAGs
- No lead answer to s34's lane request yet; picked d1 full (largest attention gap left: H200 12.2-13.3 vs ~8 us floor).
- Tool: tools/s35/mktl_d1.py (LOCAL stamp copy, never submit: clock64/globaltimer per _gd_full program: start, q ready,
  per tile, loop end, partial store, red add end; _red_fin start/after wait/loads/end) + tools/s35/run_d1.py <dir> [reps]
  (graph replay, 512 MB cold read, OV='{plan json}' env = plan override).
- **H100 d1 full (511 one-warp programs, 500 of them 5 x 32-key tiles, NS 2), us:** q phase 1.1 | first tile 2.8 after
  q | then 1.17/tile | store 0.16 | red atomics 0.28 | program 8.8 | last program end 10.0 after first start | _red_fin
  released 0.5 later, done 1.0 after release | total 11.5. Matches ~/attn s92 (2.9 us fixed chain + gather rate).
- stages 3/4: first tile arrives LATER (q+t0 3.96 -> 4.9 -> 6.4 us) while later tiles take 0.86 us (= the per-tile
  compute chain of one warp, ~1550 cyc). On the H100 the whole thing is at the random-gather rate (more bytes in flight
  only reorder). Without qf, the q-row load queues behind the prologue gathers (q phase 1.1 -> 2.8 -> 4.7 at NS 2/3/4).
- Built (submission.py, default off / code-identical at NS 2): GD_PTF (prologue page-table loads all first, then the
  copies) and plan key `qf` (query rows loaded before the first copies). Local H100: ties (13.66 vs 13.76).
- **H200 DIAG 95f8158e** (tools/s35/mkab.py v1): main 13.28/13.20 | qf 12.75/13.09 (-0.3) | splits 640 (625 programs x 4
  tiles) 12.86 | qf+640 13.86 | qf stages 3 14.37/14.37, 4 14.62, stages 3 13.65 (deeper stages lose on the H200 too).
- **H200 DIAG 1259baad** (mkab v2, 3 round-robin repeats): main 13.68/13.70/13.62 | qf 13.12/13.57/13.57 | splits 640
  13.58/13.44/13.49 | qf+640 14.22/14.16/14.16 (the combination loses every time; not understood, not pursued).
  => qf: 5/5 H200 pairs over two checks, -0.05..-0.56, mean **-0.28 us** at d1 full. ON MAIN (plan qf=True for the
  batch-1 gdec plan); tests 16/16. Not submitted: ~0.2% of the attention score is far below whole-run noise; it rides
  along with the next submit. splits 640 -0.1..-0.4 (4 samples): too small/noisy, left off.
- Checks today (attention): 11/20 (s35: 95f8158e, 1259baad, both DIAG, never submit). Scored 2/10.
- Read for the next session: d1 full on the H200 = ~2.9 us fixed chain (q 1.1, first tile pt+gather trips) + 5 tiles x
  ~0.86 us per-warp compute chain + red 0.45 + _red_fin ~1.5 after the last program. Deeper stages, more/fewer splits,
  native kernels (~/attn s73/s92), fused merges (s36/s43/s50) all measured dead. Remaining idea (unbuilt): shorten the
  per-tile Gluon chain (0.86 us = ~1550 cyc for one warp; F2FP K/V upcasts + block-diagonal waste) -- ~/attn tried tqk/tpv
  (lost on column reductions); a 2-warp program splitting each tile's keys is the untested form.

## Night4 session 36 (2026-10-27 13:14Z - ) -- attention lane: record attempt on main (qf) + csl SASS audit
- Lead has not answered s34's lane request (COORD Requests). No new Human decision for moe since 09:45Z.
- csl (d1 sliding) SASS audit (decode<1,...> from the current build): all gathers (12 LDG per thread) are issued
  first, then the Q RMSNorm/RoPE shuffles and the new-token K/V branch (CTA 15 warp 3: 5 SHFL + RSQ + STG/STS) run
  under the gather latency; the first gather use is the QK MMA. So the post-gather 0.45 us is the MMA -> max/exp/sum
  shuffles -> PV chain only; nothing independent left to hoist. "Wider per-lane rows" (1 K + 1 V row per thread,
  smem re-layout to fragments) would double sector requests (16 B loads of 32 B sectors) for the same one-trip
  latency: not built. csl is closed for me.
- **Check 76983f6b** (main 7fe6b59 = board code + qf at d1 full): **2.5964x** (1450.6 us) vs board 2.5962x = tie; NOT
  submitted. Per shape (us, vs board 957cabd7): d1 sl 3.87 (4.47), d1 full 12.63 (12.19: qf's -0.28 lost in node noise),
  d16 8.00/87.41, d32 12.04/167.79, v1 5.41/16.02, v16 11.73/104.76, v32 14.02/194.70 (all within +-0.4 of the board run).
  Attention budget 10-27: checks 12/20, scored 2/10.
- Session result: no new attention lever; lane request to the lead still open. Next session: if still no answer,
  don't spend attention checks on ties; the only unbuilt items left are <= 0.1% layer each (d1 full 2-warp tile split).

## Night4 session 37 (2026-10-27 13:26Z - ~13:40Z) -- no lane answer; last unbuilt attention item closed on paper; 0 checks
- Still no lead answer to the s34 lane request (COORD Requests); newest Human decision for moe is 09:45Z.
- d1 full "2-warp program splitting each tile" (s35's only unbuilt item) closed on paper: a 2-warp program halves each
  warp's tiles exactly like splits 1023 (2.5 tiles/program) does, minus the extra atomics. splits 1023 and 640 already
  lose or tie on the H200 (s32 4fd8b5d5, s35 95f8158e/1259baad), and the H100 timeline runs at the gather rate (1.17 us/tile)
  not at the per-warp chain (0.86). So the SM's gather/issue throughput limits it, not the warp's latency chain. Not built.
- Re-read the layer STATE prefetch dead-ends (s6/s14/s55/s61: weight L2 prefetch during attention/csl/tk2f/norm, G10_PFL2,
  UPF*): every overlap form I could add from the moe side is already measured there. Nothing new to build.
- Session result: no open moe/attention item; attention budget 10-27 unchanged (12/20 checks, 2/10 scored).

## Night4 session 38 (2026-10-27 13:43Z - ~13:55Z) -- no lane answer; no build; 0 checks
- Lead still hasn't answered the s34 lane request; newest Human decision for moe is 09:45Z (lead 10:50Z/11:50Z are layer-only).
- Re-read layer STATE/NOTES s93-s94 for the biggest open gap: twm dq-bound at v16/v32 (~6.6% weighted; moe s27-s29: ALU +
  tensor do not overlap SM-wide, dq at the 8-op minimum: closed) and g24 pooler+unpool at d32 (~2.0 TB/s, ~2.5% weighted;
  proj s50-s62: consumer/latency-bound at 30% issue, BF16 no-decode rows cost the same, g23/cuBLASLt lose: proj-closed).
  Neither has a new design I can put forward: no build, so the shared GPU stays free for layer/proj.
- Attention budget 10-27 unchanged: checks 12/20, scored 2/10 (no tie checks).

## Night4 session 39 (2026-10-27 14:00Z - ~14:15Z) -- no lane answer; per-shape attention gap re-read; no build; 0 checks
- Newest Human decision for moe is still 09:45Z (lead 10:50Z/11:50Z are layer-only); s34 lane request unanswered.
- Board re-read: layer #1 pjmullin3 2.399x (we 2.134x); attention #1 pjmullin3 3.032x (we 2.596x, 1450.6 vs 1242.3 us).
- Lowest per-shape speedups in our last attention check (76983f6b): verify 16 full 1.96x (104.8 vs floor 63.4 us), verify 32
  full 2.00x (194.7 vs 126.8), decode 16/32 full 2.30x (87.4/167.8 vs 63.2/126.4). The verify-full extra over decode
  full (+17/+27 us) is the gw_verify consumer (issue-bound ~45%, ~/attn s12-s125: FP8 PV, poly exp2, f16 ex2, WS variants,
  native nvf all dead) and layer s84 closed full attention in-layer on the H200. Even verify full at decode-full speed is
  only ~-4% of the attention geomean, so #1's 16% lead is not in a single shape we know how to attack.
- No new design; left the shared GPU to layer/proj. Attention budget 10-27: checks 12/20, scored 2/10 (no tie checks).

## Night4 session 40 (2026-10-27 14:17Z - ~14:45Z) -- attention lane: H200 bounds of the batch-16/32 FULL kernels (1 DIAG check)
- Still no lane answer (newest moe Human decision 09:45Z). Layer closed full attention in-layer (s84) and moved to lane 3, so
  the full kernels' H200 bound was the one unmeasured number: ~/attn hdiag5's d32 floors were never run on the H200 (my s32
  ran only batch-1). Built tools/s40/mkdiag.py (DIAG copy, NEVER submit; from tools/s32): per scored full shape, after the
  first eager call: main, main with all-L2 page table (= consumer-only time), main with per-request consecutive rows (dense
  DRAM order), hfloor random-row gather floors R2/R4/R8 (all K+V 128-B rows, all in flight), dense floor. 40 cold replays.
- **Check d5bf959a (DIAG; timed shapes = main code): 2.5754x** (node noise; per shape within +-0.6 of 76983f6b except d1 sl).
  H200 us (median, cold L2, minus empty graph):
  | shape | main | all-L2 (consumer) | main seq rows | random floor R2/R4/R8 | dense floor |
  | d16 full | 86.4 | 62.1 | 83.3 | 79.3/79.4/80.0 | 72.5 |
  | d32 full | 168.3 | 119.9 | 158.8 | 159.4/159.3/159.3 | 144.1 |
  | v16 full | 101.8 | 92.7 | 100.8 | 80.5/80.6/80.7 | 73.4 |
  | v32 full | 187.9 | 169.7 | 184.2 | 157.2/157.1/157.5 | 142.5 |
  (H100 same tool: d16 117.6 / floor 111.0 / dense 103.5; v32 240.1 / 221.0 / 205.9.)
- Readings:
  * The H200 random 128-B row gather ceiling is ~4.1 TB/s (d32 ~655 MB in 159 us), ~10% under dense (4.55 TB/s); TASK.md's
    126 us floor (4.8 TB/s) is not reachable with page-1 random rows. Deeper in-flight (R8) does not help.
  * Decode full is memory-bound at 92-95% of that ceiling: headroom d32 <= 9 us, d16 <= 7 us (the overlap loss between a
    consumer of 120/62 us and the gather stream). Even with dense rows main would only reach 159/83.
  * Verify full is CONSUMER-bound: all-L2 time is 90% (v32 170) / 91% (v16 93) of main, and above the gather floor at v32.
    Headroom = consumer speed: v32 up to ~31 us, v16 ~21 us if the consumer fell under the floor (this is the known
    _gw_verify issue-bound consumer; ~/attn s12-s125 dead-ends: poly exp2, f16 ex2, FP8 PV, WS variants, native nvf).
  * Upper bound on the attention score from all four full kernels at perfect overlap: ~ -4% (d32 -9, d16 -7, v32 -31,
    v16 -21 us). The #1 team's 16% lead cannot come from the full kernels alone; per-shape unknown.
- Attention budget 10-27: checks 13/20, scored 2/10. Tool commit 14e41ff.

## Night4 session 41 (2026-10-27 14:30Z - ~15:45Z) -- lead 13:30Z item (2) prototype: T=128 GEMM with RESIDENT x slice (dev/s41); 0 checks
- Attention lane: nothing open (s36-s40). Per lead 13:30Z ("a session that finds nothing to build takes the next item below"),
  took item (2) (layer's T=128/256 cuBLAS GEMMs; layer s95-s97 idle on it). proj already killed the obvious forms: g22 (12-bit
  swap-AB wgmma, per-stage x box) and g23 (BF16 SS wgmma, tile split-K, cluster DSMEM sum) died on per-SM ingest (x/W 1.33).
- New structure "xr" (dev/s41/xr_src.cu + xr.py): CTA (g, s) = n-group of NG rows x k-slice KC=512; the x slice (128 tok x
  512 k = 128 KB, 8 SW128 boxes) is loaded ONCE and stays resident; W streams as 64x64 boxes through a 10-stage TMA ring; SS
  wgmma m64n64k16 swap-AB (A = W box, B = x box; WG w = tokens 64w..); after each 64-row tile the WG red.adds its FP32 tile
  into out[N][128] (pre-zeroed). Grid pool (8 x 16), unp (16 x 8) = 128 CTAs. Ingest per SM = W + 0.25 W (x), out = 0.5 W.
  Correct (rel err 6e-7 vs FP32 matmul). XR_LIN = W pre-packed tile-contiguous + 1-D bulk copies (same speed).
- H100, cold L2 (read-flush; NOTE harness overhead = 4.9 us empty-graph replay, subtract it), us incl. that overhead:
  | variant | pooler m128 (4096x8192) | unpool m128 (8192x4096) |
  | cuBLAS torch.mm BF16 | 29.5 | 29.3 |
  | xr red.v2 | 44.2 | 44.8 |  | xr red.v4 (lane-pair shfl) | 42.3 | 42.0 |
  | xr no flush (diag) | 27.4 | 27.7 |  | xr no MMA no flush (stream only) | 27.5 | 27.4 |
  Net of overhead: stream-only 22.5 us = 2.98 TB/s (= H100 bare HBM rate; MMA fully hidden), cuBLAS 24.6 (2.73 TB/s).
  The FP32 red traffic (S x N x M x 4 = 33.5 MB) costs +15..17 us (~2.2 TB/s of L2 atomics) and does NOT overlap the
  stream; v4 vs v2 = same -> byte-bound, not op-bound.
- Verdict: the resident-x stream is at the HBM rate (BF16 -2.1 us vs cuBLAS on H100; on the H200 cuBLAS is 2.2-3.2 TB/s vs
  4.06 stream, and 12-bit W would cut the stream 0.75x), but ONLY if the k-split reduction is nearly free. Not usable yet.
- Next (unbuilt, est. 1-2 sessions): cluster = all k-splits of one n-group (pool 16 non-portable, unp 8), reduce-scatter each
  64x128 FP32 tile through DSMEM (st.async of 1/C slices to owner CTAs + mbarrier complete_tx; 30 KB DSMEM per CTA per tile,
  ~31 MB total), owner writes BF16/FP32 out with plain stores (no zeroing, no reds). Gate: H100 xr+DSMEM <= ~24 us (beats
  cuBLAS 29.5 incl. overhead) -> then 12-bit W (proj DEC3 decode into RS A fragments: m128 has 64 clk MMA per 64x16 box vs
  ~100 clk stream, dequant cheap at N=128) -> layer TUNE option at T=128. T=256 (v32): x slice 256 KB does not fit; skip.

## Night4 session 42 (2026-10-27 14:50Z - ~15:45Z) -- xr (T=128 GEMM, lead 13:30Z item 2) cluster k-split reduce: dev/s42; 0 checks
- Goal from s41: replace xr's FP32 red.add out (+15..17 us) by a DSMEM reduce inside a cluster of the 8 k-splits (unp m128:
  N 8192, K 4096, KC 512 -> S = 8 = cluster). Tools: dev/s42/xr2.py (build/correct/determinism/time, SRC=xr3_src.cu for xr3),
  dev/s42/tl.py (per-tile globaltimer timeline of cluster 0; -DXR_TLON). Times incl. 4.9 us harness; H100 cuBLAS 28.5-28.7.
- **Correction to s41 (important):** (1) s41's xr kernel ran with ALL wgmmas serialized (ptxas C7520 "serialized due to
  WG.AR in divergent path": DEPBAR gsb0 0x0 after every HGMMA). Causes: empty-slot arrive under `if ((tid & 127) == 0)`
  and no wgmma.fence per stage. Fix = every consumer thread arrives (empty count 128/256) + wgmma.fence before each
  stage's MMAs (as twf does). (2) s41's "stream-only, MMA hidden 22.5 us" = XR_RED 0 diag in which the MMA results were
  dead: ptxas replaced the HGMMAs by dummy `HGMMA.64x8x16 RZ`. So it was a pure stream number. Always keep MMA results
  live in diags (I sum them into a never-true store) and grep SASS for HGMMA shape + DEPBAR 0x0.
- (3) SS m64n64 swap-AB with both WGs reading the same 8 KB W box per wgmma = ~128 B/clk of smem: smem-bound, ~2.3 us per
  64-row tile even unserialized. m64n128 per WG (WG k-split + combine) is fine for the MMA rate but needs an intra-CTA combine.
- Cluster facts (H100, 1 CTA/SM ~200+ KB smem): max active clusters C=8: 15 (120 CTAs), C=4: 30 (120), C=2: 66 (132). So a
  16-cluster grid runs in 2 waves (77 us); use G = 15 n-groups with uneven tile ranges (128 tiles -> 8 or 9 per CTA).
- xr2 (x resident in smem 128 KB, W SS-A, pull-reduce over ld.shared::cluster, mbarrier ready/free): every form 48-98 us.
  The chain ready-arrive -> DSMEM pull -> free-arrive was ~1-2 us per hop under load, and the smem left over (ring 4-8 stages,
  1-2 staging buffers) cannot hide it. Dead.
- **xr3 (dev/s42/xr3_src.cu): x K-slice in REGISTERS** as wgmma RS A fragments (M = tokens, WG w = tokens 64w.., 32 k16 x 4
  = 128 regs; setmaxnreg 208/40/56, 512 threads, no spills), W = smem B operand (64-row x 64-k SW128 TMA boxes, 16-stage
  ring = 128 KB), D[tok][n]: column group j = owner CTA j. Correct (rel err = cuBLAS's), bitwise deterministic.
  | xr3 variant (unp m128, G15) | us |
  | x via strided 4-B __ldg, pull reduce | 47.6-48.1 (x load alone ~6.5 us at kernel start) |
  | + fragment-major x (XR_XF: xf[s][wg][wi][c][lane] uint4; host-permuted here, a producing kernel could write it) | 43.9 |
  | + **PUSH** (XR_PUSH: st.async.v2 from registers into the owner's double-buffered recv, complete_tx; reducer reads locally) | **34.0** |
  | stream only (DIAG1, MMA live, XF) | 31.3 |  | NS 20 instead of 16 | 36.2 (worse) |
  | pool m128 (S 16 -> 2 clusters per n-group, FP32 red out, G7) | 36.6 + zero (cuBLAS 29.6) |
  Push costs only +2.7 us over stream-only (vs +12.5 for pull). The rest of the gap to cuBLAS is 120 of 132 SMs (cluster 8
  cap) x 9-vs-8.53 tiles imbalance: stream-only 26.4 net vs cuBLAS 23.7 net.
- Verdict: on the H100 xr3 push is cuBLAS + 5.4 us at unp m128. Bound if the tail imbalance were fixed (n40 tail tiles):
  ~27.9 = tie. A layer win needs the H200 (cuBLAS at 2.2-3.2 TB/s there vs ~4.0 stream x 120/132) and/or 12-bit W. Since W
  is the smem B operand in xr3, 12-bit needs a dequant warpgroup writing BF16 boxes to smem (or flip roles again: W as RS A
  dequantized in registers, x resident in smem 128 KB as B with m64n128 per WG on alternate tiles = 64 B/clk smem; fits
  x 128 + recv 64 + raw ring 24 KB). NOT handoff-ready; no moe-problem DIAG for it (lead 08:13Z: a problem's checks run
  only that problem's kernels): the H200 number has to come from a layer TUNE option once it at least ties on the H100.

## Night4 session 43 (2026-10-27 15:38Z - ~16:15Z) -- T=256 12-bit GEMM (layer g30 extended) + g30 T=128 bound diagnosis: dev/s43; 0 checks
- xr3 (s42, unp m128 34.0 vs cuBLAS 28.6) is dominated by layer s99's g30 (21.6 vs 25.7): xr3 CLOSED.
- Tools: dev/s43/g31.py (copy of r-layer dev/s99_g30.py; env SRC=<.cu> T=128|256 PAIR=0|1 DEFS=... VB=1 for ptxas -v), cub.py
  (cuBLAS T=128 vs 256). H100 cuBLAS: unp 8192x4096 25.8 -> 27.2 us at T=256 (632 TF/s), pool 4096x8192 26.2 -> 31.7,
  qkv 2304x8192 20.0 -> 21.6. H200 profile 940a4fca (layer v32 = T=256): unp ~26.9, pool 33.6, qkv 20.7 -> cuBLAS at T=256 is
  compute-bound on both GPUs (H200 = H100 compute).
- g31.cu (T=256, both WGs decode the same W stage, WG w = tokens 128w.., m64n128 each, x box 32 KB NS5): unp 34.7 us vs
  cuBLAS 27.6. No-load floor (NOX+NOW) 28.5: two m64n128 RS chains ~60% of peak. DEAD.
- g32.cu (T=256, m64n256k16 RS per WG, WGs on alternate k stages, end exchange through smem, 384 thr + setmaxnreg 56/224):
  correct; unp 31.7-32.6, pool S2 33.0-36.1 vs cuBLAS 27.7 / 33.9. Ablations (unp): no loads 25.5, NOMMA 31.6, W only 27.9,
  x only 28.8 -> per-CTA x ingest (2 MB at T=256 per 64 W rows, ~70 GB/s/SM) binds; CL=2 x multicast 33.9 (no help: multicast
  still lands the full box in every SM), CL=4 doesn't fit (30 clusters). T=256 12-bit GEMM DEAD on paper and measured.
  ptxas lesson: setmaxnreg is IGNORED (C7507 -> C7512 wgmma serialized) if dec/inc sit in an if/else that re-joins, or the
  producers `goto` a shared exit label; it works with dec inside the producer branch that `return`s and inc at consumer start.
- g33.cu (T=128 g30 R2 = 128 W rows per CTA, S=2 k-split pair as a (1,2) cluster, rank 1 st.async-pushes its FP32 acc into
  rank 0's smem, rank 0 adds + BF16 out; no slabs): correct; unp 28.1-28.9 vs no-reduce 24.7, slab form (PAIR=0) 23.4-24.2,
  cuBLAS 25.7. Push+pair wait = +3.5..4 us (dedicated recv buffer, no rdy wait: same). DEAD on the H100.
- **Diagnosis of g30 at T=128 on the H100 (g33 PAIR=0, unp S2):** W stream+decode only 20.9 us (50 MB, ~2.8 TB/s net of
  ~3 us fixed), MMA+decode only 16.4, x only 9.6. => g30 is HBM(W)-bound on the H100, not ingest- or L2-bound (fits layer's
  "multicast buys nothing"). On the H200 (HBM 1.45x) the W stream drops to ~12-13 us and per-CTA x ingest (1 MB per 64-row CTA
  ~ 15 us at ~65-70 GB/s/SM) may become the bound: then R2 (128 rows/CTA, 0.5 MB x) forms win there and not here. Only an
  H200 run can rank g30 forms.

## Night4 session 44 (2026-10-27 16:13Z - 16:20Z) -- attention lane: verify 1 full softmax-reference probes (all dead); 0 checks
- Per-load speedups of board run 957cabd7 (vs SGLang H200): d1 3.66x, v1 3.20x, d16 2.55x, d32 2.33x, v16 2.11x, v32 2.09x
  (full shapes 1.97-2.58x, sliding 2.47-4.69x). #1 (3.03x) needs ~+17% geomean: not reachable from the full kernels alone
  (s40 bounds), so the #1 team must be far ahead on batch-1 and/or sliding shapes too.
- **Constant softmax reference (FREF) DEAD by arithmetic**: qk_scale_factor = 5.473, so scores reach ~63 * rms(k) log2 units
  and row maxima spread over far more than FP16's ~30 usable binades; built it (reference 12, underflow guard -> REDO) and
  every split REDOed (6.9 ms). Any fixed reference must come from the data (as FIX 1 = tile 0's max + 14 does).
- **FIXL** (FIX 1 reference taken inside the pipelined loop's first tile, `if start == lo`, no pre-pass; bitwise = FIX 1):
  H100 v1 full +0.35 us (6/6 pairs: 17.1-17.3 vs 16.8-16.9); with pre 1 +0.6. The runtime branch in the pipelined loop costs
  more than the pre-pass's 1.1 us timeline slot (which overlaps the PRE-prefetched gathers).
- fix 0 (online max per tile) vs fix 1: +1.2 us H100 (18.05-18.21 vs 16.83-16.93): the max/alpha chain costs ~0.12 us/tile.
- **L2D** (FIXED loop keeps [64, 64] FP32 running sums, one reduction after the loop instead of tl.sum(p, 1) per tile):
  +1.7 us (6/6; 32 more registers per thread). Dead.
- No code change on main. Attention budget 10-27: checks 13/20, scored 2/10 (none spent).

## Night4 session 45 (2026-10-27 16:21Z - ~16:45Z) -- attention lane: re-audit of remaining headroom vs ~/attn history; no build; 0 checks
- Board 957cabd7 per shape (H200 us, ours / SGLang): d1 4.47/20.9 sl, 12.19/31.5 full; d16 7.65/27.1, 87.4/200.7; d32 12.2/31.4,
  168.2/383.7; v1 5.63/23.1, 15.98/36.7; v16 11.5/29.3, 104.0/205.2; v32 13.9/34.3, 194.2/389.5. Score = geomean of the 6
  loads (46 sl + 16 full); load speedups d1 3.66, v1 3.20, d16 2.55, d32 2.33, v16 2.11, v32 2.09.
- Largest relative headroom per load (vs measured floors): v32/v16 full (consumer-bound, -31/-21 us max, ~5% score),
  d1 full (12.2 vs ~8.0 random-gather floor, ~2.5% score), v1 full (16.0 vs ~7.9, ~3.5%). Each one's candidate levers are in
  ~/attn's dead list: FP8 P@V (s22 ablation: even a free transpose only -4/-10 us all-L2), poly exp2 (s12: issue-bound, not
  MUFU), f16 ex2, deeper rings / HOIST (d1 full s88-ish: stages 3/4/6 +0.8..2.9), native d1 full cfl (s73: 19 vs 15.8),
  in-grid merges (s95/s113), FPIPE/ping-pong (s101/s113), my s33/s35/s44 v1/d1 probes. Batch-1 sliding: csl/vtk mined
  (s72-s126 + my s31/s36). I found no lever with a mechanism not already measured dead: no build, GPU left to layer/proj.
- Attention budget 10-27 unchanged: checks 13/20, scored 2/10. Lane request posted to the lead (COORD Requests).

## Night4 session 46 (2026-10-27 16:40Z - ~17:00Z) -- no lane answer; last unbuilt attention item bounded dead from data; 0 checks
- COORD: no lead entry since 13:30Z (LANE DECISION 4); moe s45 lane request open; layer s102/s103 and proj also report empty queues.
- d1 full "2-warp program splitting each tile" (the last unbuilt item, s35): DEAD on existing data, not built. s33's per-SM
  saturation (1/2/4 one-warp programs per SM: 0.66/0.92/1.67 us per tile = SM throughput 0.66/0.46/0.42 us per tile) shows
  _gd_full/_attn_split at ~4 programs/SM (511 programs on 132 SMs) is already issue-throughput-bound, not chain-latency-bound:
  splitting a tile over 2 warps adds warps to a saturated SM and leaves total instructions unchanged (consistent with splits
  640/1023 measured dead on the H200). The only lever there is fewer instructions per tile (the Gluon F2FP upcast + block-diagonal
  waste), which ~/attn tqk/tpv measured dead.
- Re-read the latest layer H200 profile (940a4fca d1): tk2f 27.8 sl / 21.0 full, pr1 18, g13 qkv 11, boundary 2x3.1, n2u 5.3,
  csl 4.0 us. Every item is owned + closed (tk2f layer s81, pr1 proj, boundary/glue layer s78).
- Twm consumer bound re-derived (v16: ~900M weights x 1.5 dq instr = ~40 us of full-rate issue on the H200 vs 43 us stream):
  it needs < ~1 instr/weight AND > 50% issue efficiency; INT8/FP8 tensor paths fail on the 16-wide NVFP4 block scale (k32 MMAs
  span two scale blocks) and BF16 x exactness. Nothing new to build.
- Attention budget 10-27: checks 13/20, scored 2/10 (unused; main = checked tie 76983f6b, no resubmit).

## Night4 session 47 (2026-10-27 16:59Z - ~17:35Z) -- LANE DECISION 5: g30 structure at m16/m32 ("g34") built; H100 beats g24 at m32; 0 checks
- Lead 16:45Z: build g30 (12-bit W decoded into wgmma A registers, swap-AB) at m32/m16 for the d32/d16 pooler + unpool.
- muse_moe_w4a16/tools/s47/g34.cu (+ helpers.inc = g30_src.cu's helpers): CTA = 64 W rows x k-stage range [k0,k1) (uneven split
  S = gridDim.y), x TMA box {64 k, MT rows} SW128, wgmma m64n{32,16}k16 RS, NWG consumer WGs on stages i = wg mod NWG + W / x
  producer warps (g30 roles), smem WG reduce, out FP32 [MT][N]: S1 plain stores, S2 atomicAdd into a zeroed out (0+a+b is
  order-independent: deterministic; S3 is not). Escapes = g30's A-half patch with candidates filtered to the CTA's k range and
  its WG ((stage-k0) % NWG). SAME PACK as g30 (layer._pack30 + _compress hdr): no new prepare code.
- Correct: rel err 1.66e-3 (= cuBLAS's) on the layer's real pooler/unpooler; tools/s47/g34exact.py (one-hot x at all 3607 / 2212
  escape columns + random): 0 mismatches, S1 + S2, m16 + m32.
- H100, us = graph(3 us PDL pred + kernel) - graph(pred), cold L2, median of 3-5 reps (g34.py; g24 via r-proj dev/s61/g24.py
  with the layer's T=16/32 cfgs, same method). NOW = no W loads (consumer-only = the H200 proxy, g24 is consumer-bound there):
  | shape | g24 layer cfg full / NOW | g34 NWG2 NS/NW10 full / NOW | **g34 NWG4 NS/NW20 full / NOW** | cuBLAS |
  | pool m32 (S2) | 21.0 / 16.2 | 19.1 / 14.6 | **18.9 / 12.2** | 26-27 |
  | unp m32 (S1)  | 20.5 / 15.7 | 19.1 / 14.2 | **18.9 / 12.3** | 25 |
  | pool m16 (S2) | 18.5 / 10.9 | 19.0 / 14.0 | 18.4 / 11.3 (NW16: 19.0 / 11.1) | 26 |
  | unp m16 (S1)  | 19.0 / 10.8 | 18.9 / 13.0 | 18.1 / 11.2 (NW16: 18.6 / 10.9) | 24.5 |
  => m32: -2.1 / -1.6 us full, -4.0 / -3.4 us consumer-only per GEMM (d32 has both: est. H200 -4..7 us per d32 layer call).
  m16: tie (full -0.1..-0.9, NOW +0.1..0.4): not worth porting at T=16 unless the in-layer A/B says otherwise.
- Dead knobs: NWG2 deeper rings (NW16/24: 2 CTAs/SM lost, unp S2 = 2 waves, +2..5 us); S3 (+1.5..2, nondeterministic);
  pool S1 (64 CTAs: 27 us); NWG6 (m16 23.6 full / 13.8 NOW; m32 does not fit smem).
- Integration (for layer/proj): `g34_gemm(x[MT,K] bf16, wp=_pack30, hdr, out fp32 [MT,N], scale, S, pdl)`; pooler S2 needs a
  zeroed out (the g24 EZ/ZN chain already provides one: o's ZN zeroes the pooler's out), unpool S1 needs none. Build flags
  -DMT=32 (one build per MT). PDL like g30: W streams before griddepcontrol.wait, x + out after.

## Night4 session 48 (2026-10-27 17:26Z - ~17:45Z) -- g34 superseded by proj g30n; closed; 0 checks
- Before starting the in-layer d32 A/B I found that proj had built the same g30-at-T=16/32 kernel in parallel (proj s95 16:50-17:40Z,
  r-layer branch proj/g30n 453790c, NOTES-proj s95/s96): G30_MTOK 32/16 + K split over a (1,S,1) cluster with a DSMEM push (mine: S2
  atomicAdd into a zeroed FP32 out). Same measuring method (3 us PDL pred, cold L2, H100):
  | shape | proj g30n | g34 (s47) | g24 |
  | unp m32 | 17.3-17.4 | 18.9 | 20.3-20.5 |
  | pool m32 | 18.8-18.9 | 18.9 | 21.0-22.1 |
  | unp m16 | 17.1-17.2 | 18.1 | 18.7-19.0 |
  | pool m16 | 18.4 | 18.4 | 18.5-18.75 |
  proj's version is faster or tied on every shape and writes BF16 like cuBLAS. proj s96 in-layer: g30n:unpool at T=32 is a TUNE option
  (-1.2..-2.1 us d32 with the TUNE off, the tuner picks it at sl -1.0..-1.4 and rejects it at full); the pooler loses +9..14 us at sl
  because it competes with the router for SMs. g34's pooler also fills the GPU (128 CTAs), so it would hit the same problem.
  => g34 CLOSED, not handed off. The tools/s47 files stay for reference. 0 checks, no GPU time used.

## Night4 session 49 (2026-10-27 17:42Z - 17:47Z) -- wrap-up only (sandbox moves ~18:00Z, STOP ~17:55Z); 0 checks
- Lane item (g30 at m16/m32) closed in s48 (proj g30n is the faster version and is on layer main v137 as TUNE[32] g30n:unpool).
  Too little time before the move to start a build; wrote the s50 pick list into STATE (g30n pooler with fewer CTAs, or the
  attention diag timeline). No GPU time used.

## Night5 session 1 (2026-10-27 18:57Z - ) -- pr32: T=16/32 pooler (g30n S2) + router (hi/lo) + routing in ONE launch; 0 checks
- New sandbox okc-night5 (runs to 11-05). Re-posted the 10-27 moe/attention judge runs in COORD (from these NOTES).
- Why: proj s96 found the g30n pooler (S2, 128 CTAs) loses +9..14 us in-layer at T=16/32 only because the side-stream
  router (_router_split + _routing) then has no SMs. pr8 (proj) already fuses pooler+router+routing at T=8; nothing does at T=16/32.
- Built tools/n5s1/pr32_src.cu (= g30_src.cu helpers + new kernel), harness tools/n5s1/pr32.py (exactness + cold-L2 time over a
  3 us PDL pred, proj s95 method; `SRC=... NW= NS= DEFS="PR_TL 1"` prints a per-CTA globaltimer timeline).
  Structure: grid (66, 2) in (1,2,1) clusters = 132 CTAs. x = 0, 1: 4 "finisher" CTAs (the 4 SMs 128 streaming CTAs leave).
  x >= 2: pooler group g = x - 2, K half y (g30n S2, DSMEM sum), and FIRST 8 router stages: rows 64 (g / 8).. of [hi; lo]
  (raw BF16, prepare packs them in A-fragment order [rg][kstage][4 j][128 thr][16 B], 8 KB stages through the same W ring and
  x ring). Router acc -> smem (both WGs, tok-major), bar.arrive; a NEW warp 10 sums them, rank 1 pushes to rank 0 by DSMEM,
  rank 0 writes the pair partial part[rg][g & 7][T][64] + fence + red.release count. Finisher f waits count == 64, pulls its
  8 tokens' chunks of all 64 partials in ONE bulk shot (64 x 2 KB, one mbarrier), sums in fixed order (deterministic),
  routes 1 token per warp: top-8 via redux.sync max (order-preserving u32 keys, +0.0f folds -0) + redux min (lowest index
  on ties) = same picks as the layer's _routing. Writes ids/lids/wts (+ FP32 logits [T][512] hi|lo for checking).
- Correct: routing ids/lids == layer Triton _routing on our logits for 4 seeds x 3 reps (wts within 3e-8: sum order),
  == FP64-logit routing, counters self-reset, pooled = g30 math (BF16 rounding-level vs FP32 ref).
- H100 standalone (us over a 3 us PDL pred, cold L2, M=32): g30n pooler alone (my kernel, router off) 20.0; + router stages
  (no reduce/routing) 21.1; first fused version (reduce by last-ticket CTA + routing in that CTA) 70 (!); final 22.9-23.2.
  LESSON (globaltimer timelines): while 128 CTAs saturate HBM every global round trip costs 4-7 us (Little: ~12 MB of TMA in
  flight / 3.3 TB/s), a non-streaming SM ingests only its fair share (~26 GB/s), and DSMEM scalar stores / the shuffle-based
  top-8 were slow (5 us per token!). Fixes that mattered: no reduce/route work in streaming CTAs (finishers on the spare SMs),
  pair DSMEM pre-reduce (halves partial bytes), one bulk shot per finisher (128 KB), token split (no logits exchange),
  redux.sync top-8, shallower W ring (NW 8: the fence after the partial store 4.6 vs 11-16 us at NW 12-16; pooler also faster).
- Clusters: cudaOccupancyMaxActiveClusters on the H100 with ~200 KB smem: size 2 -> 66 (132 CTAs), 4 -> 30, 8 -> 15, 16 -> 7:
  only pairs fit all 128 streaming CTAs, so in-cluster reduces beyond pairs are out.
- In-layer H100 (copy /tmp/m5l of r-layer main v137 + tools/n5s1/patch_layer_pr32.py on submission.py, TUNE {} so TUNE_BASE
  only, tools/ab_flag.py 3 rounds, GPU shared with the other agents), us off -> pr32:
  d32 sliding 196.75 -> 195.44, full 422.74 -> 421.47; d16 sliding 190.78 -> 191.87, full 300.21 -> 298.77.
  = tie..-1.4 us: the fused router removes the whole +9..14 us router-contention loss of the g30n pooler (proj s96), but the
  H100 is HBM-bound here (pr32 = 58.7 MB at ~2.5 TB/s, like g24 + side router), so it cannot show more. H200 tuner = the gate
  (on the H200 g24 pooler+unpool run at ~2.0 TB/s in-layer at d32).
- Tests with pr32 forced on (submission_on.py): bench test 13/13 PASS; route_judge 30 inputs x seeds 11,7 x d16/d32 sl/full
  0/240 fails (worst gap 0.0187 vs FP8-ref 0.0187); replay_test 20 x d16/d32 0/80.
- Capture gotcha: the pr32 branch must still fork the side stream (side_stream.wait_stream(main)) because later code joins it.
- Handoff posted (COORD): TUNE option {"pr32": True} at T=16/32, patch tools/n5s1/patch_layer_pr32.py (test form reads the .cu
  from r-moe at prepare: the integrator must embed it like _G30_SRC and put the build on _BUILD_POOL).

## Night5 session 2 (2026-10-27 20:02Z - 21:00Z) -- pr32 H200 proxy (L2-resident weights); 0 checks
- 20:02-20:40Z GPU lock blocked twice by layer's /tmp/tl_opts.py (held the flock, then bench.time_graph re-locked on a new fd =
  self-deadlock). Told layer (COORD + SendMessage); layer fixed it. Check /proc/locks when a timing run hangs.
- New compile flag PR_L2W (body3.cu): every CTA streams pooler group (g & 3) and router group 0 = ~3 MB of weights, L2-resident
  -> consumer + chain floor with HBM out of the picture (the H200 has ~1.35x the H100's HBM bandwidth).
  H100 us over a 3 us PDL pred (NW 8, NS 16): M32 normal 23.0 / L2W 17.9; M16 normal 22.2 / L2W 16.9.
  Timeline (PR_TL, M32, us from first pstart): L2W: pdone med 12.0 max 12.8, rpart (router pair partials stored) max 5.1,
  fpoll 5.4, fingest 7.7, froute 11.0 -> routing finishes BEFORE the pooler: the finisher chain is hidden.
  Normal (HBM-bound): pdone med 18.4 max 20.0, froute 17.9.
  => pooler streaming floor 12.8 us for 58.7 MB (~4.6 TB/s equivalent) vs H200 HBM ~13.7 us at 4.3 TB/s: on the H200 pr32
  should run ~HBM rate (est. 15-17 us incl. launch/tail) vs g24 pooler ~25 us/call in-layer at d32 (+ router on the side
  stream). Predicts a real H200 win at d16/d32 if the in-layer router/routing removal holds. H200 tuner decides.
- Layer FYI (s?, v140 783caa6): torch.cuda.Stream() pool streams can alias the harness capture stream -> side branch captured
  in line (~10 us outliers in in-layer A/Bs on older r-layer copies, every 32nd tuner candidate). n5s1's in-layer A/B used v137.
- Ring sweep under PR_L2W (M32, us): NW/NS 8/16 17.9, 16/8 17.5, 12/12 17.8, 10/16 18.5, 12/16 18.7; 6/16 and 8/8 invalid
  (the finisher's 64 x 2 KB ingest needs 128 KB of the ring smem: "pr32: finisher smem"). Spread <= 0.5 us: keep NW 8/NS 16
  (it won the HBM-bound H100 runs through the shorter router-fence wait); no second build for the tuner.
- zsh gotcha: `set -- $c` does not word-split in zsh (use NW=${c%:*} NS=${c#*:} with "a:b" pairs).
- Next: the H200 'layer tune T=16/32' pr32 lines from layer's 00:00Z bundle (layer promised to paste them). If pr32 wins there,
  try MTOK 16 S1 (more CTAs per K) and the router stage order; if it loses vs the 4.6 TB/s-equivalent floor measured here, the
  loss is HBM-side contention (H200 DIAG timeline via layer).

## Night5 session 3 (2026-10-27 21:00Z - ) -- pr32 PR_SA (st.async pair handoffs); 0 checks
- Budget at start: moe checks 0/20 left (10-27), attention 7/20 left (not used: no attention item; GUEST-style use forbidden).
- pr32 is already in layer main v141 (ed9b994, embedded by proj as a TUNE option) for the 00:00Z check: H200 verdict pending.
- Timeline stamps added (PR_TL): entry (8), xwait (9, consumers past griddepcontrol.wait), end (10, rank 0 after the out store),
  fxchg (3, finisher after the 64-partial sum); pr32.py prints the pair tail = end(rank 0) - max(pdone r0, r1).
  H100 L2W (H200 proxy), us from the first CTA entry, M32: xwait 2.5 (cluster barrier + escape-header sort, hidden under PDL),
  router stages done (pstart) 4.1, warp 10 rank 1 sum + DSMEM push -> rstore 6.4, pair partial store + fence -> rpart 8.7,
  fpoll 9.7, ingest 128 KB -> fingest 12.0 (one SM at ~55 GB/s under the stream), 64-partial sum 1.0, route_token 2.3,
  froute 15-16; pooler pdone max 16.9, pair tail med 1.5 (max 2.0) -> end max 18.9. M16 similar (froute 12.3, pdone max 15.1).
  => on an H200 the routing chain is about as long as the pooler stream: it was close to critical.
- PR_SA 1 (default now): every pair handoff by `st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32` into rank 0's
  smem; rank 0's tid 0 does arrive.expect_tx(bytes) on rbar/pbar at init (complete_tx may land first: fine). Pooler: rank 1's
  WG0/WG1 and rank 0's WG1 push their accumulators into 3 rbuf slots and exit (no bar.sync 1,256, no smem hop, no
  fence.acq_rel.cluster); rank 0's WG0 waits rbar and sums. Router pair: rank 1's warp 10 st.async into rpair (no fence).
  Sum order changes only by FP32 association (pooled maxerr identical to PR_SA 0 on all seeds); deterministic.
- Timeline SA1 vs SA0 (L2W): pair tail med 1.0-1.3 -> 0.5-1.0; rstore -0.8, rpart -0.8, froute -1.0..1.3 us (M16/M32).
- Timing (us over a 3 us PDL pred, 2 rounds x 3): L2W M16 16.7/16.85 -> 16.0/16.4, M32 17.8/18.0 -> 17.2/17.35;
  HBM-bound M16 21.9/22.1 -> 21.8/21.6, M32 22.65/22.9 -> 22.6/22.4. Exactness: pr32.py RS=25 x 3 reps M16/M32: BAD 0.
- r-moe 11d49b7. Layer diff: tools/n5s3/pr32_sa.diff (applies to r-layer muse_decode_layer/pr32_src.cu), or simply copy
  tools/n5s1/pr32_src.cu over it (superset: + PR_L2W/PR_TL debug paths, off by default).
- DEAD (n5s3): PR_RT route_token with per-lane pre-sorted 64-bit (key, slot) heads (19-comparator network; pick = 2 redux +
  shift; score re-derived from lg off the pick chain): exact, but finisher route step 2.8-3.1 vs 2.3-2.6 us (L2W timeline).
- In-layer validate (copy of r-layer v141 + this pr32_src.cu, PR32_DEFAULT forced True; validate --quick --skip=prep,tk2f):
  test 0/13, scored 0/12, opt 0/354, replay 0/240, route 0/240 (worst gap 0.0256), fuzz 0/362: ALL OK. Handoff posted in COORD.
- Next session (after 00:00Z): read the layer's H200 'layer tune T=16/32' pr32 lines (v141 check). If pr32 wins: make sure SA is
  in, then try MTOK16 variants; if it loses, ask layer for a PR_TL timeline DIAG. moe checks reset at 00:00Z (20).

## Night5 session 4 (2026-10-27 21:32Z - 21:45Z) -- pr32 PR_HL (router hi+lo fold in-CTA): tie, kept off; 0 checks
- Idea: the finisher chain ingests 64 x 2 KB per finisher and sums [512] hi|lo partials. PR_HL 1 packs each 64-row router
  group as [hi 32 rows | the same 32 lo rows] (host: cat([hi.view(8,32,K), lo.view(8,32,K)], 1)), so warp 10 folds
  r + (r+32) (and WG0 + WG1) before the DSMEM push / partial store: partials [rg][p][T][32], finisher ingest 64 x 1 KB,
  sum over 256 full logits, route_token uses lg[e] directly; logits scratch gets only the first 256 columns.
  FP32 association change only. Exact: pr32.py RS=4 x 3 reps M16/M32 BAD 0, routing == triton on our logits, == FP64 route.
- H100 L2W timeline M32 (us from first entry): HL0 -> HL1: rstore 5.1 -> 4.6, rpart 7.2 -> 6.4, fpoll 7.9 -> 6.9,
  fingest 10.2 -> 9.2, fxchg 11.3 -> 10.0, froute 13.6 -> 12.3; pdone max 15.9 / 16.1, end max 16.9 both.
  => routing chain -1.3 us, but after PR_SA it already ended ~2 us before the pooler: not on the critical path.
- Timing (us over 3 us PDL pred, 3 reps): M32 HBM 22.30-22.37 vs 22.16-22.48; L2W 17.12-17.20 vs 16.93-17.09;
  M16 HBM 21.60-21.89 vs 21.62-21.92; L2W 16.16-16.22 vs 16.35-16.38. = tie. PR_HL stays 0 (would also need the layer's
  _pr32_prepare packing change). Kept in body3.cu/pr32_src.cu as an option (pr32.py packs by DEFS).
- Conclusion: pr32 end-to-end = the pooler stream (+ ~1 us pair tail); chain levers are exhausted until the H200 says otherwise.
- BOUND (pr32.py WARM / WPAT / WR knobs: a warm kernel __ldcg-reads part of every CTA's stream after the cold flush, inside
  the timed graph, before the 3 us pred; time = minus warm+pred alone). Question: with pr32 HBM-bound on the H100 (22.3 vs
  L2-resident 17.1), does an L2 prefetch of the pooler W during sliding attention pay now (layer s61 killed it for consumer-bound g19)?
  M32 H100 us: WARM (first f of each CTA's 64 pooler stages + all 8 MB router W) 0 22.4 | 0.25 21.2 | 0.5 (33.6 MB) 20.3 |
  0.75 21.4 | 1.0 22.0 (L2 thrash > ~35 MB). Interleaved WPAT a/b (stages i % b < a): 1/2 (33.6 MB) 20.5 | 2/3 (42 MB) 20.5 |
  1/3 21.0 | 1/4 21.2. => best -2 us on the H100 under ideal conditions (warm fully landed, L2 otherwise empty); the H200 is
  near pr32's consumer floor (proxy 17.1 vs est. HBM ~18.5), so <= ~1 us/layer there (~0.14% at d16+d32 sliding) minus attention
  contention (O_PF lessons: prefetch at nds start is zero-sum). Not built; revisit only if the H200 tuner picks pr32 AND a
  layer DIAG shows pr32 HBM-bound there.
- Next: after 00:00Z read layer's v141 check 'layer tune T=16/32' pr32 lines (and make sure layer took PR_SA, 11d49b7).

## Night5 session 5 (2026-10-27 21:46Z - 22:05Z) -- paper bounds only; 0 checks, no GPU time
- Read COORD (no new lead decision since 20:00Z, nothing addressed to moe). pr32 (with PR_SA) is in layer main v143 as a TUNE
  option; the H200 verdict comes from layer's 00:00Z check 1 (v143).
- DEAD on paper: L2 prefetch of the pooler W issued by pr32 itself before griddepcontrol.wait. In-layer pr32 is a PDL secondary
  of _boundary (triggers at its start, ~3-4 us at T=32), so the pre-wait window is the boundary only, and the W ring (8 stages x
  8 KB x 128 CTAs = 8 MB) already takes ~2.4 us of H100 HBM (~1.7 us H200) of that window; a prefetch issued after the window
  just reorders the same HBM traffic. The only non-zero-sum form is a prefetch during attention/o (layer code; n5s4 bound <= ~1 us
  H200 = ~0.14%): not worth a request.
- MTOK 16 S1 at T=16: 64 groups x 1 CTA = 64 CTAs (wgmma M = 64 W rows fixed): does not fill the GPU: dead on paper.
- Reading the H200 verdict (next session): `grep -h "layer tune T=\(16\|32\)" ~/r-layer/muse_decode_layer/profiles/<job>.log |
  tr '|' '\n' | grep -E "layer tune|default|pr32|->"` (profiles are not in git: `okc status <16-hex job> --json` from that dir
  restores them; ask layer if absent). pr32 wins if its line beats default by > ~1 us at T=16 or T=32 (sl or full).
- Lane request posted to lead (COORD Requests): moe's items are all at measured bounds pending that verdict.

## Night5 session 6 (2026-10-27 22:04Z - 22:20Z) -- re-derived a dead end (scale compression); 0 checks, no GPU timing
- No new lead decision since 20:00Z; moe checks 0/20 until 00:00Z; pr32's H200 verdict comes with layer's 00:00Z check (v143).
- Looked for a byte-count lever: random_weights' e2m1 codes are uniform (4.0 bits, no gain); e4m3 block scales have entropy
  3.71 bits, codes 92..126 over 16 experts; per-row (256-scale) window of 32 codes fits 100% of rows (5-bit + row base =
  -4.2% expert bytes), 16-code windows fit only 99.6% of 4-scale groups (escapes needed). Then found n20's S6 (6-bit, -2.8%
  bytes) already measured on the H200: DEAD (decode consumer latency-bound, n22). A 5-bit form needs more decode ops
  (5-bit fields or nibble + bitplane), so it is dead for the same reason. Added to STATE's Dead ends so it is not rebuilt.
- Next: after 00:00Z read layer's v143 'layer tune T=16/32' pr32 lines (STATE n5s5 grep recipe); lane request still open.

## Night5 session 7 (2026-10-27 22:22Z - 22:35Z) -- re-audit only; 0 checks, no GPU time
- No lead decision since 20:00Z; lane request (n5s5) unanswered; moe checks 0/20 until 00:00Z; pr32 H200 verdict = layer's 00:00Z v143 check.
- Re-read the board-run profile 35dd53d0 for the batch-1 loads: d1 sl tk2f 20.2 us (0..3 local experts; = the 14.1 + 6.97/expert
  fit, the "fixed" part is mostly the PDL wait on pr1/rmsnorm), v1 sl tk2f 55.6 us (= fit at ~6 experts). tk2f is layer's (LANE 1).
  Early weight streaming during pr1/pr8 = RFLAG s27 (zero-sum, HBM busy). v16/v32: twm/twq/HOTQ/single pass all built + measured
  (s17-s30, n4s23). No unmeasured moe item found; did not rebuild a dead end.
- Next: after 00:00Z read v143's 'layer tune T=16/32' pr32 lines (STATE n5s5 recipe) and act on them; else wait for a lane.

## Night5 session 8 (2026-10-27 22:38Z - 23:25Z) -- pr32 at T=8 with an FP32 router (PR_F32) to replace pr8; 0 checks
- Why: layer n5s6 measured T=8 g30k pooler + side router DEAD and wrote "a g30 pooler would need the router fused like pr32
  (FP32 router at T=8)". SGLang at T <= 8 = FP32 F.linear on the FP32 router input (baseline.py ROUTER_FP32_MAX_TOKENS 8):
  a hi/lo BF16 split is NOT allowed there (x itself is FP32).
- Built (tools/n5s8: head.cu + body3.cu -> pr32_src.cu; pr32.py; patch_layer_pr8.py): G30_MTOK 8 (wgmma m64n8k16, epilogue
  store predicate) + PR_F32: router W FP32 [4 G][256 KS of 32 k][64 rows][32 k] (8 KB stages, same 8 MB as hi/lo BF16, same
  8 stages per CTA, 4 groups x 32 K slices = 128 CTAs), x = FP32 h32 by a 2nd TMA map (8 tok x 32 k, SWIZZLE_128B),
  consumer FMA (thread = rows rp, rp+32 x tokens 2tp, 2tp+1, explicit ld.shared), finisher 4 G x 16 pair slices (same 64
  partials), 256 logits (no hi+lo). T=16/32 builds unchanged (pr32.py M16/M32 BAD 0).
- Exact: pr32.py M=8 DEFS="PR_F32 1" RS=3: routing == layer _routing on our logits, == FP64 routing, logit max err
  1.8e-6 / 5.3e-6 / 9.2e-6 vs torch FP32 F.linear's 3.3e-6 / 6.8e-6 / 1.2e-5 (same seeds): FP32-class.
- H100 standalone (after a 3 us PDL pred, same method as proj s80's pr8 24.4 us): hi/lo router (not allowed, reference
  only) 20.5 us; PR_F32 21.0 us; L2W proxy (weights L2-resident ~ H200 consumer floor): hi/lo 16.0, PR_F32 17.25.
  Fixes on the way: generic LD.E.128 -> explicit ld.shared (18.4 -> 17.6 L2W); per-lane k rotation broke LDS broadcast
  (4 wavefronts) -> swizzled W pack + TMA SW128 x (17.6 -> 17.25). PR_IL (router stages interleaved with the first pooler
  stages, P R P R per WG) is WORSE: 18.25 (routing chain then ends ~3 us later: froute 16.6 vs 13.8 us).
  Remaining F32 cost: router phase xwait->pstart 2.3 us vs 1.28 for hi/lo wgmma (~0.25 us per FMA stage, 4 per WG).
- H200 question: pr8 there = 21.2 us GPU incl. ~4 us bnd wait (proj s80) ~ 17 us after bnd; pr32-F32 L2W proxy 17.25. Tie
  to small win on paper; the H200 tuner decides -> TUNE[8] options {pr32}, {pr32, cslpw}.
- Layer patch: tools/n5s8/patch_layer_pr8.py <copy>/layer.py (v144 4b30a76) + copy tools/n5s8/pr32_src.cu over its pr32_src.cu
  (T=16/32 calls get xr = h, unused). Path: inside the pr1 branch at T=8 (pr1 semantics kept: one main-stream kernel does
  pooler + routing), pooler pack = the g30n ("pooler", 16) entry's (wp, hdr, scale) (MTOK-independent), FP32 router packed
  in _pr32_prepare. PR8_32 = module default (False).
- bench test with PR8_32 = True (sub_f.py): 13/13 PASS.
- IN-LAYER H100 A/B (/tmp copy of r-layer v144 + patch, TUNE[8] pr32 entries stripped so only PR8_32 decides; tools/ab_flag.py
  SEEDS 99,7): DEAD. Run 1 (4 rounds) pr8 | pr32: v1 sl s99 142.74 | 145.01, s7 115.10 | 116.95; full s99 154.93 | 158.91,
  s7 126.39 | 130.66. Run 2 (3 rounds) pr8 | pr32 PDL | pr32 no PDL: sl s99 144.29 | 146.13 | 147.79, s7 115.95 | 117.84 |
  119.47; full s99 155.71 | 159.90 | 160.69, s7 129.20 | 133.02 | 134.16. => +1.8..4.3 us, no-PDL worse still.
- Timeline (tools/timeline.py "verify 1 full", s99): both launch at 44.0 us (PDL secondary of the 16-program boundary, which
  ends 51.3 / 50.5); pr8 ends 73.3 (22.0 after bnd), pr32 76.1 (25.6 after bnd); pr32's 132 one-per-SM CTAs also keep the
  rmsnorm from co-residing (starts 70.7 vs 48.9). Standalone "pr8 24.4 us" (proj s80, other day/contention) was not a fair
  baseline: in-layer pr8 streams 58.5 MB in ~22 us after bnd on the H100. Standalone with a 16x1024-thread pred (blocks 16
  SMs like the boundary) did not slow pr32 (GPU noisy then: 22.7 vs 23.7 us), so the late-CTA theory is unproven.
- H200 on paper: pr8 ~17 us after bnd vs pr32-F32 L2W proxy 17.25: no win there either. NOT handed off (no TUNE option:
  costs a build + tuner time for an H100 loss). Kept in tools/n5s8 (PR_F32 / MTOK 8 / PREDBT knob in pr32.py).
- Lesson: compare against the incumbent IN-LAYER (ab_flag) before building on a standalone number from another session.

## Night5 session 9 (2026-10-27 23:31Z - 23:55Z) -- lead 22:45Z step 1: batch-1 serial-chain ramp/tail audit (H100, main v144); 0 runs
- Tools: muse_moe_w4a16/tools/n5s9 (README). Bytes = ncu dram__bytes_read per kernel (cold L2, seed 99); times = graph-replay
  timeline (cold L2). H100 achievable read rate measured 3.12 TB/s (torch.sum over 2 GB). "excess" = segment time - bytes/3.12.
  Segment = previous segment's end -> this kernel's end (PDL overlap folded in).
- d1 sliding (85.1 us; 196 MB -> floor 62.8; excess 22.3):
    qkv g13      0 -> 13.2   29.0 MB  2.20 TB/s  excess 3.9   (2nd back-to-back call: 12.8 us: not a first-op effect; NOTES-proj
                                                               qkv m1 swept, 13.1-13.4 standalone, "no lever")
    csl + o g13  -> 22.5     13.7 MB  1.47        excess 4.9   (o's W prefetch already fills csl's window: pr1_npf/G10_PFL2 lost)
    bnd + pr1    -> 43.8     59.3 MB  2.78        excess 2.3
    rms + tk2f   -> 78.6     85.0 MB  2.44        excess 7.6   (3 experts; H200 s12 stamps: routing 1.5 + first copy 2.1, ring
                                                               absorbs the gemm13->gemm2 wait, CTA end spread ~2.3)
    n2u + bnd    -> 85.1      8.6 MB  1.33        excess 3.7
- v1 sliding (148.8 us; 341 MB -> floor 109.4; excess 39.4):
    qkv g14 0->14.8 28.6+0.6 MB 1.97 TB/s exc 5.5 | vtk + o g12 ->28.8 17.4 MB 1.24 exc 8.4 | bnd + pr8 ->55.3 60.2 MB 2.29 exc 7.1
    (bnd ends 4.0 us after o) | rms + tk2f ->141.1 226.7 MB (8 experts) 2.64 exc 13.1 | nu8 + bnd ->148.8 9.0 MB 1.17 exc 4.8.
- H200 (35dd53d0 profile, one input): d1 sl kernel durations tk2f 20.2, pr1 17.5, qkv 10.9 (29.7 MB = 2.7 TB/s), bnd 2x3.1,
  o 6.1, n2u 5.0, csl 3.9, rms 1.4; board 67.4 vs ~40 floor (lead). Same shape as the H100 table.
- So the summed sub-peak time is >= 8 us (d1 22, v1 39 H100), BUT what a persistent post-attention chain kernel
  (o -> bnd -> pr1 -> rms -> tk2f -> n2u -> bnd) could remove is much smaller; per segment, from measurements already on record:
  * front qkv + csl/o (8.8 d1 / 13.9 v1): OUTSIDE the chain; qkv's ~3.8 us is the GEMV's own ramp+drain (also as 2nd call);
    the csl window is already used by o's W prefetch (layer s74/s78: pre-wait pooler prefetch there made o/csl slower).
  * o -> bnd -> pr1 (2.3 d1 / 7.1 v1): pr1/pr8 already PDL-prefetch W during bnd; one launch saved minus a grid barrier:
    <= ~1 us (layer s78). v1's bnd ending 4 us after o is a layer item (T=8 boundary), not a chain-kernel item.
  * pr1 -> routing -> tk2f (in tk2f's 7.6 / 13.1): routing 1.5 + first-copy DRAM latency 2.1 are data dependencies (experts
    unknown until routing): a chain kernel keeps them; TK2FX already starts tk2f 0.4-0.5 us after pr1. Ring absorbs ph1->ph2.
    Drain spread ~2.3 us: the next op needs the full expert sum (norm2 over all hidden rows): grid barrier either way.
  * tail n2u + bnd (3.7 / 4.8 excess, 6.5 / 7.7 us after tk2f end): the only part with real room. n2u already loads its W
    into registers + L2 before griddepcontrol.wait, but its blocks become resident only as tk2f CTAs exit; the warm-W oracle
    (layer s16) gave -1.4 us; a fused tail with the unpool slice staged in tk2f's spare smem (156 KB x 132 = 20 MB >= 8.5 MB)
    during the routing gap would leave ~flag + Y read + reduce + boundary ~= 4 us -> bound ~-2.5 us d1 H100, but every
    in-kernel barrier version so far lost (N2U_BOUNDARY +2.8, BND_SPIN +2, UPF/UPF2/UPF3 tie..+7.8).
  => addressable bound ~3-4.5 us d1 (front <=1 + tail ~2.5 + routing <=1), realistically 1-3 after barrier costs (~0.3-0.6%
     overall): BELOW the lead's 8 us step-2 threshold. Stopped at step 1 as instructed; no persistent kernel built.
- The biggest sub-peak items are per-kernel ramp+drain (~3-4 us per HBM-streaming kernel: qkv m1, g30 fit 3.9 us fixed) and
  data-dependency latency (routing -> first expert byte), not launch gaps between kernels.

## n5s10 (2026-10-27 23:38Z - 2026-10-28 00:30Z): H200 verdict on pr32 = DEAD
- Source: layer's 00:00Z v144 check 4937830f, ~/r-layer/muse_decode_layer/profiles/submission.4937830fb2ff4921.log,
  parsed with tools/n5s10/pr32_tune.py (H200 prepare tuner, us per layer):
    T=16 sliding: default 166.3, pr32 169.3 (+3.0) | full: default 223.5, pr32 226.1 (+2.6)
    T=32 sliding: default 169.2, pr32 167.9, pr32+g30n:unpool 164.4, g30k:pooler 104 + g30n:unpool 162.1 (PICKED)
    T=32 full:    default 329.6, pr32 329.0, pr32+g30n:unpool 325.3, g30k:pooler 104 + g30n:unpool 323.0 (+qkv_gate 322.9 PICKED)
  => pr32 (one-launch pooler + router + routing) loses 2.3 us to g30k pooler 104 + side-stream router at T=32 and ~3 us to the
  default at T=16. The L2W proxy (n5s2: 15-17 us, chain hidden) did not hold in-layer on the H200: fusing the router into the
  pooler's streaming CTAs + finisher chain costs more than it saves vs a separate side-stream router. Without an H200 in-layer
  timeline (not available to moe: layer checks only) there is no evidence to tune against; closed.
- Ask to layer: drop {"pr32": ...} options from TUNE[16]/TUNE[32] (and the pr32 builds) to cut cold prepare time.
- 0 runs. Note: I accidentally ran `git pull` in ~/r-layer this session (fast-forward, clean tree, no harm); read r-layer only.

## n5s11 (2026-10-28 00:24Z - ): CUDA-comment stripper for the layer build (1 MB judge upload limit); 0 runs
- Lane: none open (n5s10 asked the lead). Re-read the H200 data for anything unmeasured: layer check 4937830f per-kernel
  profile (v16 sl twm 165 + moe_norm 37; v32 sl twm 201 || hot 155; d32 sl tk2f 87) and the moe board (#1 2.62x vs our
  2.25x; our weakest shape decode 32 1.72x = 124.8 us vs SOL 70.8). Every lever found was already measured dead:
  expert-W L2 prefetch beside the d32 pooler (layer s55), twm ALU+tensor bound (s28/s29), GS16 (s12, 4 H200 diags), tk2f
  closed (layer s81). Nothing new to build there.
- Built instead (enabler): layer v145 is 963.7 KB of the judge's 1,000,000 B limit (layer n5s6: 36 KB headroom; any new
  kernel breaks it). tools/n5s11/cuda_strip.py strips // and /* */ comments from the 25 raw CUDA strings embedded in the
  built submission.py (string/char literals and backslash splices respected; comment-only lines dropped; each source's
  leading // header block kept for readers). Verified per string: `cpp -fpreprocessed -P` token streams equal; the 26
  runtime .replace() anchors all still present. v145: 963,742 -> 890,149 B (-73.6 KB; -86.6 KB without kept headers).
- First A/B failed at import: layer.py text-patches moe CUDA with anchors built in Python variables that include
  comments (TK2F_LATE_TRIG: '{  // routing, as moe_tk2 GATED'). Fix: a comment is KEPT if its text occurs in any other
  (non-embedded) string constant of the file. Result v145: 963,742 -> 890,598 B (-73.1 KB; headroom 36 -> 109 KB).
- Validation (/tmp copies of r-layer v145 = d372b7e, TUNE = {} appended to both so the tuner cannot pick differently):
  (1) per string `cpp -fpreprocessed -P` token streams equal (25 strings);
  (2) SASS: both files prepared with separate TORCH_EXTENSIONS_DIRs, tools/n5s11/sass_eq.py pairs .so by kernel-name set:
      every device build IDENTICAL (moe 115 kernels, g10 42, csl 16, g24 8, nds, nvs, nvscl, g27, g30 x7, g30k x2, pr32 x2);
      only muse_layer_lt1 (host-only cuBLASLt wrapper, 0 device kernels) differs, as expected (__LINE__ in host checks);
  (3) layer tools/files_eq.py stripped vs original, 12 scored shapes x 2 seeds: 11 non-identical, all 1-2 BF16 ulp at
      d16/d32 (+ v1 2e-4) and one decode 32 sliding seed-0 routing flip (1 id, maxdiff 1.85); CONTROL original vs an
      identical copy: 10 non-identical, same 1-2 ulp pattern at d16/d32. => those diffs are the layer's own run-to-run
      nondeterminism at T=16/32 (known since moe s18), not the strip (SASS identical). FYI layer: that nondeterminism can
      flip a near-tie route (seen once in 24 here), which is the routing-check risk the lead flagged 10-26.
- Not covered by (2): builds that only TUNE options trigger (g30n, g19, g22...): covered by (1) only.
- Use (layer build.py, after strip_comments): `sys.path.insert(0, <r-moe tools/n5s11>)` or copy cuda_strip.py into the
  layer dir; `out, k = cuda_strip.strip_file(out)` (asserts token equality per string, needs `cpp` = gcc's, present).
- 0 judge runs.

## n5s12 (2026-10-28 00:51Z - 01:10Z): no lane, no build; 0 runs
- Read COORD: no lead answer to the n5s10 lane request (newest lead entry still 22:45Z). Read layer's new H200 check
  133068b8 (2.1097x) profile + tuner lines: nothing new for moe. Experts per shape (us): d1 sl tk2f 28.2 / full 20.7,
  d16 sl 64.8, d32 sl 75.4 + norm2 11.2, v1 sl 63.8 / full 35.9 (routing-dependent expert counts, consistent with the
  H200 fit 14.1 + 6.97/expert), v16 sl twm 159.2 + moe_norm 33.6 (PDL wait), v32 sl twm 208.6 || hot 115.3. T=32 tuner
  again picks g30k104 + g30n unpool over pr32 (pr32+unp 164.7 vs 164.5 default-unp+qkv). Every lever there is already on
  record as measured dead (twm dq s18-s30, tk2f s81, expert L2 prefetch beside pooler s38/s55, router-first orders s3/s5).
- Re-checked proj's FYI (g30 S2 pooler beats g30k G104 by 3.2-3.5 us standalone, lost in-layer to the side-stream router):
  a serial router-before-pooler order nets <= ~0.5 us (router ~2-3 us on the critical path): not built.
- Housekeeping: STATE-moe.md compressed to 138 lines (n5s4-n5s11 session headers -> one NOW block).

## n5s13 (2026-10-28 01:10Z - 02:30Z): lead 01:10Z items (v1 o -> bnd -> pr8; T=8 tk2f first copy); 0 runs
Base: /tmp copy of r-layer main c0413d8 (v148), H100, `TUNE = {}` appended to every A/B file (with the tuner on, a
noisy pick of g30n:o silently replaced ob8 in the first A/B: always pin the tuner for flag A/Bs).
- v1 sliding timeline (tools/timeline.py, s99): o g12 ends ~28-30, Triton _boundary ends 2.5-4.1 us later, pr8 ends
  ~22 us after the boundary. Ablations:
  * boundary with both reductions removed (rb = rs = 1): bnd tail after o unchanged (3.7/3.9 vs 2.5-4.1): it is
    wait-release + L2 load + store latency, not compute.
  * ORACLE bnd free (its gdc_wait removed: wrong output, timing only): pr8 ends 22.0/22.4 us after o vs 24.9-27.6
    => upper bound ~3-4 us v1 sl (proj s80's oracle: -4.2).
- Built: OB8 TAIL mode (tools/n5s13: s86's ob8 kernel = o GEMM + algebraic boundary partials, but instead of the full
  grid barrier only the last KT blocks to finish (ticket) wait and do the elementwise pass; the other blocks exit so pr8
  ramps on their SMs). Per-block globaltimer stamps (ob8_stamps.py) found, in order:
  1. elementwise loop with stores between dependent loads: serialized L2 round trips (fixed: loads first, float4);
  2. pre-spin loads in the spinning thread made the spin end 6-10 us late (an acquire/fence waits for them);
  3. the remaining floor: ticket -> end ~4.5-5 us = 2 round trips (partials, then cold `s` from HBM) at ~2 us each
     while pr8's 100+ blocks ramp (HBM queues full). Loading s before the spin only moves the wait into the fence.
  The Triton boundary pays ~1 round trip after o (it loads s/alpha/beta/gain BEFORE its PDL wait, during o).
- H100 ab_files 5 rounds (tuner off), v1 sliding / full: main 144.51 / 156.19 | OB8 full barrier 147.20 / 158.14 |
  OB8 tail KT=32 145.92 / 157.18 => DEAD (+1.4 / +1.0 us). Not handed off.
- Lesson: on the H100, after o every extra global round trip costs ~2 us because pr8's pre-wait ramp saturates HBM;
  the boundary's 2.5-4 us is ~ 1 round trip + launch/flush hops. Any fused version must load its operands before the
  barrier (only the blocks' OWN columns are known then -> needs all blocks alive = OB8's full barrier, which blocks pr8).
- Item 2 (T=8 tk2f: first expert copy straight from the router's output): NOT built. Same timeline: tk2f (TK2FX, PDL
  secondary of the rmsnorm which triggers right after its wait) starts 0.3 us after pr8 ends. Moving tk2f's route
  prologue into pr8's routing block keeps the chain serial (same work, one global round trip either way; routing ->
  first weight byte = data dependency). Bound ~0.5-1 us v1 only, and per the round-trip lesson above likely 0.

## n5s14 (2026-10-28 01:34Z - 01:50Z): no lane, no build; 0 runs
- No lead answer after n5s13. Read the moe board (okc leaderboard muse_moe_w4a16): #1 pjmullin3 2.623x (also #1 layer 2.399x,
  attention 3.032x) vs our febfcb1f 2.247x. Per shape from `okc status febfcb1fac2d4824 --json` (median_ns / SGLang / floor us):
  bs1 27.0/77.2/11.4, bs16 88.9/185.3/52.5, bs32 120.1/204.1/70.8, bs8 68.3/149.7/37.3, bs128 204.2/465.5/101.3,
  bs256 247.8/630.7/108.9. Our x floor: 2.37/1.69/1.70/1.83/2.01/2.27. Profile: bs128 twm 196.8 + moe_norm (tail) 40.6 +
  routegather 4.5; bs256 twm 235.1 + 38.5 + 6.4; bs1 tk2f 20.1 + norm2 3.1.
  => the expert kernels are not at a hard bound: another team is ~17% faster in geomean. Posted to the lead as a lane request
  (Requests n5s14) with a 3-consumer-WG twm proposal (more warps per SMSP; caveat twq's 16 warps were also ~50% issue).

## n5s15 (2026-10-28 01:54Z - ~02:55Z): LANE DECISION 6 (moe board as H200 A/B bench), tk2f bytes in flight; 3 moe checks (DIAG)
All three: /tmp copies of main with `DIAG = 1` (layer routings, 5 seeds per load, cold L2, graph replay; moe-only),
test runs only, never submit. Variants live in DIAG builds only (#define TK2F_SQDEEP, set by _lib() when DIAG); the
default build is unchanged. Every variant bitwise = main locally (maxdiff 0), except nc16 (16-way reduction order, 1-2 ulp).
- ddfc1353 (81043ea, 2.2586x as test run): deeper / larger SQ rings WITH L2H 4 hints (the s12 ring test predates SQ + hints).
  H200 MEAN us (control sq / repeat sqb):
            sq    r5(5x18KB) r6(6x18) r8(8x18) k44(4x36) k45(5x36) sqb
    d1     30.3   30.8       31.0     32.2     32.1      32.4      30.3
    d16    64.7   65.6       65.8     67.3     68.4      71.3      64.7
    d32    84.8   87.4       87.2     87.6     87.8      89.6      84.9
    v1     64.5   64.7       65.5     66.9     67.7      69.1      64.7
- 14ddda06 (765f129, 2.2588x): shallower rings and 9 KB stages (sq8 = _sq_pack nq 8, KPS 1):
            sq    r3(3x18) r2(2x18) k14(4x9) k16(6x9) sqb
    d1     30.1   31.9     33.2     31.9     31.4     30.2
    d16    64.6   68.9     74.0     71.2     71.0     64.6
    d32    85.3   91.1     99.1     94.7     96.0     85.4
    v1     64.2   67.3     71.9     69.3     66.9     64.2
- f5430a5c (77a6514, 2.2520x): nc16 = 16 consumer warps x KPS 1 on the SAME 18 KB SQ stage image (chunk c = h*NC + warp
  maps identically; cfg 1604 / 85604; NB kernel_fn picks norm3 only for 8x4xx-style cfg digits: 81604 ran norm2 on slot-
  ordered y32 = wrong output, caught by tools-less local A/B /tmp/n5s15/t/nc16t.py): d1 32.3 / d16 66.6 / d32 87.4 / v1 65.6
  vs sq 30.3 / 64.4 / 85.4 / 64.3 => +1.3..2.2 us. More consumer warps per SMSP do not help.
- VERDICT: tk2f's stream is at a sharp H200 optimum = 4 x 18 KB (72 KB in flight per SM) with 8 consumer warps: less in
  flight loses a lot (r2 +14 us d32), more loses steadily (r5..k45 +0.5..4.8), smaller copies lose, more warps lose.
  The lead's 01:45Z lever (a) "bytes in flight per SM (deeper/larger TMA stages, 2 CTAs/SM)" is closed on the H200
  (2 CTAs/SM: n4s?, register cap, dead on the proxy). Control repeats agree to <= 0.2 us (DIAG noise floor).
- Reading: since more in-flight bytes per SM HURT, the HBM is saturated with request queueing at 72 KB/SM; the stream rate
  (~4.06 TB/s per expert) is the H200's practical rate for this access pattern, and the remaining tk2f costs are not in
  how the bytes are requested. What is left per the H200 fit (14.1 + 6.97 E + 4.4 X): the 4.4 us per extra 8-token group
  (layer d32 has 2-3: consumer-bound L2-served rows; nc16 did not speed them up) and the fixed part (routing 1.5 us,
  norm2 ~4.5 incl. gap: in-layer mostly hidden by PDL).
- Not done (board-only, no layer value): standalone norm2/norm3 as a PDL secondary of tk2f (g_pdl bit 2; s9 H100 -0.4 us
  at bs1, never tried on the H200 with tk2f).
- 9af0d239 (1aa4ba3, DIAG_STREAM2, 2.2565x test run): H200 moe_stream probe, same unique bytes (each routed expert once, no hints),
  BLOCKED count split (each SM its own contiguous block = every kernel of ours) vs CYCLIC grid-stride deal of 1 / 4 / 16
  consecutive stages per CTA (whole GPU reads one compact window). MEAN us (TB/s over unique bytes), NSTG 4:
            blk18  cy1_18 cy4_18 cy16_18 | blk36  cy1_36 cy4_36 cy16_36
    d1      23.5   23.2   23.4   27.0    | 21.6   21.3   21.6   25.1
    d16     57.0   57.0   57.3   61.1    | 51.8   51.9   52.4   55.7
    d32     72.5   72.4   72.7   74.5    | 66.3   66.0   66.5   68.7   (blk36 4.01 TB/s)
    v1      60.0   60.1   60.5   62.4    | 54.7   54.5   55.0   56.9
    v16    127.8  128.8  128.1  130.2    |119.0  119.2  119.5  122.1   (4.23 TB/s)
    v32    139.9  141.4  139.9  142.2    |132.4  132.4  132.7  135.2
  => grid-stride = blocked (+-1%); cy16 worse (coarse deal = tail). DRAM page locality across SMs is NOT a lever. DEAD.
  (H100 locally: same tie.) The probe again prefers 36 KB stages at NSTG 4 (144 KB in flight), which the REAL kernel
  rejects (k44 +3 us d32 above): the probe's consumers are free; tk2f's are not.
- Gap budget at d32 (layer routings, H200): probe at tk2f's own ring (blk18 = 4 x 18 KB, each expert once) 72.5 us vs tk2f
  85.3: ~13 us = the multi-group re-read rows (d32 has 2-3 extra 8-token groups x ~4.4 us = 9-13 us, consumer-bound
  L2-served rows, see fit) + routing ~1.5 + norm2. => NEXT for tk2f: make an extra 8-token group cheap (both groups per
  dq pass) -- GS16/tk2h lost before (xs smem LDS + regs + i-cache; "2-group row ~6 us per segment"); find out WHY a
  2-group row costs more than two 1-group rows before building again (SASS/stall audit of tk2g_row TWO on the H100).

## n5s16 (2026-10-28 02:36Z - 03:40Z): extra 8-token groups: H200 oracle bound + GS16 hybrid consumer (DEAD: i-cache); 1 moe check (DIAG)
- TK2F_ORACLE (DIAG builds; run-time via moe_set_l2h2(v + 1000) -> kernel arg l2h bit 12; WRONG output, timing only):
  tk2f computes only each expert's first token group (ng capped at 1). A first version keyed a 2nd library on it:
  local DIAG prepare 280 s (judge limit 300 s!) -> made it run-time (prepare back to 141 s). Default build bitwise as before.
- H200 check e19541a1 (fb063c2 + DIAG=1, 2.2557x test run, never submit), MEAN us over 5 layer seeds:
            sq    or1(oracle) h3(tk2h W3)  sqb
    d1     30.1   30.0        30.0         30.0
    d16    65.0   61.0        66.3         64.9    (s99 has no 9+-token expert: or1 = sq there; h3 +2.6 = tk2h's 1-group-row overhead)
    d32    83.7   76.0        80.9         83.7
    v1     64.5   64.4        64.3         64.3
  => the extra token groups cost d16 4.0 us, d32 7.7 us on the H200 (the prize for a "free" 2nd group). tk2h W3 (the layer's
  d32 pick) recovers 2.8 of 7.7 at d32 and loses 1.3 at d16 (its 1-group rows are slower).
- Built HYB (tk2x_consume, PF 8192, cfg 1006W004 SQ, python TK2X_M): GS16 segments, 1-group rows = compile-time-gated tk2g_row
  (= tk2f's body), 2-group rows = ONE run-time-gated TWO body for both phases (3 bodies). Bitwise = sq (maxdiff 0, 15/15).
  H100 DIAG (layer routings): d16 x3 92.3 vs sq 83.6, d32 112.5 vs 108.6: LOSES, and also at d16 s99 (no 2-group row ever)
  111.7 vs 106.8. Probe xn (same kernel, TWO body compiled out, wrong output for 2-group rows): s99 106.8 = sq, d32 102.4 ~ or1.
  ncu (tools/n5s16/ncu_x.py, d16 s99): sq / xn / x3 inst 32.1M / 31.7M / 32.0M, "no instruction" stall 1.33% / 1.33% / 4.40%.
  => the mere PRESENCE of an inlined, never-executed TWO body costs ~5% per 1-group row: i-cache (kernel 6736 vs 5048 SASS
  instr). Same cause as s10/s12's GS16 loss; tk2h's +4% on 1-group rows is likely the same thing (n17/n18 suspects).
- HYB2 (tk2x_two __noinline__ for all 2-group rows of a segment, PF 32768, cfg 1005W004, TK2X2_M): MUCH worse (d16 s99 125.9):
  the call forces the ABI (REG 168 cap + 328 B stack). Dead. Both kept behind #define TK2X (0; DIAG builds set 2).
- Lesson: in tk2f any extra main-loop body costs ~5%/row via the i-cache even if never run; a GS16 that wins must have ONE
  small body for both group counts. Idea for that (NOT built): x in SMEM for both groups (no xf registers) so the stage loop
  can stay rolled (`#pragma unroll 1` over q, one ~200-instr stage body) and the 2nd group is a uniform branch per STAGE (not
  per mma). Smem: ring 73.7 KB + x 2 groups x 8 warps x 8 KB = 128 KB + s_red single-buffered 16 KB + ~4 KB statics = ~222 KB
  (< 227 KB). Risks: LDS of x per mma for 1-group rows (+1 LDS.128 per 2 mma), loss of unrolled ILP. Prize: <= d32 7.7 / d16 4.0.
- TK2S (built, a072c85; PF 65536, cfg 29004 GS8 / 1W004 GS16 in SQ, python TK2S_M {M: W}, W < 0 = GS8; DIAG builds only):
  ONE main-loop body for 1- and 2-group rows: x of the segment's groups in SMEM (lane-private [grp][qq][2] x 32 lanes, 8 KB per
  warp and group), stage loop rolled (#pragma unroll 1 over q), 2nd group = uniform branch; GS16 s_red single-buffered + 1 extra
  bar.sync per row (smem: 73.7 KB ring + 128 KB x + 16 KB red fits). Bitwise = sq everywhere (maxdiff 0, bench test 15/15).
  H100 DIAG: s8 +0.3..1.2 us; s0/s3 (GS16 W 0/3) d16 +2.7/+4.6, d32 +4.1/+3.5.
  H200 check 8ed43995 (a072c85 + DIAG, 2.2354x test run, never submit), MEAN us:
            sq    or1   h3    s8    s0    s3    sqb
    d1     30.1  30.1  30.2  30.8  30.2  30.1  30.0
    d16    64.8  60.7  66.8  68.5  77.1  76.4  64.5
    d32    83.7  75.8  81.1  91.8 100.4  96.4  83.6
    v1     64.5  64.2  64.3  65.5  64.5  64.3  64.3
  => DEAD. On the H200 tk2f's consumer is critical: the rolled smem-x body costs +3.7 (d16) / +8.1 (d32) even with ONE group,
  and the GS16 versions +12..17 us. The H100 hid most of it again (DRAM-bound). The oracle (or1) re-measured: d16 -4.1, d32 -7.9.
- CONCLUSION extra-group lever: every way to share a dequant across 16 tokens found so far loses on the H200: inlined extra body
  (i-cache, +5%/row), noinline (ABI spills), merged run-time-gated body (tk2h, +4%/1-group row), rolled smem-x body (+10%).
  tk2h W3 at d32 (layer's pick, -2.8) stays the best. Remaining untried shape: a SEPARATE kernel (or a separate CTA role inside
  tk2f, e.g. the last k CTAs) that runs only the 2-group segments with its own register-x body, concurrently with tk2f's 1-group
  rows -> one body per CTA role, no i-cache sharing. Not built; prize <= d32 ~5 (vs h3), d16 ~4 us.
- Checks 10-28 used by moe: 6/20 (+ e19541a1, 8ed43995). 0 submits.

## n5s17 (2026-10-28 03:16Z - 04:45Z): tk2f code LAYOUT probes (H100 ncu, layer routings); 0 checks
Question: does a 2-group (TWO) body cost tk2f's 1-group rows ~5-12% just by EXISTING (n5s16 HYB lesson), so a separate
CTA role is needed? All probes are DIAG-only template bits (default SASS unchanged; bench test 15/15 OK). Tools:
tools/n5s17/ncu_r.py (VAR=... one tk2f call on a DIAG layer routing under ncu), tools/n5s17/loop64k.py <so> [regex] [min] [v|all]
(lists innermost MMA loops that straddle a function-relative 64 KB boundary; v = print all MMA loops).
ncu us (d16 s7 / d16 s99), stall_no_instruction samples in ():
- sq (cfg 84204 SQ)                          80.7 (109) / 107.8 (133)
- r1 dead TWO body in a never-taken TOP-LEVEL branch BEFORE the main path   92.1 (1069) / 119.4 (981)   = +12%
- r2 same dead branch AFTER the main path                                   81.4 (107) / 108.3 (125)   = free
- p3..p8: r1 + 1..16 extra dead instrs (alignment)                          89.5..93.7  (all slow: not line alignment)
- p9: default minus the dead tk2w_bound split code (hot loops 10 KB earlier) 79.4-79.7 / 106.5-108.6     = tie (no slope)
- pad-only dead blocks before the main path: +2/+3/+4/+6 x 512 B: 79.9-81.1 (fast); +10 KB 81.0 (107);
  +16 x 512 B (gemm2 loop ends 1 KB past 0x10000) 80.8 (277); +15.5 KB (gemm2 loop 8.7 KB past 0x10000) 84.7 (444) = +4.4%.
  SASS: in r1 the gemm13 consumer loop sits at 0xd620-0x125a0 (crosses 0x10000); every fast build has no MMA loop crossing
  a 64 KB boundary (the producer warp's loop crossing is harmless).
=> RULE (H100, fits all plain-tk2f probes): a hot consumer loop straddling a function-relative 64 KB code boundary is slow
  (+4% for ~9 KB past it, +12% for the gemm13 loop); dead code itself is free when it doesn't move the hot loops across.
- BUT HYB (n5s16, GS16 3 bodies; its gemm2 loop 0xd150-0x162a0 crosses) padded so NO loop crosses (z3: dead 12 KB before
  gemm2 -> 0x10190-0x192e0; w3: 14 KB): still slow: d16 s99 118.0/116.6 vs x3 114.0 vs sq 107.8; d16 s7 91.1/91.3 vs x3 93.5
  vs 80.7; d32 s7 117.4/117.2 vs x3 117.8 vs 108.7 (no_instr 660-820 vs sq 110-150). HYB's per-row loop is 39 KB (1-group
  64 + TWO 128 MMAs in ONE loop): that loop is slow on its own, crossing or not. So for HYB/GS16 the cost is the big
  combined loop, not the 64 KB boundary. The old GS16 kernels also have crossing loops (loop64k), so they paid both.
- loop64k over the moe lib: twm/twq inner loops do not cross; tk2h (MRG) does not cross. The layer's libs were not in this
  cache (not scanned): layer could run the tool on its own builds.
- Design consequence (NOT built): TWO rows must run in their OWN loop, placed AFTER the 1-group loops (r2 = free), e.g. per
  CTA: [gemm13 1-group rows] [gemm13 TWO rows] [gemm2 1-group rows] [gemm2 TWO rows] with the TWO loop a single run-time-
  gated call site (code after the main loops), separate row spaces (1-group space = odd leftover groups, TWO space = group
  pairs), each split evenly over all CTAs (balanced without a weight), producer streaming in the same order. Unknowns: the
  TWO loop's own rate in isolation (17 KB loop; tk2g TWO = ~1.15x a 1-group row by instruction count), and whether the
  extra ordering/prologue eats the prize (H200 oracle: extra groups cost d16 4.0 / d32 7.7 us; layer's h3 at d32 gets 2.8).

## n5s18 (2026-10-28 03:55Z - ): "deferred TWO" tk2f built (n5s17's NEXT) = DEAD on the H100; 0 checks
- Built (r-moe, DIAG builds only: #define TK2DT, python TK2DT_M {M: 1}, cfg 85204 SQ + PF4 -> moe_norm3, PF bit 524288):
  separate row spaces per expert: 1-group space = the odd leftover group (s_rs13/s_rs2, s_ob[a] = its first token), pair
  space = ng/2 16-token pairs (s_rt13/s_rt2); both split evenly over the CTAs (pairs in reverse CTA order); producer streams
  13 1-group, 13 pairs, 2 1-group, 2 pairs; consumers: tk2f_consume (unchanged body) + tk2t_consume (new: ONE run-time-gated
  tk2g_row TWO body for both phases, x group 1 lane-private in 64 KB dyn smem); L2 hints: evict_last only for ng > 2 experts.
  Bitwise = sq on all 20 decode DIAG routings + 6 synthetic hot-expert routings (up to 32 tokens/expert = 2 pairs),
  tools/n5s18/dt_t.py. Layout: the compiler places the pair loop right after the gemm13 loop whatever the source order
  (state-machine loop tried) -> it straddled 0x10000; a 20 KB dead pad (never-taken `nact > 32` branch) moves it to
  0x10300-0x16120, gemm2 1-group loop to 0x16220-0x1ac10 (loop64k: no crossing).
- H100 ncu (tools/n5s18/ncu_dt.py VAR=sq|dt, 3 runs each, GPU shared): d16 s7 sq 87.9-90.1 / dt 106.8-109.5 us;
  d16 s99 (NO pair rows) 110-116 / 120-127; d32 s11 126.3 / 155.0; d32 s99 103.2 / 126.5. Instructions -1..-11%.
- Why (ncu source counters, stall samples per M instructions): 1-group gemm13 loop 285 (= sq 289), gemm2 1-group loop 307
  vs sq 242 (+27%, it now sits at 0x16220: more no-instruction stalls, 0.07 -> 0.39 per issue at s99), pair loop 478 (1.65x):
  a pair row costs about what two 1-group rows cost (d16 s7: pair loop 1964 samples vs ~2030 for the 1-group equivalent).
  Registers: DT kernel REG 168 (the cap at 288 threads) + 168 B stack (spills at loop edges, a few LDL per pair row);
  sq REG 160, no stack. TWO needs group-0 x (64 regs) + 2 accumulator sets + wv: it cannot fit in 168 next to the rest.
  Tested and ruled out: +64 KB dyn smem on sq alone = no cost (d16 s99 108.4).
- CONCLUSION (with n5s10/s12/n5s16/n5s17): in the mma.sync tk2f, a 16-token shared dequant never beats two 8-token rows:
  inlined (i-cache), noinline (ABI), merged (tk2h +4%), rolled smem-x (TK2S +10% H200), separate loop (DT: pair ~= 2 rows,
  and the layout move costs the other loops). The extra-group cost (H200 oracle d16 4.0 / d32 7.7 us) stays; layer's tk2h
  W3 at d32 is the best form. Not worth an H200 check (the H200 is more consumer-critical than the H100).
- tk3 costing (wgmma decode consumer, the last 16-token form): tools/n5s18/mb_warm.py = s28 mb_mc with WMB (MB of weights;
  24 = L2-resident = consumer-bound). G132 warm full-mode cycles per 9 KB stage (8 KB W + 1 KB scales = 16384 weights):
  N8 596 / N16 634 / N32 695 (ring alone 307); cold (512 MB, HBM): 704 / 703 / 720 (ring alone 763). tk2f warm, 1 expert
  (tools/l2warm.py VS=84204,204): T=8 19.9, T=16 22.3, T=32 34.1 us (non-linear, fixed costs dominate: not a clean rate).
  Rates: wgmma N16 consumer ~3.8 TB/s-equiv (H200 clock, 132 SMs) < HBM 4.06; tk2f consumer from the H200 fit ~6.3 TB/s per
  8-token group. => tk3 saves ~4 us per 2-group expert, costs ~+0.3 per 1-group expert, before row/epilogue overheads
  (absent in the microbench): est. d32 -3..-5 net of layer's tk2h W3, d16 ~0. NOT building. Extra-group lever closed.

## n5s19 (2026-10-28 04:25Z - 05:00Z): norm2 as PDL secondary of tk2f (the last untried small item): DEAD on the H200; 1 moe check (DIAG)
- No lead answer yet to the n5s18 lane request; took STATE's "untried small item".
- Built NORM2_PDL (python global, default 0; r-moe 153ebb8): kernel_fn calls moe_set_pdl(4 | 2) before moe_norm2/3 so it
  launches as a PDL secondary (tk2f triggers launch_dependents at entry). SASS of moe_norm2<true>: ACQBULK at 0x100, no
  load above it (w1 loads sunk below the wait, Y loads LDG.CONSTANT after it). bench test 15/15 with it on.
- H200 DIAG check 138051ec (DIAG dec = sq / np / sqb / npb, layer routings, MEAN us):
      d1 29.9 / 33.5 / 29.9 / 33.1    d16 64.1 / 65.4 / 64.2 / 65.2    d32 84.6 / 85.4 / 84.6 / 85.4    v1 64.2 / 65.3 / 64.2 / 65.1
  => PDL norm2 LOSES every load (d1 +3.4..3.6 us, others +0.8..1.3), repeatable to 0.4 us; same verdict as s5 with tk2.
  Whole-run 2.2426x (DIAG build; board rows not comparable). Option kept off; do not retry.

## n5s20 (2026-10-28 04:33Z - 04:55Z): no lane answer; audit only, no build; 0 runs
- Lead has not answered the n5s18 lane request (COORD Requests). No proj/twm* branch exists yet to review (LANE 7).
- Board: #1 pjmullin3 now 2.685x (5008 us/step; was 2.623x), ours 2.247x. Our decode shapes vs the TASK.md 4.8 TB/s floors
  (febfcb1f): bs16 88.9/52.5, bs32 120.1/70.8, bs8 68.3/37.3; at the H200's measured practical rate for our pattern
  (4.06 TB/s, n5s15) those floors become ~62/84/44 us, so the decode excess is ~14-36 us = the H200 fit's fixed 14.1 us +
  4.4 us per extra 8-token group. Every piece of that has a measured dead end (fixed: routing 1.5 + first fill 2.1 =
  data dependency/ring fill, FP producer self-routing lost in-layer s15; phase barrier absorbed by the ring n10; tail
  spread 2.3 = grid-wide dependency of norm2; norm2 PDL n5s19 dead; extra groups n5s16-n5s18 dead; tk3 costed n5s18).
- Re-checked the parked barrier-free d1 (split-I gemm13->gemm2 with FP32 partials): still <= ~0.6 us d1 (s25/n10 paper,
  9 -> 8.73 row rounding only); not built.
- So the gap to #1 on this board is most likely big-M (bs128/256: twm 197/235 us + moe_norm tail ~40 us vs floors
  101/109), which is proj's lane since LANE 7. Request refreshed in COORD.

## n5s21 (2026-10-28 04:52Z - 05:45Z): LANE 7b (lead 04:50Z: moe = bs256 hot-expert structure): exact-N moe_hot costed + H200 probe = BELOW the 5 us bar; 1 moe check (DIAG)
- Routings: layer v32 (embedded _DIAG_R) = 18-21 experts, 2 hot (73-98 tok), 1-2 of 33-50, rest <= 26. Board bs256
  (reference.generate_input seeds 1-12, judge seed unknown): ~24 experts; 9/12 seeds have an expert > 64 tok (75-223), 5/12 one of 65-128.
- Cost model (posted COORD Requests n5s21): bytes ~510-590 MB = 128-148 us at 4.0 TB/s vs twm 211.8 || hot 114.5 in-layer (af87d466) =
  consumer-bound. Stage cycles RT2 8 KB: dq 377 + tensor 8*N + ~230 (fits N64 items 36 us, hot N128 ~52 us). Untested lever found: the
  hot experts are padded to N128 (s27 pad_bound.py only covered <= 32-tok experts): exact N saves ~78 tok/call = ~970 SM-us = model -7 us.
- Probe (kept, off): #define HOT_NX (DIAG builds) adds moe_hot<.., NH 104> (x box stays 128 rows, wgmma N 104; class 65..104), python
  HOT_NXB = 16 picks it (probe only: wrong for experts of 105..128 tok). wgmma generator now emits N 8..128 (adds 104/112/120). Bitwise = N128.
- H100 (tools/n5s21/nx_ab.py, /tmp/n5s21/prof.py): total tie (+1..3), moe_hot itself -5..22 us but twm +0..6: H100 is DRAM-bound at v32.
- H200 DIAG 43eb5326 (2.2464x test run, never submit), layer routings moe-only cold, MEAN us (seeds 7/11/13/17/99):
    v32 h32 201.5 (192.7 201.0 199.2 200.1 214.3) | n104 199.2 (185.8 199.6 200.3 200.4 210.1) | repeats 201.7 / 199.1; v16 tie (no hot).
  => -2.3 us mean for 48 tok saved per seed (model 4.5): ~half converts (twm decides the makespan on seeds 11/13/17). Full exact-N (avg 78
  tok) extrapolates to ~-3.7 us/v32 call < the lead's 5 us bar: NOT building the per-expert-N (noinline per-N item) version.
- Board note: main runs no moe_hot at bs256 (HOT_M = {}); the s29 note "board bs256 has no expert > 49 tok" held for seed 7 only.
  HOT_M {256: 32} + TWF_SPLIT_M {256: 66} (= the layer's config) is an untested BOARD-only config A/B (no layer gain).

## n5s22 (2026-10-28 05:15Z - 05:50Z): board-only bs256 config = the layer's (HOT_M {256: 32} + TWF_SPLIT_M {256: 66}); 1 check + 1 submit
- Re-read s14/s15/s24/s27-s30 for an untried bs256 lever with >= 5 us: none (LDS stage-ahead prefetch dead s14/s24, HOT_TWG dead n15,
  pingpong/warp-spec dead s22/s27 (ALU and tensor work don't overlap SM-wide), per-stage overhead bisected s28, exact-N n5s21 -2.3 us).
- So spent one check on the n5s21 untested item: main 21deb8a = HOT_M {256: 32} + TWF_SPLIT_M {256: 66} (what layer runs at T=256).
  Local: bench test 15/15, tools/s29/hot_stress.py 0/40 bad, 0 non-bitwise vs HOT off.
- check cefcc133 2.2632x: bs1 27.9, bs16 88.6, bs32 120.6, bs8 62.9, bs128 203.6, bs256 240.5 (ed55a5c0 main: 245.6).
- submit 951c17e0 **2.2535x** (new moe board best, was febfcb1f 2.2469x): bs1 27.0, bs16 94.6, bs32 126.6, bs8 66.9, bs128 200.1,
  bs256 242.5. bs256 -3..5 us vs older runs (within ~2% noise, consistent with the layer's own H200 tuner picks). No layer change (layer already runs it).

## n5s23 (2026-10-28 05:25Z - 05:50Z): moe_routegather latency fix (route+gather before twm at M > 32); 1 check + 1 submit
- Found via H100 graph timelines (tools/n5s23/tlg.py <bs> <seeds>: torch.profiler on graph replays, cold L2): moe_routegather
  took 5.6 us (bs128) / 8.6 us (bs256) on the H100 and 4.4 / 6.3 us in the H200 profile of 951c17e0, on the critical path
  (twm waits for it). Cause: the id-scan loop issued one dependent __ldcg round trip per 256 ids (the __match_any_sync
  between iterations stops the compiler from batching), then the x row loads, then up to 8 serial __ldcg of the token's ids.
- Fix (on main eb8c03c): x row loads + the token's 8 ids issued right after griddepcontrol; scan loads 8 ids/thread in one
  batch (RGU 8) before the match/atomic loop; repeated-id fixup via __shfl_sync. Integer counts: bitwise = before
  (tools/n5s23/dump.py: 40 big-M routings bs33/128/256/257 normal+skewed, eager + graph), bench test 15/15.
  H100: bs128 5.6 -> 3.7, bs256 8.6 -> 5.8 us. (Step 1 alone, batched scan: 4.8 / 6.5.)
- check d3761aa5 2.2641x: H200 routegather bs128 4.4 -> 3.2, bs256 6.3 -> 4.4 us; bs256 233.3 (was 240.5/242.5).
- submit 4fb734c6 **2.2703x** = new moe board best (bs1 26.4, bs16 89.5, bs32 120.9, bs8 68.0, bs128 201.1, bs256 236.3).
- LAYER: its moe_routegathern (v32: 7.4 us H200 in af87d466; v16 runs moe_route 1.9 + moe_gathern 3.6) has the same pattern.
  tools/n5s23/patch_layer_rg.py <layer.py> ports the same 3 changes (applies cleanly to r-layer 2f66dbb; NOT compiled/tested
  in-layer). Expect ~-2..3 us at v32 and possibly RG winning at v16 too. Posted under COORD Handoffs.
- Other small latency-bound kernels may have the same serial-load-in-loop pattern: worth grepping (moe_route, moe_norm prologue).

## n5s24 (2026-10-28 05:45Z - 06:10Z): latency audit of moe_norm2 / moe_norm (STATE n5s23 NEXT); 1 moe check, 0 submits
- SASS of moe_norm2<true> (main d3fc872): ids loaded as 8 LDG.STRONG.GPU in one batch, then predicated Y loads (fine), but
  tw[tok*8+s] was loaded AFTER the ssum barrier, one LDG.STRONG.GPU per local slot inside its own branch = serial L2 trips.
  moe_norm (big-M, NORM_EARLY rdy path): ids loaded pre-spin by tid < 8, then reloaded by everyone (__ldcg) after the spin,
  tw again per slot after the barrier.
- Fix (main 725f39a): moe_norm2 loads ids + tw for all 8 slots in one batch right after griddepcontrol.wait; moe_norm keeps
  ids + tw in smem (loaded pre-spin by tid < 8; after the wait in the non-rdy path). Bitwise = before
  (tools/n5s24/dump.py: 60 routings bs1/2/8/16/32/33/128/256 normal + skewed, eager + graph), bench test 15/15.
  H100 paired ab (tools/n5s24/ab.py A.py B.py: two submissions in one process, graph replay, cold L2, alternating): tie
  (+0.6/0/0/+1.1/-0.5/+0.9 us bs1..256, noise).
- H200 check e3bd482e 2.2713x (vs d3761aa5 2.2641x, same code otherwise): norm2 per-kernel 3.1/3.5/3.9/3.3 -> 3.7/3.7/3.4/3.3 us
  (bs1/16/32/8); big-M GPU busy 210.8 -> 218.1 (bs128), 236.4 -> 225.9 (bs256): all noise. => the tw trips were hidden
  (rsqrt/LDS chain; ~1 local slot/token). DEAD as a speedup; kept on main (bitwise, harmless). Not handed to layer
  (layer's moe_src moe_norm/moe_norm2 have the same pattern but nothing to gain). Not submitted (2.2713 vs board 2.2703 = noise).
- proj/twm (7ae024a, 3 independent consumer WGs) is self-declared dead by proj (H100 +25..43%): nothing to review.
- New tool: tools/n5s24/ab.py = paired A/B of two submission.py files in one process (use for any bitwise-equal latency change).
