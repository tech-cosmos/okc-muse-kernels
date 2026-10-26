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
