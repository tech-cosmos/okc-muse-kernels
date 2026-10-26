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
