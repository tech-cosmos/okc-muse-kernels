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

## Session 2 (2026-10-25 ~19:56Z-21:20Z)
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
