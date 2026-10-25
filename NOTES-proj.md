# NOTES-proj (muse_bf16_proj) — long log, newest at the bottom

## Session 1 (2026-10-25 19:18-20:20 UTC)
- v0 (da87f72): SGLang-equivalent torch (F.linear; router FP32 <=8, cuBLAS hi/lo above). submit 5fd07be78b7a4c9a
  1.0016x: first row on the board. SGLang per-shape times in STATE.
- Local harness bench.py (judge-like: graph replay, cold L2 by reading 512 MB, minus empty graph).
- BF16 GEMV prototypes (all mma.sync m16n8k16 swap-AB, W packed so a 16x16 tile's A fragments are 512 B
  contiguous: view(n/16,2,8,k/16,2,4,2).permute(0,3,2,5,4,1,6)); all correct, none beats cuBLAS on the H100:
  * gemv.cu: grid (stripe groups, K chunks), regs. qkv m1 18.75 (cuBLAS 18.2), m32 34.8. 55% DRAM peak (ncu).
  * gemv2.cu: TMA bulk pipeline, grid (groups, splits). smem limited to 2 blocks/SM; 288 blocks = 2 waves.
  * gemv3.cu: persistent stream-K + TMA. NOCOMPUTE same speed -> not math. 66 blocks take exactly 2x of 132:
    ~18 GB/s per SM ingest no matter NS (2..8) or C. fixed cost ~9 us vs cuBLAS 5.8. Dead end as written.
  * gemv4.cu: warp-level stream-K, U tiles in regs, x frags via __ldg. m1 ~ cuBLAS+1us; m>=16 2-4x slower:
    x fragments re-read from L2 per tile (lts bytes 58 MB vs 17 MB W at o m16). Flush ~2 us.
  * dev/rd.py: pure-read probes (numbers in STATE). Contiguous-per-warp vs strided order: only 3-6% apart.
- Router FP32 kernel dev/router.cu (x loads issued before W loads: -0.3 us). H100: m1 7.2 (cuBLAS 11.97),
  m8 9.4 (27.8), m16 12.7 (18.1), m32 (M=32) 27 (bad: smem-bound) -> FP32 only to m16.
- check 510a51f0985d4f57 (c111b7f, router FP32 m<=16 + gemv4 hi/lo m17-32): 1.0594x. router m32 hi/lo 0.941x.
- check ea63620f6ece470e (087192f, m17-32 back on cuBLAS): 1.0776x. submit 0b870721666d40c7 same code: 1.0699x,
  rank 926 (overall 456). H200 router: m1 5.38 (10.56), m8 7.58 (26.05), m16 10.88 (14.66).
- torch.addmm(out_dtype=fp32) to fuse the hi/lo add: within noise; not used.
- zsh gotcha: `for m in $ms` doesn't split words; python hash() is salted per process (use hashlib for build names).
