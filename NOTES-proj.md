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

## Session 2 (2026-10-25 20:00-21:10 UTC)
- dev/gemv5.cu: block-tiled mma GEMV, x[token group, K range] staged in smem (cp.async, before W loads), B via
  ldmatrix, W register-direct (U tiles), split K over a thread-block cluster through DSMEM (no atomics/workspace).
  Correct; slower than cuBLAS (qkv m1 19.5 vs 18.2, pooler m32 43 vs 28). ncu: cluster grids > max active clusters
  (cs 16: 21 clusters; cs 8 at 2 blocks/SM: 30) -> 2 waves; SM active only 60-80% of elapsed.
- dev/gemv6.cu: same but W through a per-warp cp.async smem ring (U stages): m1 ~= cuBLAS (qkv 18.0, pooler 27.4).
  m>=16 much worse: x staging traffic = (n/RB) * m*K*2 (= W bytes at m32 with RB=32) and serialized x load.
- dev/gemv7.cu: gemv6 + only rows < m of x in smem (others ldmatrix a zero line), + optional "ilv" W layout (tile t
  of every grid warp adjacent = global sequential sweep). m1: qkv 17.9, o 8.9, pooler 26.3, unpooler 7.8 (cuBLAS
  18.2/9.2/28.0/8.7). Deeper rings (U 16-32, 128 KB/SM in flight) and the ilv layout: NO gain.
  ncu pooler m1 (2,4,16,cs1): DRAM 74% of peak while active, SM active 43k of 62k cycles (blocks finish unevenly).
- dev/gemv8.cu: no clusters, many HW-scheduled blocks, unit-contiguous W ([rg][K chunk][stripe][tile]), split K by
  red.global + last-block finalize (self-zeroing ws/cnt). Also ~cuBLAS at m1 (qkv 18.9, pooler 28.4, unpooler 8.1),
  m32 25.5 vs 18.9. Gotcha: cudaFuncSetAttribute(MaxDynamicSmem, 227 KB) FAILS once the kernel has static smem
  (later launches > 48 KB: "invalid argument"); set it to the size actually needed.
- Router: hi/lo native (gemv6/7, NA = 2: hi and lo tiles share the x fragment, one FP32 accumulator) for m 9-32;
  stacked [hi; lo] weight -> ONE cuBLAS GEMM (FP32 out, N = 512) + one add for m > 32 (was 2 GEMMs + add).
  H100: m16 12.7->8.4, m32 16.5->10.0, m128 19.0->11.3, m256 20.4->13.6.
- check c77613c2555140bf (25e165b): 1.1227x; submit fd765502babc4c69: 1.1251x (rank 831). H200 router:
  m1 5.38, m16 6.94 (SGLang 14.82), m32 8.44 (15.31), m8 7.86 (26.30), m128 10.61 (16.51), m256 11.51 (17.73).
- check c62f7f1ddc154626 (955bbc6, + BF16 m=1 on gemv7): 1.1235x (not submitted: not above 1.1251). H200 m1:
  qkv 13.29 (SGLang 13.97), o 6.66 (6.69), pooler 20.47 (21.10), unpooler 6.39 (6.82): the H100 ratios carry over
  1:1; there is NO extra H200 headroom for the same kernel.
- Conclusion: four GEMV structures all land at cuBLAS +-5% at m1, while the pure grid-stride reader is ~20% faster
  (H100). Next: bisect from the reader upward (see STATE).
