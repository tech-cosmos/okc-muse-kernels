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

## Session 3 (2026-10-25 20:30-21:45 UTC; job times: e5329f39 20:47, 3c377a1a 20:49, 1c6ae4b6 21:16, a55d6b55 21:18, 49298f3e 21:32, c569b693 21:34)
- Probe (/tmp, numbers here): empty graph 2.88 us; a nop kernel adds 1.8 us (1 block) / 2.6 us (1056 blocks); 2 nops +3.0.
  The ~4 us "intercept" in every shape's time is mostly launch, not DRAM ramp. Reader slope 3.24 TB/s (H100).
- dev/gemv9.cu (CUDA-core warp stream-K, W row-major, atomics+counter finalize): 25-40% SLOWER than cuBLAS. x per tile
  via __ldg or smem made no difference. Warp-contiguous streams are slow (rd2 mode0 too: 17 vs 14.6 us).
- dev/gemv10.cu (m=1): grid-stride like the fastest reader: chunk = U*4 KB = whole rows, virtual block (256 thr) v takes
  chunks v, v+G, ...; FP32 FMA, warp shuffle + smem per chunk, no split K, no atomics. Key: x staging traffic =
  blocks * k * 2 bytes (1056 blocks * 16 KB = 17 MB of L2 reads!) -> fat blocks (NV=4 groups of 256 share one x copy,
  grid 132). Also STACK spills at 40 regs were deadly (check cuobjdump -res-usage). H100 m1: qkv 15.9 (cuBLAS 18.2),
  o 8.6 (9.1), pooler 24.7 (28.3), unpooler 6.3 (9.1). CUDA-core at m=8: 28 us (compute-bound) -> tensor cores for m>1.
- check e5329f3904b840df (ec649e1 = d894f4e after rebase): 1.1602x. H200: qkv m1 11.61 (SGLang 14.14), o 6.55 (7.04),
  pooler 17.92 (21.68), unpooler 4.59 (7.01): H200 gains are LARGER than H100's for this kernel.
- submit 3c377a1a274a444f (same code): 1.1739x (CI 1.156-1.192). New best.
- Reader layouts (H100, /tmp/pj/rd3.py): block-contiguous ranges (block b reads [b*S, (b+1)*S), threads interleaved
  16 B) are within ~2% of grid-stride. red.global.add.v4.f32 costs ~1 us per ~70K ops (v4 as cheap as v1 per op).
- dev/gemv11.cu (m = 8..32, WIP): mma m16n8k16, x = A, W = B; W repacked to 512-B atoms (8 rows x 32 k, lane l: row l/4,
  k (l%4)*8..+8 = B frags of two k16 mmas; x A frags use the same k permutation); layout [ks][rg][atom][lane]; block =
  (k-chunk, run of rgs), x chunk staged in smem. Findings: (1) float atomicAdd on SHARED memory = CAS spin loop
  (ATOMS.CAST.SPIN): never use it for reductions; (2) x staging queued behind the W prologue takes 5-8 us (globaltimer
  per phase, /tmp/pj/dbg11.py); issuing no W before x lands: staging 1.5 us; (3) loop = W-bound ~2.9 TB/s; (4) split-K
  end phase (stcg partials, fence, counter, re-read) ~3 us serialized. qkv m8 best so far 19.7 (cuBLAS 18.1).
- dev/gemv12.cu (m=8): one warp group per 8-row group (block = NG rgs x WG warps, all concurrent, no unit loop), x
  chunk cp.async'd first, per-warp smem slots (WG-deep sums), S=1 where n/8 divides the grid. H100 vs cuBLAS:
  qkv m8 17.8 (18.0, S=4), pooler m8 26.3 (27.5, KC=K), o 9.2 (9.4), unpooler-1k 7.3 (9.3; 32 blocks x 512 best: x
  staging 64 KB/block). m16/m32: 5-25% slower than cuBLAS. Phase timers (qkv m8): staged 1.0, warp loop ends
  9.5-13.6 (1.8 spread inside a block), split-K end ~2.8 us. Late-W-prologue variant (G12_LATE_W): no change.
- cuBLAS swap-AB (W @ x^T) at m16-256: no faster than F.linear.
- Bug caught by bench test: gemv12 left partials in the router's self-zeroing st["ws"] -> router m8 failed. gemv12 has
  its own ws12/cnt12 now. Rule: every split-K kernel gets its own workspace.
- check 1c6ae4b67d9d4e46 (e71d015): 1.1772x; H200 verify1: qkv 13.36 (14.14), o 6.74 (6.85), pooler 19.06 (21.26),
  unpooler 5.71 (6.96). submit a55d6b55a2604db3: 1.1757x (CI 1.158-1.193). New best.
- gemv12 x pieces: TMA bulk (cp.async.bulk + mbarrier, one thread issuing M*XP copies) was SLOWER than cp.async
  (qkv m8 18.7 vs 17.9, unp1k 9.2 vs 7.1). cp.async pieces (commit group per piece, wait_group + barrier at piece
  boundaries) and an x ring (XR buffers over XP pieces, S=1 with full K): pooler m16 S=1 26.5 vs cuBLAS 27.8 (H100).
- Phase timers m16 qkv (S=4): loop ends ~same as m8 (W-bound ~2.9 TB/s); end phase 2.3-3.1 us = the loss.
- check 49298f3eb9994920 (39dccf9, gemv12 on all m16/m32 shapes, probe): 1.1401x. H200: m16 qkv 0.993, o 0.905,
  pooler 1.080, unpooler 0.946; m32 qkv 0.801, o 0.724, pooler 0.929, unpooler 0.852.
- check c569b69329a44aed (a0424fc, m8 + pooler m16): 1.1772x; pooler m16 19.92 (SGLang 21.57). unpooler m8 7.27
  (0.973x) although check 1c6ae4b6 had 5.71 (1.219x) with the same code: its config uses 32 blocks (32 SMs): fragile.
  Not submitted (expected gain over 1.1757 within noise; 1 submit left today).
- gemv12 R=2 (two rgs per warp sharing x frags): no gain at m16, spills at m32 (qkv m32 36.7 us). Reverted (v7 kernel).
- check b1b6c641c76a43dc (b8f0a3b, unpooler m8 on 64 blocks): 1.1729x; unpooler m8 7.20 (0.942x), o m8 0.958.
- check 9ddb59a548b24cc9 (385b40e, unpooler m8 back on cuBLAS): 1.1621x; even cuBLAS unpooler m8 = 7.46 vs SGLang's
  7.17 (same kernel!): that shape swings +-0.3 us per run, and whole runs of near-identical code give 1.162-1.177.
  Judge noise ~+-1%: single-shape gains < ~5% are not measurable in one check.
- main (after 385b40e) = a0424fc's code (m8 qkv/pool/o/unpooler-b32 + pooler m16 on gemv12), check c569b693 1.1772x.
  NOT submitted (best on board stays a55d6b55 1.1757x). Next session: submit it together with the next real gain.

## Session 4 (2026-10-25 21:45-23:25 UTC)
- Triton swap-AB split-K GEMM (/tmp prototype, TMA descriptors, last-CTA FP32 reduction): never beat cuBLAS. m16: end
  phase alone ~3 us (RED=0 variant 15.8 vs 18.6 qkv); m128/256 split-K with big tiles 1.5-3x slower. Dead end.
- cuBLASLt heuristic gives only 8 algos (all algo 66); split-K variants of them: +4-12% on some shapes. Full
  enumeration of algo 66 (634 tiles x 8 custom options x 7 clusters x 6 split-K; 45,792 pass AlgoCheck) screened in
  ~30 s/shape: H100 best vs F.linear: qkv m16 17.2/18.1, m32 17.5/18.4, m128 18.1/21.1, m256 19.8/21.2; pooler m16
  25.9/27.5, m32 26.1/27.8, m256 29.6/34.8; o -2-4%; unpooler ~0. NB attribute sizes: cluster shape and inner shape are
  uint16, the others 4 bytes (GetAttribute with wrong size silently returns nothing).
- Tuning in prepare() needs a READ flush (torch.sum over 256 MB), not zero_(): dirty lines from a write flush slow
  the next kernel and mis-rank candidates. Graph replays, 25 reps, median.
- check 1c72a94fe3b249ea (134eda6, lt at m16-256): 1.2139x. submit 202a248512c642ed: 1.2111x (CI 1.193-1.229).
- check 556e63a8ac264fb3 (42dac99, tuner also over gemv12 + lt at m8): 1.2268x. H200 tune: lt beat gemv12 at every
  m8 shape (pooler m8 22.2 vs g12 23.2, unpooler m8 9.8 vs 10.9 = lt split-K 8). submit 22109e72563f4cc3: 1.2334x
  (CI 1.215-1.252), proj rank 379. Per-shape in STATE.

## Session 5 (2026-10-25 22:22- UTC)
- Router m8 FP32 kernel: lane-0 serial atomics -> per-lane atomics: H100 m8 9.98 -> 9.50 (small). g7 (hi/lo) is faster
  at m8 (8.56) but SGLang keeps true FP32 at m<=8, so FP32 stays there. g7 at m128/256: 25-40 us (useless; cuBLAS 3-kernel
  path 12.9/13.9 H100).
- KEY FINDING: gemv10 at m=1 = a bare grid-stride reader (/tmp/pj/ic.py: 37.7 MB 15.5 us vs gemv10 qkv 15.7; 67 MB 24.8 vs
  24.8). An empty kernel costs 2.05 us over an empty graph (H100); reader intercept ~3.5 us, slope 3.16 TB/s. So m=1 is
  AT the bandwidth limit: the only way down is fewer bytes. Top teams (1.55-1.61x) must be compressing.
- Lossless 12-bit BF16 (TASK.md allows lossless compression): exponent entropy 2.54 bits (randn/sqrt(k) weights);
  window of 16 exponents below the max: escapes 0.014% (~1.15/row at k=8192), max per row well under 8.
  Format: sign|mantissa byte (s m6..m0) + 4-bit code c = e - base; decode a BF16 pair with prmt(sel 0x9180 / 0xB3A2,
  sign-replicate) & 0x807f807f | ((nibword shifted) & 0x07800780 | 0x38003800): exponent 112|c, sum rescaled by
  2^(base-112) (exact). Escapes: code 0, byte 0 (decodes to 2^(base-127)); header per row ESC=8 x (col, (true-D)/scale).
- dev/gemv13.cu (m=1, gemv10 structure, CUDA-core FMA): pitfalls hit: (1) x in registers (32 regs) + 1024 threads = 64-reg
  cap -> spills at PF=2; x as FP32 in smem instead; (2) x smem at 64-B lane stride = 4-way bank conflicts -> pad 16-float
  groups to 20; (3) per-chunk escape header loads cost ~1.1-1.4 us (!) whatever the form (cp.async+serial loop, per-lane
  regs) -> load ALL of the virtual block's headers into smem at kernel start, issued BEFORE the W prologue (with x).
  H100: qkv m1 15.7 -> 14.0, pooler 24.8 -> 20.5; o (k=1024) and unpooler-1k: gemv10 still better.
- dev/gemv14.cu (m=8, gemv12 + decode straight into B fragments, ~2 ops/weight, escapes fixed in the epilogue from staged
  x, needs XR == XP): H100 qkv m8 16.2 (g12 17.8, cuBLAS 18.0), pooler m8 23.5 (26.3/28.1), o 8.7 (8.9), unp1k 8.7 (9.0).
  m16/m32: gemv12's structure loses (split-K end phase/x staging): pooler m16 26.3 vs cuBLAS 28.3 but qkv m32 21.5 vs 18.2.
- Integration: prepare() packs (_compress/_pack13/_pack14) and the tuner picks among g10/g13 at m=1 (F.linear dropped at
  m=1: it won a tie by noise in 720c8278) and F.linear/g12/g14/lt at m8/m16.
- check 720c8278cc0e4e14 (3e807e3): 1.2371x; pooler m1 15.49 (was 17.77), qkv m1 11.27.
- check 416ae38fd88a441d (64487e7, loads reordered): 1.2435x. submit a6f4d701eb1e47f2: 1.2319x (noise; board keeps 1.2334).
- check fb5ef6d91d4342b3 (20f475d, + gemv14 at m8): 1.2439x; verify1: pooler 16.33 (18.76), qkv 13.02 (13.93); the total
  is noise-limited (other loads +-2%). H200 tune: g14 wins pooler m8 (20.3 vs lt 22.5) and qkv m8; lt keeps o/unp m8.
