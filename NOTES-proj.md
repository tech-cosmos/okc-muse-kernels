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
- check 416ae38fd88a441d (820d5cb, loads reordered): 1.2435x. submit a6f4d701eb1e47f2: 1.2319x (noise; board keeps 1.2334).
- check fb5ef6d91d4342b3 (cc055c8 = 20f475d before rebase, + gemv14 at m8): 1.2439x; verify1: pooler 16.33 (18.76), qkv 13.02 (13.93); the total
  is noise-limited (other loads +-2%). H200 tune: g14 wins pooler m8 (20.3 vs lt 22.5) and qkv m8; lt keeps o/unp m8.
- dev/gemv15.cu = gemv14 with the S k-chunk blocks of a row run as a thread-block cluster, partials reduced over DSMEM
  (no workspace/counters). H100 standalone: pooler m16 25.1 (cuBLAS 27.4), qkv m16 18.3 (18.1), unpooler m16 25.6
  (25.8); m32 31-47 (much worse: x A-fragment smem loads per 8-row atom, x staged up front per block). H200 tuner: lt
  22.05 vs g15 22.43 at pooler m16 -> not picked. Kept as a candidate (harmless).
- check 7ba50b0a348044ca (21bec6b): 1.2394x. submit f9cd875d5d5c4cbe (21bec6b): 1.2482x = NEW BEST (prev 1.2334).
- Per-load H200 (check 7ba50b0a): decode1 1.41, verify1 1.47, decode16 1.17, decode32 1.14, verify16 1.15, verify32 1.15.
  The m>=16 loads are the drag: their BF16 shapes are still cuBLASLt on uncompressed weights.

## Session 6 (2026-10-25 23:17 - 10-26 00:10 UTC): gemv16 for m16
- dev/gemv16.cu (+ dev/g16.py sweep, same CLI as g15.py plus a 7th field mc): gemv15 restructured: W = mma A operand
  (16 rows x k16; atom 16 rows x 32 k, lane = 16 B sign|mantissa + 8 B codes), x = B via ldmatrix.x4 (x row stride
  KC*2+16 B: conflict-free). Debug macros (env DEFS=G16_NOMMA,G16_NORED,G16_NOP2,G16_NOESC,G16_NOX,G16_SYNCONLY) skip phases.
- H100 breakdown, unpooler m16, k=4096, S=1 (no split-K), 128 blocks: pure W stream 19.8 us (= bare reader speed);
  warp skew inside a block ~0; reduction phase 1 (slot sum + escapes) 0.3 us; phase 2 (cluster.sync x2 + DSMEM loads
  + stores) ~2 us (!); x staging (128 blocks x 128 KB = 16.8 MB L2->SM) ~3 us; math (ldmatrix + mma) ~1.3 us.
- Fixes: (1) S=1 writes outputs straight from phase 1 (no cluster): 26.85 -> 24.7. (2) x staged by TMA bulk copies
  multicast to a cluster of mc=2 blocks with the same k-chunk (cp.async.bulk ... multicast::cluster + one mbarrier per
  block, cluster.sync after mbarrier.init): 24.7 -> 23.0-23.3 (cuBLAS 25.5). mc=4 (or S=2 x mc=2) = clusters of 4: only
  ~30 fit at 1 block/SM, 128 blocks -> 2 waves (33 us). (3) S>1: partials PUSHED into the owner block's smem
  (st via map_shared_rank) + ONE cluster barrier, instead of sync / remote loads / sync: pooler m16 S=2 27.2 -> 25.2.
  (4) conflict-free accumulator layout [warp][nt][e][lane] + escape loop stops at the first empty slot (value 0).
- Dead/neutral: deeper prefetch (PF 8/16 at 512 threads, 128 regs): no gain; separate accumulators per k-step: neutral
  (kept); issuing the W prologue before the cluster barrier / x multicast: worse (23.0 -> 23.6: x latency is critical);
  x in pieces (XP 2/4): no gain; S=4/8 splits: much worse (cluster fit + reduction); gemv15 with the conflict-free
  reduction: no change (25.18), reverted.
- H100 standalone bests: unpooler m16 23.0 (S1 mc2, 16,4,4,4,128,1,2) vs cuBLAS 25.5; qkv m16 16.9 (S2 mc2 bk48:
  16,4,4,4,48,1,2) vs 18.1; pooler m16 25.2 (S2 mc1) vs 27.5 (= gemv15). m32: S=1 impossible (x 256 KB), S2 needs mc2
  -> clusters of 4; not explored.
- H200 tuner (judge log): unpooler m16 g16 21.18-21.98 vs lt 21.44-21.76 (picked in 1 of 2 runs); qkv m16 g16 17.47 vs
  lt 16.77; pooler m16 g16 23.26 vs lt 22.40. H200 gains are SMALLER than H100 here (opposite of gemv14 at m8).
- check ae0e03fa12b24ae5 (46b45bb, unpooler g16 not picked = same kernels as the board run): 1.2595x (noise-high; not
  submitted). check 863c78fcf9174248 (4c99cac, g16 picked at unpooler m16: 17.42 us vs 17.69): 1.2565x.
- submit cf54d7712be64c9e (4c99cac, 10-26 00:01Z): 1.2480x (CI 1.229-1.267): tie with the board's 1.2482 (g16 picked at
  unpooler m16, 21.44 vs lt 21.44 in tune).

## Session 7 (2026-10-26 00:05 - UTC): exact 12-bit headers, router2
- Human 23:30Z fix done (3f173eb): escape header value = true / scale (BF16 over a power of two: exact in FP32); empty
  slots = 2^-15 (col 0); kernels add x * (v - 2^-15) as fmaf(x, v, fmaf(x, -2^-15, s)). Real escapes have |v| < 2^-15
  (e < base), so gemv16's "stop at first empty slot" test is now v == 2^-15. _compress decodes every weight (bytes +
  codes, escapes from the header) and compares bits; any mismatch -> None (that layer stays uncompressed).
- Exactness detail: with a single fma x * (v - 2^-15) the subtraction rounds for v far below the window (spike/probe
  rows came out ~5e-18 relative off). Two fmas on the row's FULL sum are exact when only that weight contributes
  (s = 8*2^-15 -> 0 -> 8v). gemv13 had the fixup on a partial sum in another warp: moved it to the row's reducing
  thread (after the cross-warp sum). No measurable cost (H100 qkv m1 13.7 vs 14.0, pooler m8 23.7-23.9 vs 23.7).
- bench.py esc: injects zeros, -0, BF16 denormals, 2^-100, values just below the window into 32 fixed columns
  (2 per row at k=8192 where natural escapes reach 6/row, 3 per row else), checks all four layers compress, outputs pass,
  and spike rows at those columns are bit-exact vs the FP64 reference (m = 1/8/16/32).
- dev/g13-16.py sweep scripts updated to the new header format.
- check 659bbeab27fd4b61 (3f173eb): 1.2515x, passed (same kernels as board otherwise).
- router2 (dev/router2.cu, dev/r2.py): router m<=8 without split K. Block = R=2 whole rows (k=8192), 512 threads,
  thread owns k = 4 (t + 512 j), j<4: x slice loaded straight into registers (BF16 pairs), 8 float4 W loads up front,
  shuffle + one smem pass. H100: m1 6.9 -> 5.4 us, m8 9.8 -> 6.5 us. Configs: m1 R1/T1024 5.5, R2/T512 5.4,
  R4/T512 6.5, R1/T256 5.6; m8 R2/T512 6.5, R2/T256 6.75, R1/T512 8.4. At m16 it loses (10.8 vs hi/lo g7 9.0), m32 far.
- Dead: g7 (hi/lo mma) with token groups at router m128/256: 24-66 us vs cuBLAS 3-kernel path 11.8/13.6 (H100).
  The cuBLAS path on H200 = splitK GEMM 5.2 + splitKreduce 1.6 + hi+lo add 1.4 us (m128).
- check dd37685bc1b440b7 (e605717, router2): 1.2853x; H200 router m1 5.13 -> 3.47, m8 8.02 -> 4.92.
  submit d8160f04086d4406 (e605717): 1.2636x (new best then; check/submit gap = judge noise, same content hash).
- gemv17 (dev/gemv17.cu, dev/g17.py): router2's structure for BF16 (uint4 W/x loads, k = 8 (t + T j)). H100: unpooler-1k
  m1 (1,4,512) 5.66 / (1,4,256) 5.76 vs 6.24; m8 (8,4,256) 7.65 vs 8.67; o (n=8192, k=1024) loses at m1 (9.3 vs 8.8)
  and m8 (13.8 vs 9.1): too few threads per row / too many blocks. Tuner candidate (BF16_G17) at unpooler m1/m8.
- check 1be6c8f5278b41eb (405439e): 1.2926x; H200 unpooler m1 4.67 -> 3.75 (tune: g17 7.20 vs g10 7.81 incl. events),
  m8 6.50 -> 5.99 (g17 9.28 vs lt 9.63). submit 6d0fc4d42b4544b1 (405439e): 1.2742x = NEW BEST.
- Budget 10-26 at session end: checks 3/20, scored 3/10.

## Session 8 (2026-10-26 01:21 - 02:10 UTC): router3 (router m9-128 in one kernel)
- dev/router3.cu (+ dev/r3.py sweep: `python dev/r3.py "MT,NT,WM,WN,KC,NS;..." m...`): split-K mma.sync GEMM, x = A
  (MT tokens), W hi/lo rows = B (NT experts), hi and lo MMAs share the x fragments into one FP32 accumulator; cp.async
  ring of 64-k stages, 128-B rows XOR-swizzled for ldmatrix. Grid = (K/KC, 256/NT, m/MT) ~ 128 blocks.
- Split-K reduction, measured (H100, m128, MT128 NT64 KC256): body alone ~6.5-7 us; partial tile -> smem -> float4
  atomics ~+1.5-2.4 us (red throughput ~1M floats / 2.4 us: m256 with 2M floats pays ~4 us); persistent ws + per-tile
  counter + last-block copy: fence/counter +2.3 us, copy +2 us (serial ld->st loop; batching the loads saved ~1.3).
  Winner: a tiny zero kernel for out (griddepcontrol.launch_dependents at its start) + the GEMM as its PDL dependent
  (griddepcontrol.wait just before the atomics): no ws, no counter, no copy. m16 8.3 -> 6.9, m128 12.7 -> 10.1.
- Dead: clusters of 4/8 along split K with a DSMEM pre-reduction (8-block clusters don't fit at 1 block/SM: 2 waves,
  14 -> 21 us); clusters of 2 (+0.5-1 us); rotating the atomic start row per split (no change: not contention);
  smaller tiles to cut red volume (L2->SM load traffic grows: (64,32,KC1024) 14 us at m128).
- H100 (bench.py): m16 6.94 (was 8.9 g7), m32 7.04 (10.3), m128 9.78 (12.2 cuBLAS 3 kernels), m256 12.7 = cuBLAS (kept).
  m256 limits: red volume (2M floats ~4 us) and mma.sync compute (~3.5 us for 2.15 GFLOP).
- check a478daeaa3d64100 (a1f0af7): 1.3137x; H200 router m16 7.38 -> 5.55, m32 8.66 -> 6.36, m128 10.24 -> 9.00 us.
  submit 71145e34fe424844 (a1f0af7): 1.307x = NEW BEST (proj rank 196; prev 1.2742).
- a566ba9: ROUTER_R3_TUNE: prepare() times router3 configs vs cuBLAS at m256 (_tune_router / _graph_medians). H200
  (check 1a9781226b444d11, 3,698.5 us = tie): cuBLAS 14.21 vs r3 15.10-15.26 (incl. events) -> cuBLAS; not submitted.
- H100 m256 sweep (no win): (256,32,4,2,512) 12.9, (256,32,8,1,512) 13.2, (256,64,8,2,256,3) 14.1, KC 1024 / NT64 KC512
  18-20 (64 blocks). Budget 10-26 at session end: checks 5/20, scored 4/10.

## Session 9 (2026-10-26 02:06 - 03:30 UTC): gemv18 = TMA-ring streamer for the layer's m16/m32 GEMMs (dev only)
- Lead 01:45Z: proj now serves the layer (T=16/32 qkv/o/pooler/unpool are cuBLAS there, ~60 us at d32). Built
  dev/gemv18.cu (+ dev/g18.py sweep "M,KC,AP,NS;...", env SH, GRID, XWARM=1 (x written by a kernel right before, as in
  the layer), DEFS=G18_NOFIN/NOMMA/NOESC/XFIRST/TRACE; dev/g18t.py M,KC,AP,NS <layer> = per-block globaltimer timeline).
  Design: 12-bit weights repacked (g18.py pack18) so each stage (ks, 128-row tile, AP atoms x 8 row groups) is ONE
  contiguous block -> one cp.async.bulk per stage from a producer warp into an NS-deep smem ring; 8 consumer warps (one
  16-row group each, W = mma A decoded in regs, x = B by ldmatrix from x[:, KC chunk] in smem); stream-K over the
  flattened (ks, tile, stage) space (balance within one stage); segments covering all of k write BF16 directly, else
  partials -> smem -> a finisher warp (red.global.add.v4 into FP32 ws, fence, per-row-group atom counter, last arriver
  converts + re-zeroes). Correct incl. probes/escapes/self-clean (3 reps each).
- Lessons (H100, timelines):
  * 16 small bulk copies per stage (2 KB / 1 KB) cost ~75 ns each: AP=1 was 2-3x slower. One copy per stage fixed it.
  * Bare stream (DEFS=G18_NOFLUSH, old layout): qkv m16 13.2, pool 20.6, unp 20.5 us (cuBLAS 18.1/27.7/25.8) = ~3 TB/s
    slope. Ring of ~36-48 KB in flight per SM already saturates (A2N3, A4N2).
  * In-consumer flush (fence + atomic + last pass) stalled all 8 warps at once (same segment boundary): +8..+25 us.
    Moved to a finisher warp; mid-stream fence+atomic still takes 5-7 us under load (fine, off the critical path), but the
    END of the kernel pays the chain red -> fence -> atomic -> load -> store (~4 us tail; was ~9 us with per-row-group
    serial loads, batched now).
  * Escape headers loaded at segment start = cold HBM loads under full load: stalls short segments. Now loaded one
    segment ahead (first at kernel start). Must not __shfl after a divergent break (hung the GPU once).
  * x[:, chunk] lands 3-5.6 us after start when cold (all blocks read the same 64 KB chunk; W streams queue ahead).
    Waiting for x before streaming W (XFIRST): slightly worse. Warm x (XWARM) saves ~1 us.
- Numbers (H100 graph replay, us; cuBLAS in brackets), config 16,2048,4,3 / 32,1024,4,3:
  full kernel, x warm:   qkv m16 19.5 (17.0) m32 20.9 (18.1); pool 26.2 (27.6) 28.7 (27.9); unp 27.0 (25.3) 30.5 (25.7)
  NOFIN (reds, no completion), x warm: qkv 15.8 (17.2) 17.6 (18.3); pool 24.2 (27.9) 25.0 (28.3); unp 23.6 (25.5) 25.3 (26.0)
  -> the GEMM part beats cuBLAS everywhere (pooler -13%); the in-kernel completion (~3.5-5 us) eats it.
- Next: (1) completion OUT of the kernel: standalone = PDL-chained convert+zero kernel (ws n x m FP32 -> BF16, L2-hot);
  in the LAYER = fuse "round FP32 ws to BF16 + re-zero" into the next op (qkv -> attention prep, o/unpool -> boundary,
  pooler -> rmsnorm_gain): numerics identical (BF16 rounding on load). (2) x startup: split x into pieces with per-piece
  barriers in consumption order, or PDL (producer streams W before griddepcontrol.wait; xload after). (3) H200 check
  via the tuner (BF16_G18) once (1) is in: H200 gains of compressed kernels differ from H100 (gemv14 bigger, gemv16 smaller).
- No judge runs this session (budget 10-26: checks 5/20, scored 4/10).

## Session 10 (2026-10-26 03:12 - ~04:30 UTC): gemv19 = FP32-out PDL 12-bit GEMM for the layer's T=16/32
- gemv18 F32 mode (pdl=2: FP32 out by red.add, zero kernel as PDL primary, no counters): H100 warm x pool m16 24.5,
  qkv m16 16.8 (cuBLAS 27.0/17.7); m32/o weak. Stream analysis (/tmp/pj/tmab.py-style microbench, now gone: TMA ring
  only, consumers just read smem): a 50 MB TMA stream = 21 us graph on H100 for ANY stage size 12-48 KB / depth 3-12;
  slope 3.1 TB/s + ~4 us intercept. gemv18's MEDIAN block ends with the pure stream's median; its excess is the spread
  (blocks end 21-27 us vs 23-25.5 for the stream). NOMMA = same time: memory-bound, not compute.
- Dead: deeper rings (NS 5-12: worse), x before W (XPRE: x lands 3.5 instead of 6 us, total same), interleaved stage
  layout (block b's stage i at i*G+b: same), dynamic tail (static ranges + atomic-grabbed items of IS stages: producer
  ends tighten but every tail stage pays a flush: +2-4 us), per-SM calibration (per-SM end times don't correlate between
  runs: r ~0).
- The lever: PDL. gemv19 (dev/gemv19.cu, sweep dev/g19.py "M,KC,AP,NS,IS,TAILPB;..." env MODE/PRED/NPRE/XWARM/ESCW/CVT):
  producer streams the first npre W stages BEFORE griddepcontrol.wait (W never depends on the previous kernel), then
  x; FP32 out; mode 2 zeroes out in-kernel (finisher warp zeroes a slice, then epoch-counter grid barrier: atomicAdd
  ctr[1], wait until ctr >= old - old % G + G, never reset; plain full-k stores disabled in mode 2). Producer tags each
  ring slot (u | x buffer | run end), x double-buffered, finisher driven by pbuf tags.
- H100, after a 3 us predecessor kernel that triggers launch_dependents at start and writes x (PRED=3000), mode 2:
  m16 qkv (1024,4,4) 11.8 (cuBLAS 17.1), o (512,4,3) 7.3 (8.5), pool (1024,4,4) 20.3-20.8 (27.5), unp 20.7 (25.2);
  m32 qkv (512,4,3) 16.3 (18.2), pool (512,4,4) 24.0 (28.2), unp (512,4,4) 23.5 (25.7), o 9.1+ (8.5: keep cuBLAS).
  WITHOUT a predecessor (judge standalone): ~cuBLAS at m16 (pool 24.8 vs 28.0, qkv 16.1 vs 18.2 with npre 2; npre 0
  worse), loses at m32 (pool 27.9 vs 28.1, unp 28.0 vs 26.0). + g19_cvt (BF16 copy, PDL) ~ +1 us.
- red.global.add.f32 flushes subnormal results to zero: spike rows at escapes with |8 w| < 2^-126 come out 0 (accumulator
  behaviour, the 12-bit decode is still exact; tests compare against FTZ'd expectations). ESCW=1 (bench esc injection,
  40 reps per config, all four layers, m16/m32): pass, rel err <= 4e-7 vs FP64.
- dev/g19_lib.py = layer interface (pack19 from _compress output, build, call doc + numbers); dev/t19lib.py tests it.
  Handoff posted in ~/COORD.md (proj s10).
- Submission (b812fe4 -> this session): BF16_G19 tuner candidates (+ cvt) and _diag19 (print only: 3 us PDL predecessor +
  g19 vs the pick, per shape). H100 standalone: never picked. diag (pick vs g19): qkv m16 15.4/11.5, m32 16.5/16.0,
  pool m16 24.1/20.2, m32 25.9/24.6, unp m16 21.3/20.5, m32 25.1/22.9, o m16 7.9/6.9.
- s10 later: xb (1 x buffer released at its last use), ts (2 warps per row group: dead, +5 us: decode doubled), rg 16 /
  rpw 2 (dead, no gain on H100), grouped smem loads (m32 -0.7 us). No-W-load mode (G19_NOW) + timelines: per stage the
  consumers run at ~6 TB/s-equivalent; the fixed costs are x arrival (~2 us to the first finished stage) and ~1 us per
  segment flush (escape fixups + pbuf handoff) + tail.
- Checks (diag only, never picked standalone; prepare prints "diag19"): 9b09c536 (6996d0c) 1.3136, 60ddb481 (966241b)
  1.3296, f13197773 (cfd1faf) 1.3146. H200 after a 3 us PDL predecessor (g19 vs best pick): m16 pooler (1024,8,3,xb1)
  15.8 vs 18.9, unp 16.5 vs 17.5, qkv (1024,4,4,xb2) 11.2 vs 13.2 [AP8 12.4], o +0.7 worse; m32 all worse (+0.8..2.8).
  Bare TMA stream 50 MB: H200 13.7, H100 18.3 us. Budget 10-26 at session end: checks 9/20, scored 4/10.

## Session 11 (2026-10-26 04:24 - ~05:45 UTC): fused pooler + router + routing for the layer at T=1 / T=8
- Lane: lead 01:45Z (layer's GEMMs as they run in the layer). Layer's own Next 1 (pooler+router+top-8 in one kernel at
  T=1/8); I claimed it (COORD Requests 04:30Z), layer acknowledged (r-layer 24aac63) and stays on g19 at d16.
- Measured in a COPY of r-layer (/tmp/pj11/layer, layer.py patched, `python build.py`, `python bench.py test`,
  tools/timeline.py, tools/route_judge.py, dev/layer_ab.py = interleaved one-process A/B of module-attribute variants).
- pr1 (dev/pr1.cu, test dev/pr1.py): T=1. Blocks [0, rb) = FP32 router (4 rows per group, x32 in registers, next group
  bulk-prefetched to L2, W of the first group loaded before griddepcontrol.wait), blocks [rb, 132) = gemv13 pooler body
  (W prologue + escape headers before the wait, x after). Last router block (atomic ticket) does the routing in one warp
  (sigmoid via expf, top-8 of score + bias, lowest index on ties, weights, local ids). Launch = PDL secondary of
  _boundary, which must trigger at its start (without the trigger: +2 us vs today's side-stream router).
  H100 layer (5-round A/B): d1 sliding 95.1 -> 92.2-92.7, full 117.4 -> 114.7-115.4. rb 22-30, npf 0-8, R 2/4, PF 2: flat.
  Timeline: pr1 runs ~21 us after the boundary end for 58 MB (pooler 50 + router 8) = HBM-bound; old: router 5.3 us
  delayed the pooler start by 3.3 us.
- pr8 (dev/pr8.cu, test dev/pr8.py, TIME=1 / TRACE=1 = globaltimer per block): T=8. gemv14 pooler config is S=1 with
  128 x 1024-thread blocks (one per SM), so no SMs for a router role: every block also does one router item (16 rows x
  1024 k of the FP32 router; W in registers before the wait; x32 slice cp.async'd to smem first), partial logits
  [8 slices][8][256] to scratch, red.release.gpu counter; an EXTRA finisher block (blockIdx 128) spins (ld.acquire +
  nanosleep), sums the slices in order (deterministic), routes the 8 tokens (warp = token), resets the counter.
  * Lesson: dependent memory round trips while HBM is saturated cost ~3-4 us EACH. First version had per-row-group and
    global tickets inside the pooler blocks: the global-last block took 17.7 us for its ticket chain + routing (+12 us
    to the kernel). Moved to the finisher block: full = no-ticket time.
  * H100 layer: v1 sliding 158.1 -> 157.0, full 172.4 -> 171.0 (small: 58 MB at ~2.4 TB/s; the boundary also slows,
    its latency-bound loads queue behind pr8's pre-wait traffic; router loads after the wait: same; no trigger: worse).
- Both: tests 13/13 (identical errors to before: outputs unchanged), route_judge 25 x 3 seeds 0 fails at d1 and v1.
  Handoff in COORD (proj s11 05:15Z), wiring = dev/pr1_layer.patch (vs r-layer 8ac162e).
- gemv13 VT (virtual-block threads 256/512): qkv m1 has 2304 rows = 4.36 rounds of 528 vbs -> 5 (13% tail); VT 512 =
  half-size rounds (9 of 8.73). H100 13.7 -> 13.1 us, but H200 (check 6b50e844, 1.3064x, diag) 14.21 vs 14.14: no gain.
  Kept as a tuner candidate (g13 call takes vt now).
- Checks 10-26: 10/20 used (6b50e844). Scored 4/10.

## Session 12 (2026-10-26 05:03 - UTC): gemv20 = g19 with wgmma consumers; why g19/g20 are consumer-bound
- dev/gemv20.cu (namespace g20, same args as g19 + an SW128 x tensor map encoded per launch), sweep dev/g20.py (same
  env as g19.py; PROF=1 = plain launches for ncu; DEFS=A=B values ok). Consumers: 8 warps = 2 warpgroups; warp w =
  16-row group w (dec8 output = wgmma A fragment, unchanged packing); per k16 one wgmma m64nMk16 per WG, B = x chunk
  as [KC/64][M][128 B] SW128 (TMA box {64, M}); A fragments double-buffered (wg_wait<1>); escape fixups read the
  swizzled x. Correct first try (g20.py asserts incl. probe rows).
- H100 full (MODE 2, PRED 3000): ties g19 (pool m16 20.45 vs 20.3-21.5, m32 KC512 24.8 vs 24.1). But KC 1024 at m32
  (x 64 KB, xb 1, fits) is a real gain over KC 512: pool 22.1 vs 24.6, unp 21.5 vs 24.3, qkv 15.4 vs 16.7 (g19 likely
  the same: the layer/handoff used KC 512 at m32).
- Consumer-only (G19_NOW/G20_NOW, no W loads): g20 is NOT faster than g19 (pool m32 KC512 24.3 vs 23.3, m16 19.8 vs
  17.7). Ablations on g20 NOW pool m16 KC1024 AP4: full 20.1, no wgmma 16.7, no decode 15.5, neither 13.1, neither +
  no wgmma fence/commit/wait 12.1, no red/no fixups ~-0.5. Skeleton vs AP (stages per block): AP2 16.1-16.8, AP4
  12.0-12.4, AP8 11.6-11.8 => ~0.28 us FIXED per stage per warp (mbarrier wait, tag, index math, lds, arrive) and
  decode + mma add serially on top: 0.86 us/stage at AP4 m32 vs H200 HBM ~0.58 us/stage (18.4 KB per SM at
  ~32 GB/s/SM) => consumer-bound on H200, as the s10 judge diag said. ncu: issue slots 27% busy, 2.44 warps/scheduler.
  wgmma does not fix it; the fix is more independent consumer streams per SM (latency hiding).
- 2 blocks per SM (G20_MINB=2, G20_SB single-buffered A, G20_NPB=1 partial buffer; grid 264; 89-93 regs; needs
  cudaFuncAttributePreferredSharedMemoryCarveout 100, else 1 block/SM fits and the grid barrier HANGS: g20 now
  TORCH_CHECKs co-residency via cudaOccupancyMaxActiveBlocksPerMultiprocessor): NOW pool m16 19.94 -> 16.67 us
  (consumer throughput +20%); H100 full: ties at m16 (pool 21.1, qkv 13.1-13.8), loses at m32 (needs KC 512 to fit:
  pool 22.8, unp 23.5, qkv 17.0 vs 1-block KC1024 22.1/21.5/15.4). H200 decides: _diag20 in prepare().
- H200 judge diags (my board, print-only `_diag20`, after a 3 us PDL predecessor; us over the predecessor):
  check 2ef3aafd (a90ffda, 1.2953x) and 88800611 (e9cd9b5, 1.2934x; adds variant c = g19 2 blocks/SM):
  | shape | g19 1 blk (BF16_G19 cfg) | g20a 1 blk | g20b 2 blk | g19 2 blk (c) |
  | qkv m16 | 12.51-12.67 | 13.86 (1024,4,4) | **11.58-12.00** (1024,4,2) | 11.90 |
  | qkv m32 | 15.49-15.84 | **15.04-15.14** (1024,4,4) | 15.30-15.42 (512,4,2) | 16.54 |
  | pooler m16 | **15.94-16.10** | 20.03-20.35 | 17.31-17.44 | 17.15 |
  | pooler m32 | **20.70-20.77** | 21.73 | 20.96-21.09 | 22.50 |
  | unpooler m16 | **16.74-16.96** | 19.94-20.00 | 17.12-17.31 | 17.38 |
  | unpooler m32 | **19.97-20.26** | 21.60 | 20.61-20.77 | 22.05 |
  | o m16 | **7.26-7.46** | 8.00-8.32 | 7.62-8.00 | 8.48 |
  g20 (wgmma) at the SAME config is 4 us slower than g19 at pooler m16 on the H200 (no ptxas serialization warning,
  no spills): 8 dependent wgmma m64n16/32k16 per stage into one accumulator vs g19's independent mma.sync chains.
  2 blocks/SM: the consumer gain is eaten by the smaller ring/x/partial buffers (KC 512, NS 2, NPB 1).
  => DEAD for the layer: wgmma consumers, 2 blocks/SM (both kernels). Only qkv m16/m32 gain, and qkv is the
  layer graph's first op (no PDL predecessor to overlap: the diag's case does not apply). g19 stays as handed off.
- Budget 10-26: checks 12/20 (2ef3aafd, 88800611 diag only), scored 4/10. No submits (my board is a test bed).

## Session 13 (2026-10-26 06:14 - ~07:20 UTC): lead item (5), T<=8 GEMV ramps. QG overlap + gemv21: both DEAD
- Idea (QG): qkv_gate = q 1024 | k 128 | v 128 | gate 1024 rows; the gate is only used in csl's final merge, and csl
  (decode 1 sliding attention) runs on 32 CTAs (two 16-CTA clusters, 128 threads, 79-88 regs, 22 KB smem) while ~100
  SMs idle. Plan: qkv GEMV counts finished q/k/v rows (qg[0]) and gate rows (qg[1]) with red.release; csl launches as
  its PDL secondary, gathers the window's KV first, spins (ld.acquire) on qg[0] before its q/k/v loads (ld.cg) and on
  qg[1] before the gate in the merge; csl's last CTA (atomic ticket) resets qg. Implemented in a copy of r-layer
  (patch: muse_bf16_proj/dev/qg_layer.patch = diff vs r-layer layer.py/build.py at 4afa8ea): tests 13/13, same errors.
- Co-residency needs BOTH: register room (g13 is 60 regs x 1024 threads = the whole RF; csl needs 88 x 128) AND the
  smem carveout (cudaFuncAttributePreferredSharedMemoryCarveout 100 on g13; without it csl CTAs only start when g13
  blocks exit, even with registers free). With g13 on 768-thread blocks (64 regs) + carveout, csl starts at 2 us
  (beside the GEMV) instead of 13 us.
- But g13 needs its full RF to be fast: 768 threads 16.1-16.6 us, 1024 x __maxnreg__(48) 15.5, vs 13.2 (H100 qkv m1;
  bytes in flight per SM = registers). H100 layer A/B (5 rounds, d1 sliding): baseline 93.1, QG single g13 (768 thr)
  95.1, QG split (q/k/v rows on the fast config, gate rows on a 768-thread launch with csl beside it) 99.6. DEAD.
- gemv21 (dev/gemv21.cu, dev/g21.py): m=1 GEMV over gemv13's packing with the weights in a TMA bulk ring (one row
  per stage: bytes 8 KB + nibbles 4 KB + header 64 B), NG consumer groups of WG warps (x in registers), a producer
  warp; few registers, so csl would fit beside it. Correct first try (probe columns exact through escapes).
  * v1 (consumer-side finalize via smem ticket, static rows): H100 qkv 14.9 (g13 13.8; pure ring = consumers do
    nothing: 12.7). H200 (check 96165c87, tuner us): qkv 17.7-18.3 vs g13 14.0, pooler 26.2 vs 18.7, unp1k 11.2 vs g17
    7.3. DEAD on the H200.
  * Trace (globaltimer, ~256 ns ticks on H100: use clock64 within an SM): per-CTA end times spread 11.0-14.9 us with
    static rows (slow SMs); per-row ready->stored latency ~0.8 us.
  * Dynamic rows by ONE global ticket counter (never reset: a call takes exactly n + TQ*grid tickets, row = ticket mod
    that) are throttled by same-address atomics: 132 CTAs bumping one counter serialize at ~0.5 us per round.
    Static rows + tickets only for the last DT rows: spread fixed, but still slower overall.
  * Lessons: an mbarrier.arrive.release (TMA expect_tx) waits for the SAME thread's pending global atomics/stores:
    keep ticket atomics and output stores on other lanes than the TMA-issuing lane. A shifted register queue of pending
    atomics (q[i] = q[i+1]) waits for every atomic: unroll by the queue depth. try_wait suspend hints: no effect.
  * v2 (producer finalizes rows lazily): worse on H100 (17.0). Kept in the tree as reference; BF16_G21 = {} (not tuned).
- Launch-completion edges (dev/lc_edge.py): during capture, cudaStreamUpdateCaptureDependencies with edge data
  {from_port = cudaGraphKernelNodePortLaunchCompletion (2) or Programmatic (1), type = Programmatic} after a cuBLAS
  GEMM lets the next kernel (which must griddepcontrol.wait) launch early. H100: -0.8..1.0 us per edge at m16/32/128,
  but port 2 = port 1, and port 1 is what launching the next kernel with PDL already gives (layer: "PDL on glue after
  cuBLAS: no gain" on the judge). Nothing new.
- PDL hoisting audit (after layer v42 found ptxas hoisting __ldg above griddepcontrol.wait in g17): SASS of the layer's
  pr1/pr8/g19 builds has no nc load of primary-written data before ACQBULK (pr1's 8 pre-wait LDG.128.CONSTANT are
  the PR1_BND prologue's s/alpha/beta/gain inputs). Made pr1.cu robust anyway: x32 and x after the wait by __ldcg
  (dev/pr1.py OK). pr8: x32 by cp.async, ls by __ldg (input, not written by the primary): fine.
- Checks 10-26: 13/20 (96165c87 1.3091x, diag only). Scored 4/10.

## Session 14 (2026-10-26 07:08 - 08:30 UTC): gemv19 run-end cost = escape fixups; ballot fixup (handoff)
- Lead item (5), T>=16 part. Tools: dev/g19.py PROF=1 (plain launches for ncu; ncu at /usr/local/cuda-13.0/bin/ncu:
  hold /tmp/okc-gpu.lock around the ncu run, but then the script must not time (bench.time_graph re-takes the lock:
  deadlock)), G19_TRACE now also stamps: consumer after griddepcontrol.wait (9), first x arrival (5), stage-0 compute
  end (6), stage-1 end (8), first run end: start (11) / after fixups (12) / handed to finisher (14), finisher first
  two buffers done (2, 3). Diag defines: G19_OLDFIX (old fixup loop), G19_NOFIX (skip fixups).
- H100, pool m32 (1024,8,2,xb1), consumer-only (G19_NOW, NPRE=0): stage ~1.15-1.3 us each, but every run end (SPK = 4
  stages) added ~1.06 us; NOMMA (no decode/mma) has the SAME 1.06 us run end, NOFIX 0.1 us => the escape fixup loop:
  16 slots x (2 shfl -> compare -> divergent branch -> 8 LDS -> fma chain), fully serialized (SASS: BSSY/BSYNC per
  slot, shfls not hoisted). Fix: 4 ballots (owner lanes test their own 2 slots per h), OR-fold over the 8 row lanes,
  then a warp-uniform `continue` for empty slots. Same math/order: bit-identical. Run end 1.06 -> ~0.7 us.
  Full kernel (MODE 2, PRED 3000, H100, 2 interleaved rounds): pool m16 21.3 -> 20.5, m32 23.7 -> 22.95, unp m32
  23.6 -> 23.0, unp m16 22.1 -> 22.0. ESCW 12/12, tests 39/39. Patch for r-layer: dev/g19_ballot_layer.patch.
- Not smem-bound: RG16 RPW2 (one x fragment feeds 2 row groups: half the ldmatrix traffic) gives the same per-stage
  time (NOW 1.15 us). NOX (decode, no x/mma) 0.9 us/stage, NOMMA 0.5 us/stage: decode ~0.4 and mma+ldsm ~0.4 add up.
  ncu (NOW): issue 0.31/scheduler, 2.5 warps/scheduler, top stalls = first-stage and x barrier waits.
- Startup: with the pre-wait W prologue (npre 2 stages x 49 KB x 132 blocks = 13 MB), the x bulk loads issued after
  griddepcontrol.wait queue behind it: x arrives at ~6 us (NPRE=0: ~2.6 us with a cold L2). In the layer this is
  hidden only when the PDL predecessor runs longer than the prologue (~4 us on H100).
- Next for g19: (a) fixups in the finisher warp (consumers write a compacted escape list per warp next to pbuf; the
  finisher applies the same two fmas to pbuf before its red pass; x lifetime: finisher arrives on xempty too) ->
  consumer run end ~0.1 us; (b) cap npre so x is not queued behind W when the predecessor is short; (c) x-arrival
  2.6 us cold: 132 blocks hit the same 7 x chunks at once (L2 hot spot?): try TMA multicast over clusters of 2.

## Session 15 (2026-10-26 07:40 - 08:35 UTC): gemv19 escape fixups off the register path (s14 Next (a))
- Real escape density (pack16, KC 1024): qkv 1.15/row, pooler 1.11, unpooler 0.40, o 0.10 -> per 128-row tile per
  1024-k chunk ~13-18 escapes; per consumer warp per run ~2-3.
- Tried (a) as planned: consumers push a compacted escape list per warp (ballot + popc, cheap), the finisher applies
  them on pbuf before its red pass and releases x (mbarrier arrive count NCW). Correct (ESCW, ECAP=2 overflow
  fallback) but SLOWER by 1-3 us (H100 MODE 2 PRED 3000): bisected (G19_NOLIST / G19_NOAPPLY): the finisher's
  serial apply loop (~18 entries per tile, one warp, LDS->fma->STS chain ~150 cycles each) sits on the kernel's
  critical path (x release + last run). Unrolled/prefetched loads: no change. Kept as /tmp only (dead).
- Tried (b): each lane RMWs its own escapes in pbuf over all M tokens (8 rounds by slot): 3.7 us per run end
  (trace): one lane walks 32 tokens serially and the compiler can't overlap the RMWs (pointer aliasing). Dead.
- (c) = kept: after the pbuf store, per warp, escapes walked one at a time (ballot per (rr, h, slot parity), ffs
  loop, shfl col/hy from the owner) with lane = token: one parallel RMW per escape. G19_RFIX = s14 register path.
  H100 (MODE 2 PRED 3000, 3 interleaved rounds, median, vs G19_RFIX = s14): pool m16 (108 blk) 21.58 -> 21.44, pool
  m32 24.26 -> 23.26, qkv m16 11.60 -> 11.42, qkv m32 16.93 -> 16.34, unp m16 20.85 -> 20.51, unp m32 22.56 -> 22.27.
  ESCW ok in modes 0/2, TS2, RPW2. Ported to r-layer's (older) g19_src.cu: dev/g19_s15_layer.patch (vs fbc5684).
  Copy of r-layer + patch: bench.py test 13/13; tools/ab_files.py 5 rounds: d16 sliding 215.6 -> 213.8, full 318.9 ->
  314.3; + g19 at T=32 (pooler (32,1024,8,2,2,1) grid 108, unpool same): d32 sliding 210.3 -> 205.0, full 439.4 -> 433.1.
- Check ce84da07 (b17feac, 1.3001x, diag only), H200 g19 after a 3 us PDL predecessor vs the tuned pick (us):
  m16 pooler 15.81 vs 19.20, unpooler 15.62 vs 17.34 (layer cfg 4,4: 15.94), qkv (8,3) 11.14 / (4,4) 10.62 vs 13.25,
  o 5.98 vs 6.40; m32 pooler 19.71 vs 19.84 (tie), unpooler 18.40 vs 19.01, qkv 15.39 / 14.75 vs 13.66 (loses).
  s10 had all m32 losing by 1.5-5: the fixup rework moved m32 to tie/-0.6.
- Layer request (s15): qkv at T=16/32 (graph's first op, no predecessor) with BF16 out. Built gemv19 mode 3: BF16 out
  ob, FP32 workspace that stays zero (finisher reds, __threadfence, atomicAdd(atoms) on a never-reset per-tile counter,
  the one reaching a multiple of KA bulk-copies the tile to smem (fence.proxy.async first), writes BF16, re-zeroes),
  runtime tm = tile-major stage order (pack perm (1,0,3,2,4)). g19.py MODE=3 TM=1 tests (40 reps ESCW, ws clean).
  * first version (tile read back by __ldcg loops): +5..15 us: each dependent L2 round trip ~1 us under load.
    Bulk copy: m16 22.1 -> 19.2. Tile-major: 18.4 (H100 cuBLAS 18.1, mode 0 FP32 + zero kernel 17.0). npre 0/1: same.
  * H200 (check 4e5fad5b, 1c90a55, 1.2951x, diag only, standalone graphs incl. launch): m16 (4,4) 19.62 vs lt 16.90,
    m32 23.33 vs lt 17.54. DEAD: g19's win is the PDL overlap; with no predecessor it loses ~3-6 us (x arrival +
    startup). Mode 3 kept in the kernel (cheap, tested) in case a predecessor appears; DIAG19Q = {} now.
- Checks 10-26: 14/20 (ce84da07, 4e5fad5b diag). Scored 4/10. Board best unchanged (71145e34 1.307x).

## Session 16 (2026-10-26 08:37 - UTC): gemv22 (m128/256 12-bit wgmma GEMM): DEAD; qkv m8 without split-K: WIN
- gemv22 (dev/gemv22.cu, dev/g22.py "M,AP,NS[,E]" env CL/SH/DEFS): swap-AB wgmma m64nMk16 (M = 128/256 tokens as N),
  A = 12-bit W decoded into registers (gemv19/20 atom packing, rg 8), B = x box by SW128 tensor map; W stage (12 KB) and
  the stage's x box share one ring slot; tile-major stages, tile-aligned split (each tile gets floor/ceil(G/ntr) blocks:
  one run per block), 3 warpgroups (setmaxnreg 40/232). Lessons, in order found (H100 traces, G22_TRACE stamps):
  * wgmma in a data-dependent branch -> ptxas C7518 "serialized ... divergent path": consumers must run straight-line
    stage pairs (static A/B sets) with runs computed from the static range; 384 threads, warp id via __shfl_sync.
  * escape fixups that read x from global at run end: each dependent round trip under saturated HBM ~2-3 us. Final form:
    escapes packed INTO each W stage (E = 32 entries x 8 B: row-in-tile | k-in-stage << 8, true/scale), one fixup warp
    follows the ring and red.adds x (from the slot's x box) * (hy - 2^-15) * scale into out. No extra round trips.
    Rounding: the fixup is a separate FP32 red (the decoded 2^-15 term cancels in a different add): probe rows exact to
    ~1e-17 abs instead of bit-exact (ordinary FP32 accumulation; bench.py esc tolerates it at m >= 128 only).
  * scalar red flush 4.3 (m128) / 9.6 us (m256); v4 reds through a [tok][16 rows] smem transpose: ~2 us per run end.
  * blocks straddling 2 tiles flush twice (tail): tile-aligned split fixed it (qkv m128 22.3 -> 20.8 H100 = cuBLAS).
  * THE LIMIT: per-SM smem ingest. NOMMA and NOX each leave the loop as slow; W-only stream ~2.9 TB/s on H100. Per stage
    W 12 KB + x 16 KB (m128) / 32 KB (m256) -> ~47-50 GB/s per SM (moe s16 found ~60 for a bare ring). x/W bytes = 1.33
    (m128) / 2.67 (m256) at BM 128. Cluster-2 x multicast halves L2 reads but NOT smem ingest: no gain (after fixing
    remote arrives: mbarrier.arrive.release.cluster was 2x slower than the default .release.cta form CUTLASS uses).
  * H200 (check fb2fe4ee, 1.3071x, diag): tuner us g22 vs best lt: qkv m128 22.1/18.8, o 17.9/10.6, pooler 29.6/24.4,
    unp 30.7/23.0; m256 qkv 30.9/20.7, o 26.1/12.4, pool 41.3/28.7, unp 46.2/27.2. BM 256 would halve x/W but double the
    flush volume (132 x BM x M x 4 B of reds) -> ~tie at best. Dead; BF16_G22 = {} (BF16_G22_S16 keeps the configs).
- gemv13 cp.async ring (G13_CPA, smem ring of PF chunks): H100 qkv m1 13.4 vs U2V4 14.4, but = the existing VT 512 config
  (13.6): the gain is the round-count tail, not bytes in flight. L2 bulk prefetch of the vb's chunks (G13_L2PF): +1..3 us
  on H100. gemv13 for o (k 1024): loses to gemv10 (11.3 vs 11.0 tuner H100: 2 rounds per vb, latency-bound). All off.
- cuBLASLt picks the layer doesn't use (Handoffs proj s16 10:10Z): pooler T=256 (66,18,35,1,0,0,0,2) -4.3..7 us H200.
- **qkv m8 without split-K**: gemv14 (8, 8, 8, 4, 96, 2) = kcu 8 (S = 1), 96 blocks x 768 threads: no partials/counter
  end phase (~2.5 us at S=4). H100 g14.py 15.2-15.6 vs 16.8; H200 check a9b13469 (1.3161x): 15.39 vs 16.58 tuner us.
  Submit 3810db09 1.3110x = new board best (was 1.307). Handoff to layer (G14_CFG qkv). m16 gemv14 kcu 4 / 96 blocks: tie.
- g19 m32 in the layer (layer request s16 09:30Z: d32 pooler+unpool 50 us on H200). H100 consumer-only (G19_NOW, MODE 2,
  PRED 3000, pool m32 (1024,8,2) 132 blocks): full 16.45 us, NOX (decode only) 12.06, NOMMA (W lds + barriers) 9.76;
  m16 13.3 / 11.0 / 8.0. On the H200 the W stream needs ~13.9 us for 50 MB -> the consumers bound it (as s12 found).
  mma+ldmatrix = +4.4 us (0.57 us per 49 KB stage = mma.sync at ~2/3 of the wgmma rate), decode +2.3, skeleton 9.8
  (trace: producer->consumer handoff ~0.5-0.6 us per stage with NS 2 even with no W loads). More warps are slower
  in NOW mode (RG 16 20.4-21.0, TS 2 21.4), RPW 2 17.2 (not smem-bound). No cheap fix found; wgmma consumers = g20 (dead
  s12). In the layer the pooler runs on 108 blocks (router beside it): consumer-bound -> 132/108 slower per call.
- H100 layer timeline d32 sliding (r-layer 66fef4c copy): boundary 51.0-57.6, g19 pooler 51.6-83.7, _router_split
  57.5-76.4 (18.8 us on its ~24 SMs), routing 76.3-81.1, rmsnorm 83.7-86.3; unpool 181.9-211.6 after norm2 (190.1).
- g19 x staging (H100 traces, NOW+NOMMA, after the 3 us pred): x chunk arrives ~2.1 us after griddepcontrol.wait at m32
  KC 1024 (32 row copies x 2 KB), 1.7 us at KC 512 (32 x 1 KB), 1.5 us at m16 (16 x 2 KB): latency ~1 us + bytes +
  copy count. G19_XPAR (producer warp in lockstep, x rows issued one per lane): x arrival unchanged, times -0.1..0.5 us
  (noise level). G19_XQ (x in 4 quarters with one mbarrier each, stage waits only its quarter; 4x more 512 B copies):
  correct but SLOWER (pool m32 27.4 vs 22.8, qkv m32 20.4 vs 15.0): small bulk copies are expensive. A 2D tensor-map
  x box (SW128, 16 boxes per chunk) + swizzled ldmatrix addressing would cut copies; est. <= 1 us per call. Both off.
- KC 2048 at m16 (AP 8, NS 2, xb 1): H100 108 blocks after pred -0.75..0.85 us (pool/unp), consumer-only +0.5: likely
  no H200 gain (consumer-bound there). m32: KC 1024 stays best (512: +1..2, 2048: +6).
- **H200 PROXY for g19** (moe s16's trick): H100 with GRID=88 and times x 0.73 reproduces the judge's g19 numbers after
  a 3 us PDL pred (ce84da07): pooler m16 (8,3) 15.55 vs H200 15.81, unp m16 16.08 vs 15.62, unp m32 17.88 vs 18.40,
  pooler m32 18.14 vs 19.71 (orderings right); qkv m32 off by 3 (12.15 vs 15.39). `GRID=88 MODE=2 PRED=3000 python
  dev/g19.py ...` then x 0.73 (H200 108 blocks ~ GRID 72). NOT valid for gemv13 (orders VT 512 last; H200 first).
  Proxy verdicts: KC 2048 m16 16.1 vs 15.0 (loses: retracted the tune suggestion); G19_XPAR 0; G19_MMANV (mma asm not
  volatile) 0; fused [pooler; router 512 rows] at 132-equiv (G88) m16 18.0 / m32 20.1 vs pooler alone at 108-equiv
  (G72) 16.15 / 20.81 -> with routing then on the critical path: loses; G19_NOXW (consumers never wait for x: upper
  bound of faster x staging) -0.2..0.7 us only.
- Session 16 end (11:00Z): checks 10-26 16/20, scored 5/10. Board best 3810db09 1.3110x.

## Session 17 (2026-10-26 11:00 - UTC): m=1 GEMVs are at bare-stream speed on the H200 (diag check 29a2b16b)
- Question (layer s17: "m=1 GEMVs run 2.5-3.2 TB/s on H200, ~11 us of d1 room"): is gemv13 short of bytes in flight on
  the H200 (it holds 48 KB/SM in registers: U2 x 24 B x 1024 threads)? H100 g13.py XF=-DG13_CPA (cp.async smem ring,
  PF 3/4/6, NV 2/4, VT 512): all tie or lose (qkv 13.0-16.0 vs 13.3, pooler 20.5-23.3 vs 20.2). (NV 4 x VT 512 = 2048
  threads: invalid launch.)
- **H200 check 29a2b16b (77e925e, 1.3173x, diag; not submitted: no new pick)**, graph us incl. the judge's ~4.6 us
  1-kernel-graph cost: pooler m1 g13 18.53 vs g13c 18.69-22.02; qkv m1 g13 (VT 512) 13.98 vs g13c 14.30-15.74.
  Bare TMA stream (g19_stream, (24576 B, 4 stages), 132 blocks, 1-kernel graph subtracted): 50 MB 13.89 us, 28 MB
  8.51 us. So pooler g13 = 13.9 + 4.6 = exactly a bare stream; qkv g13 = bare + 0.9 us. The profile's "2.5-3.2 TB/s"
  is kernel duration incl. ramp/drain (a bare 28 MB stream itself is 3.3 TB/s). DEAD: no m=1 kernel lever left;
  only fewer bytes or overlap with latency-bound phases (layer tried o-W L2 prefetch, pr1 npf: ties).
- Fewer bytes: exponent entropy of all 4 BF16 weights = 2.54-2.55 bits (uniform-like: each octave halves; qkv/pooler
  exps 121..115 = 15/32/24/14/7/3.5/1.8%). Ideal lossless ~10.5 bits vs our 12 (-12%); a 3-bit window covers only
  98.6% (1.4% escapes: too many for per-row headers) -> needs a 2-level code (3-bit code with an escape plane indexed by
  popc of a per-group mask). Est. d1 ~-1.5..2 us/layer (qkv, pooler, unpool; o stays BF16 g10) = ~0.3-0.4% layer score,
  m>=8 kernels are issue-bound and would not gain. Big change (format in g13 + pr1 + layer packers). Not started.
- Answered layer's g22 m128 cfg question from fb2fe4ee (COORD Requests): both cfgs equal, both lose 3-8 us.
- Checks 10-26: 17/20, scored 5/10. Board best unchanged 3810db09 1.3110x.

## Session 18 (2026-10-26 ~11:25 - UTC): g19 consumer ILP experiments (layer d16/d32 pooler/unpool)
- No answer from the lead on the s16/s17 lane question; stayed on the layer's d32 g19 item (pool/unp m32 ~2.1 TB/s in the layer).
- G19_SWP (software-pipelined groups: group gi+1's lds/ldmatrix issued before group gi's decode+mma, G 1/2/4):
  H100 proxy (GRID 88/72, MODE 2, PRED 3000) and consumer-only (G19_NOW) = base within noise (pool m32 G72 28.29 vs
  28.3-29.1). The consumer's latency is not the smem load chain. Removed.
- 11-bit format re-estimate: a 3-bit window + escapes needs ~1.4-2.5% escapes per weight; per-row escape lists eat
  most of the 1-bit saving (~3% net), group-offset escape planes ~7% but need prefix counts + a second stream in g13.
  Not worth it for ~0.3% layer score. Router FP32 weights have full mantissas (no lossless halving).
- Layer v16/v32 GEMMs (profile ea9421fa): cuBLASLt at T=128 standalone ~qkv 14.2 / o 6.0 / pool 19.8 / unp 18.4 us
  kernel (pool/unp ~3.4-3.6 TB/s already); room mostly qkv/o, ~6 us at v16 if a kernel reached 3.6 TB/s (~0.3%).
- New tool: dev/g19ab.py = in-process interleaved A/B of two gemv19 builds (DEFS_A / DEFS_B) after a PDL predecessor.
- G19_ILV2 (decode of fragment f+1 issued between fragment f's mmas; also volatile prmt, MMANV): consumer-only G72
  pool m32 base 25.73, ILV2=1 25.22, ILV2=0 25.81, VPRMT 25.76, MMANV 26.42. Full-mode interleaved A/B (dev/g19ab.py,
  G72, 7 rounds, B-A us): qkv m32 +0.03 m16 -0.06, pool m32 +0.05 m16 +0.18, unp m32 -0.06 m16 -0.21 => DEAD (ptxas
  already schedules decode beside the mmas as well as it can). Removed.
- Conclusion: g19's consumer cost is not an instruction-order problem (SWP, ILV, MMANV, SPIN all null). What's left is
  the instruction count itself (mma.sync rate + decode) = a different MMA path (wgmma, failed s12) or fewer bytes.
  No proj item left above ~0.3% layer score; lane question to the lead stands.

## Session 19 (2026-10-26 11:56 - 12:25 UTC): review only, no build
- Still no lead answer to the lane question (Requests s18).
- Layer H200 profile ea9421fa (v45), proj-owned kernels: d16 g19 qkv+unpool (16,1024,4,4) 37.5 us for 2 calls, pooler
  g19 20.7 (incl. PDL wait), o cuBLAS nvjet 7.1, qkv cast 1.3; d32 g19 pool+unp 50.2, qkv lt splitK 12.7 + reduce 2.0,
  o nvjet 7.0. Every alternative proj has for these (g19 qkv/o at T=16/32, lt picks, LT_ALT) is already a layer TUNE
  option, and the H200 tuner settles them in the layer's ~14:30Z check: nothing new to hand off.
- qkv cast at d16: mode 3 (BF16 written in-kernel) ~= mode 0 + cvt on H100 (18.4 vs 17.0 + ~1, s15), so no proj kernel
  fix. Posted a layer request: attention reads g19's FP32 qkv and rounds to BF16 when it loads it (bit-identical), est. -1.2 us d16.
- Checks 10-26: 17/20, scored 5/10. Board best unchanged 3810db09 1.3110x.
