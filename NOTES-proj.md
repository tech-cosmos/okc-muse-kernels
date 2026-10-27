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

## Session 20 (2026-10-26 12:15 - 12:50 UTC): two quick dead ends for the layer's d32 qkv, no build
- Still no lead answer to the lane question (Requests s18). The layer did my s19 request (v67 QF32 at T=16; s29 WIP
  extends it to T=32 = QF32_NDS).
- Idea 1: the layer's d32 qkv = cuBLASLt splitK 12.7 + splitKreduce 2.0 us (H200). With FP32 out (attention reads FP32
  via QF32) could split-K go in-kernel (reduction scheme INPLACE)? NO: H100 /tmp/pj/s20/kc.py + heur.py: every algo-66
  split-K config launches cublasLt::splitKreduce_kernel for FP32 out too, for reduction schemes 1/2/4 alike, and
  the heuristic's other algos for BF16 and FP32 out (m16/m32) are all split-K + reduce. FP32-out splitK=1 = BF16-out
  splitK=1 (H100 graph us m32 17.70 vs 17.54, m16 17.54 vs 17.50). Dead. Script: /tmp/pj/s20/lt32.py (lte with OUTT).
- Idea 2: g19 qkv at T=32 as the graph's first op, now that QF32 drops the cast: H100 MODE 2, no predecessor, layer cfg
  (32,1024,8,2,2,xb1): 21.84 vs cuBLAS 18.69 us. Even -1.2 us of cast can't close a 3 us gap (H200 s15: mode 3 23.3 vs
  lt 17.5; mode 0 after a 3 us pred 14.75 vs 13.66). Told the layer the qkv T=32 g19 TUNE option should not win, so
  QF32_NDS only pays if the H200 tuner picks it.
- Checks 10-26: 17/20, scored 5/10 (none used this session). Board best unchanged 3810db09 1.3110x.

## Session 21 (2026-10-26 12:30 - 12:50 UTC): answered the layer's s29 lt32f ask from existing H200 data, no build
- Layer asked for an lt_gemm writing FP32 [32, 2304] without split-K (its v68 QF32_NDS reads FP32 qkv at T=32).
- H200 data (every proj judge log, 12 runs, tuner us, cold L2): qkv m32 splitK=1 configs (66,314/80/12,...) 18.3-18.9
  vs best split-K (66,27 or 16,35,3,2,...) incl. splitKreduce 16.96-17.54. H100 s20: FP32-out splitK=1 = BF16-out
  splitK=1 (17.70 vs 17.54). So lt32f = +1.0-1.5 us at d32: told the layer (Requests s21) not to wire it.
- No judge runs. Checks 10-26: 17/20, scored 5/10. Board best unchanged 3810db09 1.3110x.

## Session 22 (2026-10-26 12:46 - UTC): QSS = the layer's d32 qkv as 2 FP32 bmm slabs, attention sums them at load
- Idea: the layer's d32 qkv is cuBLASLt split-K + splitKreduce (H200 12.7 + 2.0 us), and attention already reads FP32
  qkv at T=32 (layer v68 QF32_NDS). One strided batched GEMM over k-halves (torch.bmm out_dtype FP32, the layer's
  pooler-slab trick) has 2x the CTAs of splitK=1 and needs no reduce kernel; attention adds the 2 slabs in FP32 and
  rounds to BF16 at each load (= cuBLAS split-K reduce + BF16 output, just a different k split).
- H100 standalone (graph, cold L2; /tmp/pj/s22/slab.py): qkv m32 bmm2 24.9 vs every lt config 26.6-27.3 us; o: loses.
- **H200 diag check ccf41026 (4762be7, 1.3162x, print-only `_diag_slab`; not submitted, no pick changes)**, tuner us
  incl. ~4.6 graph: qkv m32 lt pick 17.50, mm FP32-out 18.11, bmm2 16.03, bmm4 16.16, bmm8 16.19; qkv m16 lt 17.22,
  bmm2 15.90 / bmm8 15.84 (layer d16 uses g19 instead); o m32 lt 10.24 vs bmm2 10.66; pooler m32 22.59 vs bmm4 22.27.
- Layer patch muse_bf16_proj/dev/qss_layer.patch (vs r-layer 20b864e; `git apply --check` passes on its working
  tree): attn_src.py nds tag type qs2 (qld8/qld4 add the slabs; new int qss arg, module renamed muse_nds_a2s10q2),
  Gluon _gd_full(_body) and Triton _combine/_combine_hb/_combine_bal/_red_fin get QSS constexpr (_qld/_gqld/_gqldm
  helpers); kernel_fn falls back to slabs.sum(0) on every other path (csl/vtk/nvs/verify/generic). layer.py: QSS_NS 2,
  QSS_DEFAULT {32: True}, TUNE[32] += {"qss": False}, {"qss": False, "lt:qkv_gate": True}; data["qkv_ss"/"qkv_slabs"].
  New tools/qss_eq.py: attention on slabs == attention on their FP32 sum, bit-exact (cache + out; full layers nondeterministic at ctrl level): bad 0.
- H100 layer A/B (tools/ab_files.py, 12 rounds, TUNE = {} in both so the per-prepare tuner can't pick differently):
  d32 sliding 204.98 -> 203.44, full 433.95 -> 432.42 us (-1.5 each). With the tuner on, A/B is dominated by tuner
  pick noise (one run said sliding +2.3): pin options for A/Bs. Timeline (tools/timeline.py): attention ends 1.9 us
  earlier; attention itself +0.3 (nds/combine) .. +0.9 (full) us for the 2nd slab loads.
- Tests on the patched build: bench.py test 13/13, replay_test decode 32 x 20 inputs x 3 router seeds: 0 fails.
- Session 22 end: checks 10-26 18/20, scored 5/10. Board best unchanged 3810db09 1.3110x. DIAG_SLAB off again (c5cf31c).

## Session 23 (2026-10-26 13:01 - ~13:45 UTC): gemv13 (12-bit) for o at m=1: DEAD on the H200
- Layer had not answered QSS (Handoffs proj s22) yet; its last check is ~14:30Z. No lane pick from the lead.
- o (8192 x 1024, 16.8 MB) is the only big BF16 weight still uncompressed at m=1 (gemv10). H100 dev/g13.py SH=o:
  U2 V2 B2 P1 on 264 blocks 8.16-8.38 us vs gemv10 (U2 V4 B2 g264) 8.53-8.58; U2 V1 B4 g528 8.54; U2 V4 g132 8.67;
  PF 2 / U4 / B8 worse. H100 tuner: g13 10.66 vs g10 11.07.
- Check 2722e8c495294b9e (97ec405) 1.3108x, DIAG: H200 tuner o m1 g10 9.70, g13 (U2V2B2 g264) 10.05, (U2V1B4 g528)
  10.11: loses. As s17 found for qkv/pooler, the H200 m=1 GEMVs are not byte-bound at these sizes; o is latency-bound.
  Candidates removed again (comment kept in sub_template.py); submission.py = 2480c53 code. Not submitted.
- Same log: o m8 lt 9.76 (g14 10.43-10.50, g12 10.11); unpooler m1 g17 7.17 (g13 8.35).

## Session 24 (2026-10-26 13:12-13:25Z): QSS verified on layer v69; G19_RAW (plain BF16 g19) DEAD
- QSS (dev/qss_layer.patch) was not in layer v69 (7d4f3ee). Applied it to a /tmp copy of v69: `git apply` clean, bench
  test 13/13, tools/scored_test.py "decode 32" x3 bad 0, tools/replay_test.py 20 11,12 "decode 32" 0/20 each layer.
  tools/qss_eq.py printed bad 2-3 at batch 2 full: the full attention path is itself nondeterministic there (6 identical
  controls differ among themselves; the QSS output equals one of them; /tmp/qss_det.py), not a QSS bug. Posted a
  Requests line so the layer bundles it into its ~14:30Z last check (it is a TUNE option, H200 can revert).
- G19_RAW (DEFS=G19_RAW in dev/g19.py; gemv19.cu WB = 1024): stages hold plain BF16 weights already in mma A-fragment
  order ([atom][s][lane][hk][h][e], pack19raw in g19.py), no decode, no escape headers/fixups, scale 1. Correct (g19.py
  asserts). Idea: g19 at m32 is consumer-bound on H200 (decode ~0.4 us/stage), so trade 33% more bytes for no decode.
  H100 proxy (GRID=88 MODE=2 PRED=3000, us; x0.73 ~ H200):
    m32 cfg (kc,ap,ns,xb1)   12-bit  RAW   | consumer-only (G19_NOW) 12-bit RAW
    pool 1024,8,2            25.18   27.52 |                         24.02  21.66
    pool 1024,4,3            25.84   27.33 |                         25.14  21.49
    unp  1024,8,2 / 4,3      24.90/25.02  28.48/26.94
    qkv  1024,4,3            17.06   17.52 (cuBLAS 17.3-17.5)
  RAW consumers are ~2.5-3.6 us faster but the stream then binds (27.3 vs its NOW 21.5): loses 1.5-3.5 us. DEAD.
  Also: even the RAW consumer-only time (21.5) is far above the bare stream (~18.3 H100 / 13.7 H200 for 50 MB): the
  per-stage fixed skeleton is the floor of the g19 design at m32, not decode alone.

## Session 25 (2026-10-26 13:20-13:50Z): judge-image GEMM libraries, H200 diag (check 5abb71f1a9c74bed, bf4689c, 1.3040x)
- `_diag_ext` (prepare, print only) lists the image's GEMM entry points and times DeepGEMM `bf16_gemm_nt(x, w, out)` vs
  our tuner pick (graph replay, cold L2, us incl. ~4.6 graph cost). Versions: deep_gemm 0.2.1.post1+muse.cbe1727,
  flashinfer 0.7.0.post1, sgl_kernel 0.4.9.
- pick / deep_gemm: qkv m16 16.70/17.66, m32 18.02/17.98, m128 18.98/20.51, m256 20.70/22.94; o m128 10.72/10.53,
  m256 12.45/12.42; pooler m32 22.18/24.80, m128 25.44/28.70, m256 28.35/33.89; unpool m128 22.88/23.01. DEAD (ties at
  best, -0.2 us on o; not worth a layer option). First call 0.0 s (no JIT visible).
- Untested candidates seen (for tomorrow's budget): flashinfer `tinygemm_bf16`, `trtllm_low_latency_gemm` (+
  `prepare_low_latency_gemm_weights`), `mm_M1_16_K7168_N*_bf16` (fixed shapes, not ours), `mm_bf16`/`bmm_bf16` (likely
  SM100 only), `routergemm`; deep_gemm also has `cublaslt_gemm_nt` and `bf16_mega_moe`/`bf16_mega_gate` (MoE). tinygemm /
  low-latency gemm at m8/m16 vs g14/lt is the only one plausibly worth a diag.
- DIAG_EXT = {} again (only the import/name print stays). Board 1.3110x unchanged. Budget 10-26: checks 20/20.

## Session 26 (2026-10-26 13:27-13:40Z): flashinfer tinygemm_bf16 (tinygemm2) ported and A/B'd on the H100
- trtllm_low_latency_gemm (flashinfer 0.7.0) = FP8 + Blackwell only: dead. tinygemm_bf16 = TRT-LLM tinygemm2, SM90 path:
  `pip download flashinfer-python==0.7.0 --no-deps` (no install), csrc/tinygemm2.cu ported to load_inline in dev/tinygemm2.cu
  (TVM FFI -> torch binding tg2; needs -lcuda for cuTensorMapEncodeTiled). Design: n/16 blocks x 384 threads (4 mma.sync
  consumer warps split K, 4 W TMA warps, 4 x TMA warps), STAGES x 4 x (16x64 W + 8x64 x) tiles; stock STAGES 16 = 196 KB smem.
  PDL mode: only x warps wait (W streams during the predecessor).
- Standalone H100 (dev/tg2.py, cold L2 graphs, us; current = submission pick): unp1k m8 6.53 vs 7.26 (g17) WIN, m4 6.59 vs
  8.45; unp1k m1 6.98 vs 5.57; qkv m1/m8 17.8/18.0 vs 13.3/16.0; o 12.8/13.4 (S4 9.1/9.25) vs 8.6/9.0; pool 26.5 vs 20.2/23.5;
  every m>=16 2-2.3x slower. S4/S8 never beat the current pick standalone.
- Tuner candidate BF16_TG2 = {"unpooler": [(8,)]} (separate guarded load_inline "muse_proj_tg2a"); local tuner picks it
  (9.25 vs g17 9.86 tuner us); bench verify-1 unpooler 6.62 us; tests 39/39. Not judge-checked (checks 20/20 today).
- After a 3 us PDL predecessor (g19_pred; dev/tg2pdl.py, dev/tg2o.py): tg2 PDL unp1k m8 +2.3..2.8 vs g17 +6.0..6.5 / tg2
  no-PDL +5.2..5.7; o m8 S8 +6.0 vs g14 +7.4..7.7; o m1 S8 +6.4 vs g10 +7.2; unp1k m1 S16 +2.7 vs g17 +4.2. Handoff + layer
  request posted (COORD proj s26). Layer's own o/unpool paths may already be PDL secondaries / fused n2u: it A/Bs.

## Session 27 (2026-10-26 13:34-14:05Z): TG2 for the layer's o at T=1/T=8: DEAD in the layer harness (H100)
- No judge runs (proj checks 20/20 today; layer budget spent too). Layer had not tried TG2, so I did it in a /tmp copy of
  r-layer v70 (909010f): built submission.py, then patched it: after `_g19(o, "o")`, if TG2_O[(T, sliding)] = S:
  tg2s(o, W_o, out, use_pdl=True, S) (W warps stream before griddepcontrol.wait, x after). bench.py test 13/13 with
  TG2 on at T=1 and T=8 (sliding + full).
- tools/ab_flag.py, 5 interleaved rounds, medians us (layer's pick = gemv10 PDL after csl at T=1; g12/g14 at T=8):
    shape              current   tg2 S4   S8      S16
    decode 1 sliding    90.42    92.67   93.81   94.45
    decode 1 full      112.58   113.79  115.57  116.00
    verify 1 sliding   152.29   153.47  153.90  155.71
    verify 1 full      167.34   168.03  169.12  171.01
  Loses +0.7..4 us everywhere: in the layer the predecessor is attention (busy SMs + smem), so tg2's 50-200 KB/block
  can't become resident early; my 3 us PDL predecessor (32 tiny blocks) was too kind. Unpool slice at T=1/8 is already
  fused (n2u / nu8 with W in smem before the wait), which is the same overlap idea: not tried. TG2 stays only as the
  proj-board tuner candidate at unpooler m8 (3addab4, not judge-checked).
- Lesson: PDL-overlap microbenchmarks must use a predecessor that occupies SMs/smem like the real one (csl/nds/tk2f).

## Session 28 (2026-10-26 13:52-14:30Z): gemv23 (BF16 SS-wgmma GEMM, m128/256): DEAD on H100 and in the H200 proxy
- Why tried: lead's option (a); the verify-16/32 GEMMs are the largest gap to floor left (H200 lt qkv m128 18.8 tuner-us vs
  8.4 floor; cuBLASLt picks 40x64 / 64x64 tiles without split-K, so W and x are re-read across tiles).
- dev/gemv23.cu + dev/g23.py "M,BW,NS[,OB[,CL[,GRID]]]": swap-AB SS wgmma m64nMk16 (A = W stage [BW rows][64 k], B = x box
  by SW128 tensor map), W repacked tile-major + pre-swizzled (pack23: chunk c of row r at c ^ r%8) so each stage is ONE bulk
  copy; tile-aligned split-K; BW 128 = 2 WGs x 64 rows, BW 64 = 2 WGs split the k16 steps (smem sum); out FP32 by
  red.add.v4 (zero kernel PDL primary) or direct BF16 when one block per tile; CL 2/3 = cluster split-K with DSMEM sum
  into rank 0 then BF16 store. All configs pass vs float64 reference (rel err < 1e-5 FP32, < 4e-3 BF16).
- Lesson 1: W through a 2-D tensor map (box 64 k x BW rows, rows 16 KB apart) streams at only ~2.3 TB/s on H100 (W-only
  diag 28.6-30 us for 67 MB); contiguous packed bulk stages fixed nothing measurable here (end-to-end same), so the
  W-only skeleton itself is ~4 us over a bare stream.
- H100 (us, cold-L2 graphs; F.linear / bench lt pick): best g23 per shape: qkv m128 22.1 (BW64 c2) vs 20.5 / 18.2; o m128
  11.55 (BW64 direct) vs 9.9 / 9.5; pool m128 30.0 (BW64 c2) vs 27.4 / 27.1; unp m128 27.9 (BW64 direct) vs 26.8 / 26.3;
  m256 all +3..20 us. BW 128 (x/W 1) is SLOWER than BW 64 everywhere (unp c2 31.6 vs direct 27.9): not ingest-bound.
- H200 proxy (GRID 88, x0.73): qkv m128 22.75 -> 16.6 vs judge lt ~15.9; pool m128 37.2 -> 27.1 vs ~21.4. Dead.
- submission.py unchanged. No judge runs (proj checks 20/20 today).

## Session 29 (2026-10-26 14:13-14:50Z): M=8 gemv19 for the layer's T=8 qkv (layer request s15): DEAD
- No new lane from the lead yet; checks for 10-26 all spent. Built the item from the layer's list: gemv19 at M=8 (one
  mma n8 token group; NT 1: one ldmatrix.x4 loads both k16 halves of x; dev/gemv19.cu static_assert relaxed, other M
  unchanged). Correct (g19.py probe checks, MODE 2).
- H100 standalone qkv m8 (graph, cold L2, us): g19 MODE 2 best 1024,4,4,2,0 r8 16.61 (r4 17.04, kc512 17.5, AP2 17.9+,
  NS6 17.6; MODE 0 + tail 16.83); g14 (current pick) KC8W8P4 X1 b96x768 15.44 / X2 15.74; cuBLAS 18.2-18.3.
  g19 loses ~1.2 us and qkv is the graph's first op (no PDL predecessor to hide g19's x arrival) -> dead.
- Correction to the layer's s15 premise: gemv14 is NOT CUDA-core FMAs, it is gemv12's mma.sync structure + 12-bit decode.
  On the H200 g14 qkv m8 is ~2.3 us over a bare 28 MB stream (tuner 15.39 vs 13.05): <= ~0.2% layer score of room.

## Session 30 (2026-10-26 14:19-14:40Z): does cuBLAS let a PDL dependent start early? (dev/pdl_cublas.py)
- Graph: stamp kernel -> torch.matmul (cuBLAS) -> our kernel launched with/without the PDL attribute (132 blocks,
  globaltimer at entry and after griddepcontrol.wait). H100, cold L2 (256 MB flush), median of 30 replays, us from stamp:
  | GEMM (M,K,N) | cuBLAS kernels | no PDL start | PDL start / wait done |
  | 16,4096,4608 | nvjet tst (no split) | 20.58 | 21.09 / 21.09 |
  | 32,4096,4096 | nvjet tst (no split) | 19.90 | 19.74 / 19.74 |
  | 16,4096,1024 | nvjet splitK + splitKreduce | 10.46 | 9.15 / 9.50 (-0.96) |
  | 32,8192,4096 | nvjet splitK + splitKreduce | 37.06 | 35.20 / 35.71 (-1.35) |
  So cublasLt's splitKreduce_kernel DOES trigger its dependents (STATE-layer's "cuBLAS never triggers" holds only for
  the plain nvjet kernels): a PDL secondary with griddepcontrol.wait after a split-K cuBLAS GEMM saves ~1-1.4 us.
- But the layer has no splitKreduce left (judge profile 703d3a7e: T<=32 qkv/pooler/unpool are g19 or QSS bmm; the
  remaining nvjet kernels (o at d16 full/d32, verify T=128/256) are non-split). Nothing to apply now; useful only if a
  future tuner pick brings a split-K cuBLAS back (then launch its successor with PDL). Not checked on the judge.

## Session 31 (2026-10-26 14:40-15:35Z): L2 cache hints on proj kernels inside the layer (moe s21 idea): pr1 evict_last WIN
- No judge runs (proj checks 20/20 today). All in a /tmp copy of r-proj's muse_decode_layer (v76 = r-layer 5c5329d), TUNE = {}
  in every file, tools/ab_files.py interleaved, H100 medians us. Tools: dev/s31/l2h_patch.py (evict_first cache_hint on the
  ld.global.nc W loads of g10/g12/g13/g14/g19(W bulk)/pr1/pr8, or g17(+n2u)), dev/s31/pf_last.py (evict_last cache_hint on
  every cp.async.bulk.prefetch.L2 after `namespace g10 {`, optional site list), dev/s31/evl_survive.py.
- evict_first on the weight streams (moe's idea for one-pass streams): NULL. proj kernels all hinted vs base, 2 runs: every
  shape within +-1.3 us, no consistent sign. g17/n2u hinted: null too.
- evict_last on the L2 PREFETCHES: WIN at T=1. Sites in order: 0 pr1 router-W next-group prefetch (FP32 rw, before the PDL
  wait), 1-2 pr1 pooler npf chunks (inactive: npf 0 default), 3 g10 o pfl2, 4 g17, 5-6 upf in tk2f/norm, 7 n2u, 8 nu8.
  d1 sliding / full deltas vs base (5 runs): all sites -1.75/-0.65, -1.1/-1.0, -1.9/-1.7, -1.8/-1.7; pr1 sites only
  -1.7/-1.25, -1.1/-0.3; site 0 alone -0.85/-1.2, -0.6/-1.6 (= the pr1 part). v1 (no pr1) flat with site 0; all sites
  v1 sl -0.2..-1.0. T>=128 all sites: v16 sl +3.3, +1.5, -4.0 (noise there is +-3), v32 -1.7/-2.6 once.
  Why: pr1's router W prefetch for the next row group was being evicted by the pooler stream before use.
- Not an artifact: evict_last-prefetched lines do NOT survive a 512 MB ordinary-read flush (evl_survive.py, 16 MB buffer:
  cold 12.1, evict_last+flush 12.4, warm 10.5 us event-timed), so the cold-L2 timing stays cold.
- Bitwise: decode 1 outputs identical to base (tools/files_eq.py, 3 seeds); tests 13/13 on every variant.
- Dead: pr1_npf 4/8 WITH evict_last: d1 +0.7..+6 us (pinning pooler chunks hurts, like plain npf).
- Handoff: dev/s31/pr1_evl_layer.patch (site 0, pr1_src.cu, `git apply` in r-layer), optional all sites via pf_last.py.

## Session 32 (2026-10-26 15:03-15:35Z): evict_last on all layer L2 prefetches, rebased to layer v77/v78 + KV_PF hint
- No judge runs (proj checks 20/20). /tmp copies of r-layer v77 (b66a869) and v78 (5c27e0a), TUNE = {} appended to every
  file, tools/ab_files.py interleaved, H100 medians us. New tool dev/s32/pf_all.py (any prefetch site in the file, by index;
  `list` prints them): 0-3 attention O_PF (3 kernels), 4 moe twm prefetch_l2 (left alone), 5 Triton boundary PF, 6-14 = s31's
  pf_last sites (6 pr1 router W, 7-8 pr1 npf, 9 g10 o, 10 g17, 11-12 upf, 13 n2u, 14 nu8). Plus KV_PF (g13's
  `prefetch.global.L2` of the d1 sliding KV rows) -> `prefetch.global.L2::evict_last` (the 28 MB qkv stream runs between).
- v77, 5 rounds (two runs): A base / B site 6 / C sites 6-14 / D = C + KV_PF evict_last / E = D + O_PF + boundary:
  d1 sl 88.43 87.71 87.18 86.90 87.12 | 88.05 - 87.07 86.56 86.54; d1 full 108.77 107.86 107.70 107.66 107.98 | 108.40 - 107.62
  107.81 107.86. d16/d32 (A C D E): all within +-1 (E +0.5..1.6 at d16: O_PF hint dead). verify (A C D): noise (+-0.8).
- v78, 6 rounds, A vs D: d1 sl 88.72 -> 86.69, full 107.81 -> 106.94, v1 sl 149.36 -> 148.58, full 160.88 -> 161.04.
  Est. layer score ~+0.3% (d1 weights .161/.056 per us).
- Correctness: bench.py test 13/13 (v77+D and v78+D), scored_test all shapes x2 seeds 0 bad, replay decode 1/16 0/20,
  files_eq d1 bitwise identical x3 seeds (v1 diffs = the base's own nondeterminism, orig vs orig shows the same).
- Handoff: dev/s32/evl_all_layer.patch (layer.py + pr1_src.cu on r-layer 5c27e0a; then build.py). Supersedes s31's patch.
- Dead too: evict_last on the remaining CUDA `prefetch.global.L2` sites (pr8 router W WS/WN, g13 pb, upf, vecs) on top of
  D (v78, 6 rounds, A/D/F): d1 sl 88.20/86.92/87.05, full 109.09/107.71/107.61, v1 sl 148.73/148.36/148.68, full
  162.30/162.08/162.10: F = D. Repeat confirms D at d1 -1.3/-1.4 us.

## Session 33 (2026-10-26 15:22-17:15Z): gemv13 (12-bit) for the layer's T=1 o as a PDL secondary: TUNE candidate
- No judge runs (proj checks 20/20 for 10-26; sandbox ends ~19:10Z). Work in /tmp copies of r-layer v79, then v82 (f970e18).
- Idea: in the layer, o at T=1 is gemv10 (BF16, 16.8 MB) launched as a PDL secondary of csl; s23's H200 diag only
  compared g13 vs g10 standalone. g13 PDL mode: the W prologue (ld.global.nc) before griddepcontrol.wait, x by __ldcg
  after it (SASS checked: LDG...CONSTANT before ACQBULK, x LDG.STRONG.GPU after). Script dev/s33/g13o.py (layer.py only).
- Timelines (H100 d1 sliding): A (g10) csl 13.7-19.7, o ends 24.4 (4.7 after csl). Every g13 config leaves ~4.6 us after
  csl. globaltimer stamps per block (dev/s33/dbg13.py): all 528 o blocks resident at entry; after the wait x arrives in
  0.3-2 us (blocks with W still in flight get x late), then ~2.3-2.8 us of decode + FMA + reduction per block: the
  post-wait part is ISSUE-bound (12-bit decode), not bytes.
- g13p (dev/s33/g13p.py): every chunk loaded AND decoded to packed BF16 pairs before the wait (ptxas sank the decode
  below ACQBULK until a branch on the XOR of all decoded words pinned it: 150 PRMT/LOP3 before, 55 after). Tail after
  csl 4.7 -> 3.0 us, BUT loading all 12.6 MB beside csl stretches csl 6.0 -> 8.1 us: net +1..2 us. Half before / half
  after the wait (pq variants): ties. Same zero-sum as the layer's prefetch-during-attention attempts. DEAD.
- Plain g13 PDL (u2 nv1 minb4 pf1 528 blocks, half of W before the wait): ab_files tuner off (TUNE = {}), d1 full
  -1.3, -1.0, -0.9, -1.2, -0.7, -0.9/-0.9 (v82) us over 6 runs; d1 sliding bimodal: -0.2, -0.1, -1.85, -0.35, -0.4/0.0.
  Configs B (2,2,2,1,264) ~-0.3, C/G (pf 2: all W before the wait) +1..2, D (u4) +1.5, F (nv4 132) +1.
- Handoff: dev/s33/g13o_layer.patch (on r-layer f970e18 = v82): TUNE option {"g13o": True} at T=1 (default off; packs o
  with _pack13, +12.6 MB). Forced on: bench test 13/13, scored d1 x3 0 bad, replay d1 0/20 + 0/20; o output = the
  BF16 rounding of the float64 product (dev/s33/chk_o.py, 5 seeds). Est. H200: full d1 ~-1 us x 0.056%/us = ~+0.05%.

## Session 34 (2026-10-26 15:50-16:25Z): DEC4 = 12-bit decode in 4 instead of 5 instructions per BF16 pair: WIN (H200 proxy)
- No judge runs (proj checks 20/20 for 10-26; sandbox ends ~19:10Z). Layer's ask (Requests layer s38): g19 at M=32 on H200.
- H200 proxy (H100 GRID 72, MODE 2, PRED 3000, NPRE 2) pool m32 (1024,8,2,xb1) = 29.6 us; configs AP4/NS4, KC512/NS3,
  AP4/NS3, AP2/NS8, RG16/AP4, TS2, RPW2: all tie or lose (as s16). Decomposition: consumer-only (G19_NOW) 27.1, NOW+NOMMA
  13.8, NOW+NOX 19.65 => consumers bound it; decode +5.9, mma+ldmatrix +7.5 us on top, ADDITIVE.
- ncu (pool m32 G72): issue slots 39% busy (2.8 warps/scheduler), stalls per issue: long_scoreboard 1.7, wait 1.5,
  short_sb 0.7, math throttle 0.5; ~29% of stall samples in mbarrier-wait branches. SASS mix per HMMA: LOP3 3.3, IMAD 1.3,
  PRMT 1.0, SHF 0.7. Each decoded word was SHF/IMAD.SHL + LOP3(& 0x07800780) + LOP3(| 0x38003800) + PRMT + LOP3(& | ):
  ptxas cannot put two immediates in one LOP3, so (s & M2) | C took two.
- DEC4 (dev/s34/dec4.py, a source transform, applied to gemv19/14/13.cu, pr1.cu, pr8.cu): both constants in registers made
  opaque by `>> (gridDim.y >> 16)` (always 0, any launch shape), one 3-register lop3 (s & m2) | cc, and the final combine
  as an explicit lop3. Bit-identical decode. H100 H200-proxy interleaved A/B (dev/g19ab.py, now SRC_A/SRC_B, 7 rounds,
  non-overlapping ranges), B-A us: pool m32 -1.36, unp m32 -1.22, qkv m32 -0.58; pool m16 (8,3) -0.85 / (4,4) -1.60,
  unp m16 -0.69 / -1.63, qkv m16 -0.50 / -0.93. (x0.73 ~ H200: ~-0.4..-1.2 us per call.) ESCW (adversarial escapes x40)
  ok on qkv/o/pool/unp m16/m32; proj bench.py test + esc pass; proj submission.py rebuilt (g13/g14 picks carry DEC4).
- Layer (/tmp copy of r-layer 61fe6fe, TUNE = {} in both, dec4.py on g13/g14/g19/pr1/pr8 _src.cu + build.py): bench.py
  test 13/13. tools/ab_files.py has a FILE-POSITION BIAS of ~1.3-1.9 us at d16/d32 sliding (A,B order: B +1.3/+1.3;
  B,A order: B -1.3/-1.9): order A B B A to cancel it. Run 1 (A,B, 5 rounds): d1 sl/full +0.1/-0.2, d16 full -1.4,
  d32 full -1.1, v1 sl/full -0.1/-1.1. H100 timeline d32 sliding: g19 pooler 30.5 vs 30.6 us (shares HBM with
  _router_split on the H100: bandwidth-bound there, not consumer-bound), so the H100 layer under-reports the H200 gain.
- Run 2 (B,A order, "sliding", 7 rounds): B-A d1 -0.1, d16 -1.3, d32 -1.9, v1 -0.1, v16 +0.7, v32 -0.2. Run 3 (A B B A,
  "decode", 4 rounds; mean of the two positions, B-A): d1 sl 0.0 full +0.2, d16 sl +0.5 full -1.0, d32 sl +1.3 full -0.5.
  => in the H100 layer DEC4 is within noise (+-1.5 at d16/d32 sliding); H200 decides (bit-identical, strictly fewer
  instructions: safe to bundle).
- Layer correctness (B copy): bench.py test 13/13, tools/scored_test.py all 12 shapes x 2 seeds 24 PASS 0 bad,
  replay_test decode 0/20 on all 6, files_eq A vs B decode 1 full x3 seeds bitwise identical (one earlier A-B d1 full
  seed-1 diff 1.6e-4 did not reproduce: run-to-run nondeterminism of the base path).
- Dead: G19_G (atoms per load group 1/4 vs default 2): +-0.2 us (s18 again: issue order is not the bound).
- moe's dq() has no two-immediate logic op (already 1 imm each): nothing to hand to moe from this trick.
- Handoff: dev/s34/dec4_layer.patch (g13/g14/g19/pr1/pr8 _src.cu; then build.py), applies on r-layer f5fe1ec.

## Session 35 (2026-10-26 16:25-16:45Z): DEC3 = 12-bit decode in 3 instructions per BF16 pair: WIN (H200 proxy)
- No judge runs (proj checks 20/20 for 10-26). Item: layer request s38 (g19 consumer-bound at M=32/16 on the H200).
- Idea: re-arrange the same 12 bits per weight so the decode needs neither the sign-replicating prmt + 0x807f807f mask
  nor the exponent constant OR: byte' = c0 << 7 | mantissa (code bit 0 sits exactly at exponent bit 7); the code word
  holds sign, c1, c2, c3 of weight 2P+h at bits (15, 8, 9, 10) + 16h + d_P (mod 32), d = (0, -4, -8, +4). Kernel word P:
  p = prmt(bytes, 0x38383838, 0x4140 / 0x4342) (byte 1/3 = 0x38 = exponent bits 11-13), result = p | (sh_P(nw) & 0x87008700)
  with sh = (nw, nw << 4, nw << 8, rotr(nw, 4)): 3 instructions per pair, P0 needs no shift: dec8 11 instead of 16.
  Escapes (byte 0, code 0) decode exactly as before (2^(base-127)); headers unchanged.
- dev/s35/dec3.py: tr3 (old packed -> DEC3), kernel-op emulation check (1M random + edge groups: identical bits);
  dev/s35/dec3_src.py: source transform (decw<P>, old decode kept under -DPJ_DEC4 for A/B builds), applied to gemv13-16,
  18-22, pr1, pr8 (15/16/18/20/21/22 first got s34's dec4.py). sub_template: _dec3(sm, code) in every packer (_pack13/14/
  16/19, _pack22 via _pack19), raises if its bit round trip fails. g19.py / g19ab.py: DEFS containing G19_DEC3 = tr3 the
  packs; now default: g19.py / g19ab.py tr3 their (old-layout pack16) packs unless DEFS has PJ_DEC4 (old decode).
- H100 H200-proxy interleaved A/B (g19ab, GRID 72, 7 rounds, A = DEC4, B = DEC3; ranges non-overlapping except unp m32):
  pool m32 (1024,8,2) -0.48, unp m32 -0.16, qkv m32 -0.06; pool m16 (8,3) -0.74 / (4,4) -1.65; unp m16 -0.70 / -1.39;
  qkv m16 -0.43 / -1.02 us (x 0.73 ~ H200). g19.py ESCW (adversarial escapes x40) ok on qkv/o/pool/unp/poolr m16/m32.
  proj bench.py test 39/39 + esc pass. m32 gains less than m16: there mma + ldmatrix dominate the consumers (s34).
- Layer (/tmp copy of r-layer cf16011, kernels + packers patched; dev/s35/dec3_layer.patch applies on 8a24599): bench.py
  test 13/13, tools/scored_test.py 24/24 PASS (0 bad), files_eq vs main (TUNE cleared): T=1/8/v32 bitwise identical;
  d16/d32/v16 differ at the same size as main vs main (g19's red.add order). ab_files A B B A (3-4 rounds): all shapes
  tie within +-0.5 us (d1 full -0.4); the H100 layer cannot show g19 consumer gains (HBM shared with _router_split), as
  with DEC4. tools/esc_probe.py is stale on layer main (g13_gemv signature).
- The layer agent picked DEC3 up from the r-proj commit while I tested (its working tree, g22 left on the old layout via
  _pack19(dec3=False): correct, since g22_src.cu has the old decode).

## Session 36 (2026-10-26 16:42-17:25Z): pr8 bulk L2 prefetch after the wait: WIN in the H100 layer (v1 -1.6 us)
- No judge runs (proj checks 20/20 for 10-26). Item: layer s41 "pr8 23.5 us at 2.4 TB/s (m8 issue-bound)".
- New probes (dev/pr8.cu / pr8.py): PRED=<ns> (PDL predecessor pr8_pred, 32x256 threads, triggers at start, like g19.py),
  DEFS=PR8_NOW (W re-read from L1/L2), PR8_NOMATH (no decode/mma), PR8_RLAST / PR8_RMID=<atom> (router item placement);
  TRACE=1 now honours PRED.
- H100 standalone (no pred) 31.4 us; pooler only (npf -1) 26.2; NOW+NOMATH skeleton 17.7 (!); NOMATH 24.8: NOT issue-bound
  (math adds ~1.4 us). PF 2/4/8 flat, PF 16 +8. Trace after a 3 us pred (median us): t1 (x32 slice landed) 10.5,
  t2 (router partials stored) 14.1, t3 14.6, t4 (pooler loop end) 29.4, t5 31.0; loop = 50 MB in ~15 us (3.3 TB/s). So the
  loss = the serialized prologue: router W (8.4 MB) + PF atoms (6.3 MB) queue ahead of x32, then ~3.5 us router math/sync
  where HBM idles; plus a ~9 us t4 spread (blocks on the pred's SMs start ~3 us late, fixed work per block).
- Fix: npf 2xx = one cp.async.bulk.prefetch.L2 per (row group, WS/WN) of atoms [PF, PF + xx), issued after the wait right
  after the x/x32/header cp.asyncs (streams during the router phase). Standalone after a 3 us pred, 4 interleaved rounds:
  npf 0 25.8-26.4 vs 212 25.0-25.7 (-0.8). 204-216 similar, 224 +0.8, PF2+208 no gain.
- Dead: same prefetch BEFORE the wait (npf 1xx: 104/108 tie, 112+ worse; with PRED 6000 108 -1.4 but layer-like PRED
  3-4 us loses), pre+post combos (npf 1000 + 32 pre + post: +0.4..3), router item after the loop (RLAST +5 us: the finisher
  then waits on the slowest pooler block), router item mid-loop (RMID 4/8 tie, 16 +1), the old per-warp prefetch.global.L2
  (npf 4-28: -0.5 at best), gemv14 qkv m8 with the same post-x prefetch (DEFS=G14_BPF=8/12/20: +0.2..1.2 us; g14 has no
  router stall to fill).
- Layer (/tmp/pj36 copies of r-layer 0dcefe6, TUNE = {} appended to both submission.py; B = new pr8_src.cu + PR8_CFG npf
  212): ab_files A B B A, 7 rounds: v1 sliding 149.16 -> 147.60, full 161.87 -> 160.20 (-1.56/-1.67); 1st run (5 rounds)
  full -1.6; 208/216/220 vs A also -1..-2.5 (noise between them). Tests 13/13, scored_test v1 x3 PASS, route_judge v1
  0/80, replay_test 0 fails. files_eq: run-to-run differences in output[1] (next attention input) at v1 sliding seed 0
  appear B vs B and A vs A (decode 1 full seed 1: 2 elements) = pre-existing nondeterminism, not pr8 (B with npf 0 == A).
- Handoff: dev/s36/pr8_bpf_layer.patch (pr8_src.cu + PR8_CFG), `git apply -p1` on r-layer 0dcefe6. COORD Handoffs proj s36.
- g14.py now packs with sub_template's _pack14 (DEC3 layout; its own pack14 was the old layout) and takes DEFS.
- Dead too (17:12Z): DEFS=PR8_PFLATE (pooler PF register preloads moved after the wait, atoms [0, PF) bulk-prefetched
  first, so x32 doesn't queue behind them): +2.0 us after a 3-4 us pred (212: 24.4 -> 26.4). Pre-wait W loads are worth
  more than an earlier x32.

## Session 37 (2026-10-26 17:09-17:30Z): pr8 block-finish spread measured; interleaved row groups DEAD
- No judge runs (proj checks 20/20 for 10-26). Item: STATE Next -25 (pr8 block-finish spread).
- dev/pr8.py TRACE=2 (new): per-block start vs loop end, by block-id range. After a 3 us PDL pred (npf 212): the 28 blocks
  on the pred's SMs (ids ~100-127) start 2.8 us late but end only ~1.5 us later on average (t4 27.2 vs 24.1-26.3 for the
  others): they catch up (HBM is shared). The spread is mostly RANDOM: early starters alone end anywhere in 18.7-28.4 us,
  and which blocks are last changes every run. t4 mean ~25.7 vs max ~29.2: ~3 us tail (standalone, traced).
- DEFS=PR8_RGI (new probe, off by default): block b owns row groups b, b+128, b+256, b+384 instead of 4 contiguous ones.
  Bit-identical outputs, test 0 bad; interleaved A/B, 4 rounds, PRED 3000: base 24.35-24.42 vs RGI 24.56-24.70 us: +0.25, DEAD.
- So the tail cannot be fixed statically (late-start blocks matter little; the rest is random): only dynamic balancing.
  Design for a future session (est. <= -2 us at v1, ~+0.2% layer): split each row group's K into 2 halves; a block finishing
  its own work early claims (atomicCAS on a per-(rg, half) flag) a second half not yet started by its owner and writes its
  FP32 [8 tok x 8 rows] partial; out = (h0 + h1 + fixups) * scale in fixed order whoever computed each half (deterministic,
  bits differ from today's single accumulator: re-run layer route/scored tests). Needs a per-call flag reset that cannot
  race (epoch in the flag value, e.g. a call counter bumped by the finisher block) and an owner spin on the stolen half's
  done flag (safe: the stealer is resident).

## Session 38 (2026-10-26 17:22-17:40Z): pr8 dynamic balancing: whole-row-group work queue ruled out on paper; no build
- No judge runs (proj checks 20/20 for 10-26). Sandbox expires ~19:10Z, push by 18:30Z: no time for a K-split build+layer test.
- Considered the cheap version of s37's design: no K split, the block's work as units of whole row groups (WG 16 = 2 rg per
  1024-thread block, 256 units; or WG 32 = 1 rg, 512 units), first unit static, the rest from an atomic counter
  (grab issued early in the unit so its ~1-3 us round trip hides; W register prefetch runs on into the next unit; header
  double buffer fits the smem the launcher already reserves for 4 rg). Outputs would be deterministic (one block per rg).
  Ruled out without a build: with <= 4 units per block a unit lasts ~6 us (~12 us at 2 units), which is LONGER than the ~3 us random
  finish spread it should absorb, and grabs made early in a unit hand out the queue in start order (with 2 units/block
  every block simply gets one more unit: no balancing at all). Balancing needs units well under the spread (>= 8-16 per
  block) = K splitting, so partial combines (stealer partial + done flag/ticket, ~1-2 us at the tail).
- Value estimate: <= ~1.5 us net at v1 (score weight ~.096%/us sliding) = ~0.1-0.15% layer. Moved below the line: not worth
  a next-sandbox session unless everything bigger is gone.

## Session 39 (2026-10-26 17:40-18:00Z): overlap probe for layer s45's "pooler W into SMEM during csl/o/bnd" idea
No judge runs (checks 20/20). H100 probe dev/s39/fill.py (run from /tmp: `cp dev/s39/fill.py /tmp/x/; cd /tmp/x; python fill.py`):
one CUDA graph = 256 MB flush, then a side-stream `fill` kernel (132 blocks x 128 thr, 1/SM, TMA bulk copies of 26 MB into
192 KB SMEM per SM) forked beside a main-stream chain of NO tiny kernels (`xs.mul_`, XS elements), joined. Times = median
of 60 replays minus the flush-only graph. Env: PB bytes/SM, NB blocks, CH chunk (>0: all chunks issued at once;
<0: ONE |CH|-byte chunk in flight per SM; 999999: unpaced L2 prefetch; 1000000+g: 32 KB L2 prefetch per SM every g ns).
- Unthrottled fill (all 192 KB/SM issued at once) is ZERO-SUM against ANY chain: m=1 cuBLAS o GEMV 2x: 15.5 + 13.2 ->
  28.6 (sum); 8 tiny 32-element kernels (8.5 us, launch-bound, no DRAM): +9.3 us (= the fill's whole stream time); 40 tiny
  kernels: 39.8 -> 49.1 (+9.2). Fewer SMs (NB 66/33, 13/6.5 MB) dilate by their own stream time too. So under a deep HBM
  queue even L2-hit / launch-latency-bound kernels slow down by ~the stream's duration (each dependent round trip waits).
- **Dilation is set by bytes IN FLIGHT, not by bandwidth.** 40-kernel chain (39.9 us), 26 MB fill, one chunk in flight/SM:
  CH  fill alone  chain dilation (32-elem kernels / 256 KB kernels)
  all-at-once 13.3  +9.2 / +9.9
  -32768      13.9  +3.8 / +6.4     <- ~full rate, 40-60% less dilation
  -16384      17.1  +2.1 / +3.5
  -8192       22.6  +0.7 / +2.6
  -4096       33.8  +0.0 / +2.7
  So a fill/prefetch that keeps ~32 KB in flight per SM (4.3 MB total) streams at ~95% of full rate and leaves latency
  chains much faster than one that queues its whole range. Net saving vs streaming later: 26 MB loaded for ~3.8-6.4 us of
  chain dilation instead of ~10.5 us of serial stream = ~4-6.5 us per 26 MB on H100 (H200 untested).
- L2 prefetch (cp.async.bulk.prefetch.L2, 32 KB per op): unpaced +2.4/+3.1, paced 1-3 us per op +3.6..4.8 dilation; a
  follow-up reader test (torch.amax over the 26 MB, itself slow: 22 us, compute-bound) was inconclusive (unpaced prefetches
  may be partly dropped). Not a clean result; prefer the SMEM fill (completion is known) when testing this for real.
- Implication for the layer (s45 idea): an SMEM-resident pooler W at T=1 (pr1 resident during csl/o/bnd) is worth at most
  ~4-6 us/d1 layer, and ONLY if the filler keeps few bytes in flight (one 32 KB TMA per SM): a deep TMA ring (like g13's or
  pr1's prologue) issued beside csl/o/bnd will dilate them by its whole duration (explains the zero-sum L2 prefetch results,
  layer s2/s15/s34, if those were issued as one deep burst). Real csl/o/bnd touch more memory than this chain: expect the
  256-KB-kernel column or worse. Also the same rule for our own PDL prologues (W stages BEFORE griddepcontrol.wait in g19/pr1/
  pr8/g13): a shallow throttle there might help the predecessor more than it hurts us: untested, cheap to test in g19.py
  MODE=2 PRED=3000 (the PRED kernel is a spin, not memory-latency-bound: use a memory-touching PRED for that test).

## Session 40 (2026-10-26 17:50-18:05Z): throttled g19 PDL prologue vs a MEMORY-touching predecessor (STATE Next 26): DEAD
No judge runs (checks 20/20). dev/gemv19.cu `pred` now takes PRED < -1 = -N dependent __ldcg loads per thread (32x256
threads) through a 256 MB random index table (latency-bound, memory-touching, like the layer's glue); g19.py prints the
pred's own graph time. H100, MODE=2, numbers = graph(pred + g19) - graph(pred) in us (2 rounds for pool, within ~0.2):
  pool m16 (16,1024,8,3,1,0,1)  npre 3 / 2 / 1 / 0:  chase-8 22.0 / 21.8 / 22.4 / 23.2   chase-24 22.3 / 21.9 / 22.6 / 23.5
                                                    spin 3us 19.8 / 19.3 / 21.1 / 23.0
  unp  m16 (16,1024,4,4,2,0)    npre 4 / 3 / 2 / 1 / 0:  chase-8 22.1 / 22.1 / 22.7 / 22.7 / 23.2
                                                    spin 3us 19.9 / 20.2 / 21.4 / 22.0 / 23.0
- Beside a memory-touching pred the pre-wait W stream still pays, but only ~1.1-1.5 us (vs ~3.2 us beside a spin pred):
  the chase pred is dilated by the stream (as s39 predicted). Shallower prologues do NOT dilate it less enough to win:
  npre 1 is worse than deep everywhere; npre 2 vs 3 at pool m16 -0.1..-0.4 us (noise-level; the layer's pooler m16 cfg uses
  npre 3: a cheap A/B for the layer at most). Unpool: deepest best. Throttling our PDL prologues: dead. The s39 "one 32 KB
  in flight" rule matters for SIDE streams that run long beside a chain, not for g19's few-stage prologue.

## Session 41 (2026-10-26 17:57-18:10Z): housekeeping only
- No judge runs (proj checks 20/20; scored 5/10 but no check beats 1.3110). Sandbox ends ~19:10Z, push deadline 18:30Z: too
  short for a build + layer A/B. Condensed STATE-proj.md (155 -> 146 lines): s36-s40 summarized, "Next sandbox" plan rewritten
  (first check = main; then a fresh layer idle budget from v88; remaining small ideas ranked).

## Session 42 (2026-10-26 18:12Z): sanity only, sandbox expires ~19:10Z
- No judge runs (proj checks 20/20 for 10-26). main (6f068c9) re-verified on the H100: `python bench.py test` 39/39 PASS,
  `python bench.py esc` PASS (12-bit round trip + adversarial escapes). The first check in the next sandbox = main as is (STATE Next 1).

## s43 (2026-10-26 18:28-18:35Z): notes only
No judge runs (proj checks 20/20 for 10-26; sandbox expires ~19:10Z). main unchanged (s42 re-verified: bench test 39/39, esc PASS).
Next sandbox: first proj check = main as is; then pick up the layer's request list (COORD Requests) — the only open proj
item is pr8 dynamic K-half stealing (<= ~0.1% layer, low priority).

## s44 (2026-10-26 19:13-19:35Z, sandbox okc-night4, session 1): g19 m32 LONGRUN (fewer run ends) DEAD
- No judge runs (proj checks 20/20 for 10-26, reset 00:00Z). Sandbox re-verified: `python bench.py test` all PASS.
- Question: is g19 m32's per-run-end flush (pbuf store + escape fixups + finisher red.add of 16 KB, every SPK stages =
  every k-chunk) part of its consumer bound? dev/s44/gemv19_lr.cu (copy of gemv19.cu + G19_LONGRUN: tile-major stage order,
  run ends only at a tile change, so a block's run spans k-chunks; x released at chunk ends; G19_NOFIX2 skips the smem
  escape fixups) and dev/s44/g19ab.py (g19ab + TM packing for LONGRUN builds, env TOL). Both arms NOFIX2 (upper bound for B).
  H200 proxy (GRID 72, PRED 3000, R 6), pool m32, B-A: (1024,8,2,xb1) +4.99 us (x reloaded per chunk into a single
  buffer: stalls), (512,8,2,xb2) +0.34. => run-end flushes are NOT the m32 bound; tile-major order costs more x reloads
  than it saves. DEAD. (s18's conclusion stands: g19 m32 = consumer instruction count, mma.sync + decode.)

## s45 (2026-10-26 19:23-19:50Z, sandbox okc-night4, session 2): layer pooler T=16 g19 npre 2 vs 3 = TIE (dead)
- No judge runs (proj checks 20/20 for 10-26, reset 00:00Z). STATE Next 2's cheap A/B, in a copy of r-layer 4c98aa3 (v88,
  /tmp/pj45): G19_CFG ("pooler", 16) (16,1024,8,3,3,1) [A, main] vs (16,1024,8,3,2,1) [B], TUNE = {} in both (TUNE_BASE kept),
  tools/ab_files.py "decode 16" 8 rounds, files A B B A (H100 us, medians): d16 sliding A 198.34/199.22 B 198.97/199.10
  (B-A +0.26), d16 full A 295.98/295.26 B 295.32/295.90 (B-A 0.0). Standalone s40's -0.1..-0.4 does not show in the layer.
  Keep npre 3. Nothing for layer.

## s46 (2026-10-26 19:40-20:00Z, sandbox okc-night4, session 3): T=1 pooler W pre-fill beside csl/o/bnd = DEAD
- No judge runs (proj checks 20/20 for 10-26, reset 00:00Z). STATE Next 2's "SMEM-resident pooler" tested in its feasible
  form: a side-stream filler kernel (dev/s46/pwf_layer.patch on r-layer b5e07ab = v91; pwf_k: 132 x 32 thr, one 32 KB
  cp.async.bulk -> smem in flight per SM with an L2 evict_last hint) pulls pr1's pooler rows [424, 424+R) (the rows pr1
  reads right after its PF prologue) into L2; joined at the end of kernel_fn. (True SMEM residency for pr1 would need pr1
  resident during csl/o; it can't co-reside with o's 1024-thread blocks, and pr1 is already resident during bnd.)
- H100 layer ab_flag (6 rounds, medians, us) d1 sliding / full: base 86.8/105.5; fork after qkv R 1700 (20 MB)
  90.3/111.2, depth 2 90.7/111.2, R 900 89.4/108.2; 2nd run base 85.7/105.0, R 300 90.6/106.1, R 300 CH 16K 90.5/106.1,
  R 600 CH 8K evict_normal 88.0/106.7. Timelines (dev/s46/tl2.py = tools/timeline.py + exec stmt): R 1700 dilates csl
  5.4 -> 9.1 and o 8.4 -> 15.0 us (o's W stream runs pre-wait during csl: that window is NOT idle) while pr1's post-bnd
  time drops only 19.8 -> 16.9. Even R = 1 (24 KB) costs csl +1.3, o end +1.8 us: a concurrent side kernel in that
  chain costs by itself. Fork after attention / after o: the filler only gets SMs once o's blocks / pr1's blocks leave
  (starts 22.3 / 30.3) and then competes with pr1: worse.
- So at T=1 the only HBM-idle window is the boundary (~4.4 us), already covered by pr1's pre-wait prologue (deeper
  prologue = pr1_npf: lost on H200, s35). Overlap ideas at d1 are closed; m=1 kernels are at bare-stream speed (s17).
  What's left for d1 in proj's lane is fewer bytes (entropy-coded weights, s17 est. ~0.3% layer, big format change).

## s47 (2026-10-26 19:48-20:05Z, sandbox okc-night4, session 4): PDL .nc-hoist SASS audit of proj kernels in layer v91 + unpool T=32 grid A/B
- No judge runs (proj checks 20/20 for 10-26, reset 00:00Z).
- Audit (moe n4s2 found a real hoist in moe_norm): dev/s47_pdl_audit.py lists every load before the first ACQBULK
  (= griddepcontrol.wait) per kernel of r-layer's built layer_g10_6aab317afa96.so (proj kernels inside the layer, v91 era);
  each pre-wait load mapped to its param (c[0x0][0x2xx] base) by hand:
  * g13 (all, incl. the g13o o@T=1 PDL secondary): pre-wait = W (.NA), escape headers H, KV_PF page table (prefetch only);
    x pre-wait only when pdl == 0 (layer's `!pdl` guard), else x after ACQBULK by LDG.STRONG.GPU (__ldcg). SAFE.
  * pr1 <2,4,{1,2},4>: pre-wait = W (.NA), router W (__ldcs, EF), headers, and bpre's s/alpha/beta/gain (prepared).
    x / x32 loads sit AFTER ACQBULK, but the layer's pr1_src.cu non-bnd path (the live one: PR1_BND = False) still reads
    them with __ldg (r-proj dev/pr1.cu has __ldcg since s13): safe in today's SASS, fragile -> Request to layer.
  * pr8, g17, g19 (all cfgs, 0 pre-wait loads), g12, g10, pr3 (pre-wait LDGSTS = router hi/lo W only): SAFE.
  * n2u / nu8 (layer's) read ids/tw BEFORE the wait by design: safe because pr1 -> _rmsnorm_gain (triggers after its own
    wait) -> moe_tk2f (PDL secondary of the rmsnorm, no wait of its own, triggers at start) -> n2u: pr1 has completed
    before n2u can launch. Holds only while the rmsnorm triggers after its wait (TRIG2): noted for layer.
- In-layer A/B (copy of r-layer 5e93d29 = v91 in /tmp/pj47/L; tools/ab_flag.py "decode 32" 5 rounds, H100 medians us):
  G19_GRID ('unpool', 32) 132 / 120 / 108 / 96 / 132: d32 sliding 197.68 / 197.46 / 197.89 / 197.41 / 197.10, full
  415.89 / 415.76 / 415.62 / 416.34 / 415.73: all ties. On the H100 the T=32 unpool is not SM-bound in the layer (H200 is
  consumer-bound per the proxy, so H100 can't decide this). Dead on H100; not worth an H200 check alone.
- H100 timelines v91 (tools/timeline.py): d16 sliding: o is g19 (PDL), no gap > 0.5 us; d32: qkv nvjet 16.9 -> nds +0.5,
  o nvjet 6.3 (16.8 MB, 2.7 TB/s) -> boundary +0.5 us. The "T>=16 cuBLAS gaps without PDL triggers" item is <= ~1 us per
  shape on the H100 now (H200 gaps need the 00:00Z layer check's profile).

## s48 (2026-10-26 19:56-20:25Z, sandbox okc-night4, session 5): g19 escape-header/fixup cost at m32: measured; 2 fixes DEAD
- No judge runs (proj checks 20/20 for 10-26, reset 00:00Z). Question: of g19's m32 per-call overhead, how much is the
  run-start escape-header __ldg (cold global loads under full HBM) and the run-end fixup walk (s15 pbuf path)?
- New diag/option switches in dev/gemv19.cu (all default OFF; default build unchanged): G19_NOHDR (no header loads, empty
  slots: escapes unfixed), G19_NOFIXB (headers loaded + ballots run, but no slot in range via an opaque khi), G19_FIX2
  (fixup walk 2 escapes per round, loads first; bit-identical order), G19_HPF (next run's headers, predicted = static
  order's next run, __ldg'd one run ahead into registers).
- H200 proxy (H100 GRID 72 MODE 2 PRED 3000, dev/s44/g19ab.py, medians us, B-A):
  NOHDR vs main: pool m32 -1.45, unp m32 -0.98, pool/unp m16 -0.42/-0.44 (upper bound: loads + fixups).
  NOHDR vs NOFIXB: pool m32 -0.19, unp m32 -0.46 => the load stall is ~0.2-0.5, the fixups themselves ~0.5-1.2 us at m32.
  FIX2 vs main (R 6): pool m32 +0.72, unp m32 +0.18, m16 -0.46/-0.07: noise/loss -> DEAD (the walk is not latency-chain bound).
  HPF vs main (R 8): +0.38/+0.62 m32, +0.88/+0.35 m16: LOSS (run-start register copy waits on the prefetch / extra loads) -> DEAD.
- Left over: <= ~1 us/call (proxy) of escape fixups at m32 (~0.7 H200, 4 calls at d32 = ~0.2% layer). Only a different escape
  scheme (no per-run fixup: e.g. escapes as a tiny extra mma operand, or escape-free per-atom bases) could take it; not cheap.

## s49 (2026-10-26 20:08-20:30Z, sandbox okc-night4, session 6): "fewer bytes at m=1" bounded on paper: CLOSED
- No judge runs (proj checks 20/20 for 10-26, reset 00:00Z). STATE Next 2's last untried proj item (entropy-coded weights).
- dev/s49_entropy.py on the judge's distribution (randn/sqrt(k) -> BF16): sign + 7 mantissa bits are uniform (8 bits),
  exponent entropy 2.54 bits => 10.54 bits/weight is the lossless floor; 12-bit is within 1.46 bits (-12% bytes max).
  Fixed-rate windows below 16 codes escape far too often: 8 codes (11-bit) 3.5% of weights = 288 per 8192-row (a sparse
  side list at ~48 bits each costs more than the 1 bit saved), 6 codes 14%. So below 12 bits only variable-length coding
  works. Cheapest GPU-friendly VL scheme I found: per 4-weight group a 1-bit flag, 2-bit codes if all 4 exponents are in
  the top-4 window (p = 0.855^4 = 53%), else 4-bit: 11.2 bits avg = -6.7% bytes; needs per-warp prefix sums for offsets
  in g13/pr1/g14 (g13 already needs its whole register file). Upside at d1 ~ -6.7% of ~71 MB (qkv 21 + o 12.6 + pooler
  37.5) ~ -1.3 H200 us per d1 layer (~0.25% layer) if decode stays free, which it would not at m=1. Not worth a multi-
  session build. Closed. Also: _compress uses ONE base per matrix (global max exponent); other weights (zeros, outlier
  rows, NaN/Inf) only add escapes -> per-layer BF16 fallback, so correctness on reviewers' weights is not at risk.
- With this, every proj-lane item on the list is measured or bounded below ~0.3% layer. Request to lead for a new lane.

## s50 (2026-10-26 20:27-21:20Z, sandbox okc-night4, session 7): BF16 wgmma at m16/m32 DEAD; H200 g19 decomposition DIAG queued
- No judge runs (proj checks 20/20 for 10-26, reset 00:00Z). Lead 20:16Z: the layer's T>=16 GEMMs are what count.
- (a) gemv23 (BF16 SS-wgmma swap-AB, plain BF16 W by bulk TMA, no decode) instantiated at M = 16/32 (wgss<16>/<32> added
  to dev/gemv23.cu). H100 standalone graph, cold L2, us (cuBLAS F.linear | best g23): qkv m16 17.46 | 16.94, m32 17.70 |
  17.66; o m16 8.91 | 11.28, m32 9.07 | 11.20; pool m16 27.38 | 26.78, m32 27.58 | 27.66; unp m16 24.78 | 27.01, m32 25.26 |
  27.54. NOMMA pool m32 = 26.9-27.3: W-stream-bound (67 MB at ~2.5 TB/s). BF16 bytes floor on the H200 (67 MB ~ 18.3 us)
  ~= g19's current m32 standalone: no room. DEAD (the 12-bit decode is not worth dropping at m32).
- (b) New print-only `_diag19d` in dev/sub_template.py (DIAG19D, DIAG19D_DEFS, DIAG19D_GRIDS): g19 pool m32/m16, unp m32
  rebuilt 5 ways (main, NOW, NOW+NOMMA, NOW+NOX, NOFIX) x grids 132/108, after the 3 us PDL predecessor. Adds ~1-2 min
  compile. Purpose: the H200's own split of g19 m32 into skeleton / decode / mma+ldsm / W stream (only the H100 proxy so
  far). H100 local (us over pred): pool m32 G132 main 21.50 | NOW 16.19 | NOMMA 10.08 | NOX 11.81 | NOFIX 21.60;
  G108 23.04 | 19.01 | 11.74 | 13.63 | 23.33; pool m16 G132 19.62 | 14.14 | 10.56 | 12.35; unp m32 G132 21.44 | 16.00 |
  10.02 | 11.20 | 21.70. bench.py test all PASS (3:56 incl. builds).
  Reading it at 00:00Z: if H200 main ~= NOW, consumers bound it (then the skeleton NOMMA share says whether barrier/run
  structure or decode+mma is the target); if main >> NOW, it is the W stream (then nothing kernel-side is left).
- (c) H100 timelines (dev/g19.py STAGES=1, MODE 2 PRED 3000 G132, pool m32 (1024,8,2,xb1)), new diag G19_TR2 (stamps
  stage 2 of warp 0: slot 6 after tag/x/header, 7 after the atom loop, 8 after the empty arrive; default build unchanged):
  skeleton (NOW+NOMMA): pred ends 4.9, x0 7.2 (2.2 us x staging, known), then ~0.5 us per stage although every stage is
  ready: got -> slot 6 0.23 us (only a tag lds + integer div/mod!), atom loop 0.16, arrive -> next got 0.13. NOW (decode +
  mma, no W): got -> 6 0.22, loop 0.67, arrive -> got 0.13 => ~1/3 of a consumer stage is OUTSIDE the decode/mma loop.
  G19_SPIN (test_wait) changes nothing (not try_wait suspension). Kernel has a 16 B stack frame (xph/xpc[xb] runtime-
  indexed: LDL/STL once per x use, not per stage: minor). Unexplained 0.35 us/stage: next step = clock64 stamps (rule
  out globaltimer cost) and an ncu source view of NOW mode around the loop top. If real, the target is ~1/3 of the
  consumer time at m32 (H200: g19 consumer-bound) = up to ~-3..5 us/call; the H200 _diag19d at 00:00Z says how much
  of main is consumer time there.

## s51 (2026-10-26 21:05-21:45Z, sandbox okc-night4, session 8): g19 loop-top overhead explained; G19_TAG2 queued for H200
- No judge runs (proj checks 20/20 for 10-26, reset 00:00Z). Target: s50's unexplained ~0.35 us/stage outside the atom loop.
- Cause: runtime integer div/mod in the consumers' loop top. proj's g19 has a runtime `tm` (tile-major order, mode 3), and
  both branches of `tm ? .. : ..` were computed: u/SPK % S, u/(S*SPK), u%(ntr*SPK), u/(ntr*SPK) = several runtime divides
  per stage. (The layer's g19 copy has no tm: one runtime div pair.)
- New switches in dev/gemv19.cu (default OFF, default build unchanged): G19_TAG2 (producer writes tag2[slot] = ks | st<<8 |
  tr<<13 beside tag; consumers read it: no div/mod; +4*ns smem; bit-identical math), G19_NOTM (tm = 0 at compile time =
  the layer's copy), G19_CLK (trace stamps = clock64 instead of globaltimer). dev/s44/g19ab.py: TOL=skip for diag builds.
- H100 A/B (dev/s44/g19ab.py, MODE 2 PRED 3000, medians us, B-A), consumer-only (G19_NOW):
  vs proj main (runtime tm), G72: pool m32 -1.49, m16 -1.35, unp m32 -0.98, m16 -0.94; G132: -1.46/-1.10/-0.36/-0.75.
  vs NOTM (= layer-like), G72: pool m32 -0.36, m16 -0.98, unp m32 -0.13, m16 -1.08. NOW+NOMMA pool m32 G72: -0.76.
  Full kernel (W stream on), G72: main vs TAG2 -0.2..+0.3 (tie); NOTM vs NOTM+TAG2 -0.23..+0.19 (tie): the H100 is
  W-bound at this proxy, so only the H200 can show it (it is believed consumer-bound at m32).
- Clock64 trace (dev/g19.py DEFS=G19_TRACE,G19_TR2,G19_CLK STAGES=1 MODE 2 PRED 3000 G132 pool m32 (1024,8,2,xb1)), stage 2
  of warp 0, kcycles: NOTM got->loop start 0.30, atom loop 1.51, arrive 0.02; NOTM+TAG2 0.14 / 1.45 / 0.02. So ~160 cyc
  (~0.09 us) per stage was the div/mod chain; the rest of s50's "0.35 us" was NOW-mode producer turnaround (2-slot ring:
  arrive -> producer's empty wait -> its full arrive -> wake ~0.13 us), not consumer work. (NOW builds fail g19.py's
  correctness check and print nothing; use the g19ab TOL=skip path for NOW numbers.)
- _diag19d DEFS += "G19_NOTM", "G19_NOTM G19_TAG2" (7 builds; bench.py test 39/39, H100 diag19d pool m32 G132 NOTM 22.72 |
  +TAG2 22.37, m16 19.30 | 18.98, unp m32 21.28 | 21.82: ties). Read at 00:00Z: NOTM vs NOTM TAG2 on the H200; if TAG2
  wins >= ~0.3 us at m16 or m32, hand it to layer (small patch on its g19_src.cu: tag2 after ptag, cbuf shifted, +4*ns smem).
- Also in diag19 (main) at 00:00Z: o m16 g19 vs lt (H100 now 5.79 vs 7.52; H200 lost +0.8 before s15/DEC3): if g19 wins
  now on the H200, that is a layer TUNE option at T=16 o.
- Layer copy (s51 later): dev/s51/tag2_layer.patch = TAG2 on r-layer's g19_src.cu (088d8d7; no tm there, single producer
  lane). dev/s51/g19ab_layer.py = g19ab for the layer's g19 interface (6-arg G19_CASE), prints max|A-B| vs max|A-A'|
  (split-K red.add order is nondeterministic: equal = same math). H100 H200-proxy, medians B-A us: G72 R8 pool m32 -0.94,
  m16 -0.06, unp m32 -0.53, m16 -0.48; G72 R10 -0.85/-0.52/-0.46/-0.11; G88 R10 -0.62/+0.09/-0.38/-0.09 (non-overlapping
  ranges at m32). qkv/o m16/m32 tie. In a /tmp clone of r-layer 088d8d7: bench.py test 13/13 PASS; tools/ab_files.py
  6 rounds A B B A (tuner on): all 12 scored shapes tie (d32 full 414.66/414.05 vs 413.97/414.79; d16 sl 196.92/196.62
  vs 196.71/196.74). Handoff posted, conditional on the 00:00Z diag19d (H200) or a layer A/B check.

## s52 (2026-10-26 21:32-22:15Z, sandbox okc-night4, session 9): PDL flag handoff, g19 SWP, g19 ACC4: all DEAD (H100)
- No judge runs (proj checks 20/20 for 10-26, reset 00:00Z). Target: layer's g19 at T=16/32 (lead 20:16Z, layer s60).
- (a) dev/s52/pdlwait.py: latency of griddepcontrol.wait vs a flag handoff (pred blocks: __threadfence + atomicAdd a
  counter; secondary spins ld.acquire.gpu) at a PDL boundary (132 x 256 pred, 3 us spin, pred triggers at start). H100,
  us after the last pred block's final write: gdc.wait 0.77 (blocks 0.77-0.86); flag 1.02 (0.96-1.22): fence + atomic
  cost more than the wait. DEAD: griddepcontrol.wait is the fastest boundary (relevant to every layer PDL edge).
- (b) ncu (SourceCounters, dev/g19.py PROF=1 SH=pool GRID 72 MODE 2 PRED 3000 NPRE 2 DEFS G19_NOTM,G19_TAG2,
  "32,1024,8,2,4,0,1"; report /tmp/pj52/g19m32.ncu-rep): issue 0.32/scheduler, 2.46 warps active, 0.40 eligible;
  stalls: long_sb 28% (nearly all on mbarrier try_wait branches = waiting for W stages: H100 W-bound), short_sb 24%
  (HMMA/PRMT waiting on the just-issued LDS/ldmatrix), wait 15%, selected 14%, math 4%.
- (c) G19_SWP (gemv19.cu, default off, RPW 1): next atom group's LDS + ldmatrix issued before this group's decode + mma
  (register double buffer). H100 G72 R8 B-A: full kernel -0.18..+0.39 (tie), consumer-only (G19_NOW) -0.09..+0.14 (tie)
  on qkv/pool/unp m16/m32. ptxas already hides this; short_sb is not the bound. DEAD.
- (d) G19_ACC4 (with SWP): odd atoms into a second accumulator pair (8/16 independent HMMA chains): consumer-only B-A
  +0.01..+0.60 (worse, more registers). Not HMMA-latency-bound. DEAD.
- Arithmetic: NOW pool m32 G72 22.9 us = 14.2 stages/block -> 1.6 us/stage = 512 HMMA/SM/stage (~0.17 HMMA/cyc/SM,
  ~25% of mma.sync peak), issue 32%. With smem latency, MMA chains and div/mod (s51) ruled out, what's left of the consumer
  time is the per-stage skeleton (~0.4 us) + run ends + x arrival (s50/s51), and low TLP (2 consumer warps per
  scheduler; more warps were slower in s14/s34). The 00:00Z H200 diag19d decomposition decides whether anything is left.

## s53 (2026-10-26 21:47-22:20Z, sandbox okc-night4, session 10): gemv24 = per-warp TMA rings: WIN in the H100 layer (handoff)
- No judge runs (proj checks 20/20 for 10-26, reset 00:00Z). Lead 20:16Z: ONE bigger item for the layer's T>=16 GEMMs.
- Idea: g19's consumer cost at m16/m32 includes the block-wide producer -> consumers -> empty-barrier handoff per stage
  (s12/s50/s51: ~0.3-0.5 us per stage, the slowest warp gates the ring). dev/s53/gemv24.cu: NO producer warp. Each warp
  owns split-K tasks (16 rows x KC k = KC/32 atoms, contiguous = pack19 with rg 1) and streams them through its OWN
  NS-slot ring (lane 0 issues one bulk copy per slot and refills the slot right after reading it; fence.proxy.async
  before the refill). Block b owns tasks [T b/G, T (b+1)/G) statically (t = ks * n/16 + row group), warp w every WPB-th;
  the block's <= XB x chunks are bulk-loaded once after griddepcontrol.wait; out zeroed in the kernel + epoch-counter
  barrier before the first red (as g19 mode 2); task end: escape fixups (g19's register ballot path, headers __ldg'd at
  task start), partials -> per-warp smem tile -> red.global.add.v4. W prologue (npre slots) before the wait; npre bit 16
  = launch_dependents at start (layer convention). Template R (row groups per task, x fragment reuse) exists: R 2/4 LOSE
  (fewer warps fit; H100 +1..8 us). Harness dev/s53/g24.py "M,KC,AP,NS,WPB,XB[,R]" (env GRID, PRED, NPRE, DEFS
  G24_NOW / G24_NOFIX, ESCW=1 adversarial escapes + exact spike rows, 20 reps: PASS all shapes/configs).
- H100 after a 3 us PDL pred (us, g24 | g19 layer-like NOTM+TAG2, both with escapes), G132: pool m16 18.6 | 19.1,
  m32 21.1 | 21.6, unp m16 18.8 | 19.7, m32 20.2 | 21.2. Consumer-only (NOW, no escapes): pool m16 10.1 | 11.8, m32
  14.5 | 15.0, unp m16 9.8 | 12.3, m32 14.5 | 14.5, qkv m16 8.7 | 9.0, m32 11.7 | 12.4.
  H200 proxy G88 (best cfg each): qkv m16 11.7 | 11.8, m32 14.7 | 15.7; o m16 6.7 | 7.2, m32 8.1 | 8.7 (cuBLAS 8.4);
  pool m16 19.6 | 19.6, m32 22.3 | 23.0; unp m16 19.8 | 19.8, m32 22.3 | 22.8.
  G72: qkv m16 11.8 | 11.6, m32 16.1 | 16.4; o m16 7.1 | 7.5, m32 8.7 | 9.9; pool m16 20.4 | 19.1 (loses), m32 24.8 | 26.6;
  unp m16 20.3 | 20.6, m32 23.9 | 24.7. Best cfgs: m16 (16,1024,2,4,16,2) / o (16,512,2,4,16,2); m32 (32,512,4,3,12,2) or
  (32,512,2,4,16,2). m32 KC 1024 doesn't fit with enough warps (x 64 KB x 2).
- Layer: dev/s53/g24_layer.patch (on r-layer b889e7c): g24_src.cu (= gemv24.cu) inlined by build.py ("24"), G24_CFG,
  _pack24/_g24_prepare (separate load_inline, per-pick F.linear check, inside _g19_prepare), _g19() takes TUNE option
  g24:<key> first (same grid G19_GRID, same trig), TUNE[16] += g24 pool/unp/both/o/qkv, TUNE[32] += same + pool+unp+o.
  Layer tests 13/13 with all g24 options forced (4 g24 calls per layer at T=16/32). H100 ab_flag, tuner off, 5 rounds:
  d16 sl 195.1 -> 193.2 (pool+unp) / 192.9 (all), d16 full 294.8 -> 293.6 / 291.2, d32 sl 194.1 -> 192.5 / 194.9,
  d32 full 413.5 -> 412.5 / 413.0. H100 is W-bound; the H200 (consumer-bound g19) should gain more.
- Submission: print-only _diag24 (DIAG24: pool/unp m16 x2 cfgs + m32 x2, qkv m16 x2 + m32, o m16/m32; grids 132/108;
  +~1 min compile) beside _diag19d for the 00:00Z check. bench.py test 39/39, esc PASS.

## s54 (2026-10-26 22:09-22:45Z, sandbox okc-night4, session 11): gemv24 SK (stream-K over ring stages): H100 layer d32 WIN
- No judge runs (proj checks 20/20 for 10-26, reset 00:00Z).
- G24_TRACE (dev/s53/gemv24.cu + g24.py DEFS=G24_TRACE): per-warp globaltimer start/end after the wait, per-block ends.
  H100 G132 static g24: m32 KC512 (2.6 tasks/warp) warps do 2 or 3 tasks: ends 10-12 us (2 tasks) vs 17-20 (3 tasks),
  intra-block spread ~6 us; m16 KC1024 nq 0-1 (qkv: only ~27% of warps have a task). Coarse task quantization.
- SK (template RR = R | 256; host decodes rr >> 8): block b owns global stages [SG b/G, SG (b+1)/G), warp w an equal
  contiguous 1/WPB of them, flushing partials at every task boundary it crosses; escape fixups restricted to the segment's
  k range [ks KC + st0 AP 32, ks KC + st1 AP 32) (bug found by the probe row: x offset must stay col - ks KC). ESCW 20 reps
  PASS, bench test 39/39, esc PASS. With SK the work is balanced (10-11 stages/warp) but ends still spread 11-20 us
  (per-warp HBM share variance, not quantization).
- DEAD: DYN (block-local smem chunk claims, lane 0 claims at issue time, per-warp fifo, merges contiguous chunks):
  C=1/2 +5..25 us (flush per chunk), even C=SPK (no extra flushes) +4 us vs static. Code dropped (not committed).
- Standalone after a 3 us pred (us, SK | static, best cfg each), H200 proxy G88 x2 runs: qkv m16 11.4 | 11.8, m32 13.9 |
  14.9; o m16 7.2 | 6.7, m32 9.0 | 8.2 (o loses); pool m16 20.3 | 19.5, m32 21.1 | 21.5; unp m16 20.9 | 19.9, m32 21.3 | 22.1.
  bench diag24 (H100) G108: pool m16 (16,1024,2,4,16) 21.5 static | 19.5 SK, (16,1024,4,3,12) 19.1 static; unp m16 23.4 | 20.1.
- Layer (copy of r-layer working tree incl. layer's uncommitted g24 s65 in /tmp/pj54/L; tools/ab_flag.py TUNE {}, 5 rounds
  x seeds 99,7): T=32 g24sk pool+unp: d32 sl -1.3/-1.9, full -1.5/-1.1 us (WIN); g24sk qkv at T=32 +1..1.5; T=16 all
  g24sk +0.4..0.9 (lose on H100). Layer tests 13/13 with g24sk forced at T=16/32 (pool/unp/qkv).
  ab_files T=16 pooler cfg (16,1024,4,3,12,2,1) vs layer's (16,1024,2,4,16,2,1): sl -0.8, full tie; + unpool too: sl -0.6,
  full -0.3: small, left to the H200 diag24 G108 read.
- Handoff dev/s54/g24sk_layer.patch (applies on r-layer working tree as of 22:30Z incl. uncommitted g24): new g24_src.cu
  (same kernel for r=1), G24_SKCFG, state["g24sk"], TUNE option g24sk:<key>, TUNE_BASE T=32 g24sk pool+unp ON, TUNE[16]
  += 3 g24sk sets, TUNE[32] += g24sk off/qkv sets.
- Submission: DIAG24 += SK twins (r 257) of every config (print-only, for the 00:00Z check).

## s55 (2026-10-26 22:32-23:15Z, sandbox okc-night4, session 12): gemv24 tail balancing: dynamic tail + L2 lend-prefetch DEAD
- No judge runs (proj checks 20/20 for 10-26, reset 00:00Z).
- Motivation (G24_TRACE, H100 G132 SK): warp ends q0/50/90/100 ~11.5/17.2/18.5/20.2 us at pool m32 (same work per warp),
  block-end med 18.5 / max 20.2, intra-block spread ~3 us: up to ~2 us/call if perfectly balanced.
- dev/s55/gemv25.cu (= gemv24 + new kernel gemm_dt + g25_gemm) and harness dev/s55/g25.py (cfg 7th field "dPT:C" =
  gemv25 with PT pool tasks per ks and C-stage chunks; 257 = g24 SK; 769 = g24 SK + PF). ESCW PASS for gemm_dt.
  * gemm_dt: SK over the static part; the last PT row-group tasks of each ks = a global pool of C-stage chunks
    (ctr[4 + ks], reset by block 0 before its epoch arrive), claimed by atomicAdd after the static range; ring sequence
    data-driven (lane 0 writes the slot's stage index / -1 into smem meta before arrive.expect_tx), flush on task change.
    H100 G132 pool m32: d0 22.35 vs SK 23.02 (tie), d64:4 26.8 (+4: 4-stage chunks = ~6 us of a warp, warps ended with 7-16
    stages). v2 (first claim NS stages ahead, pre-claim the next, no blocking fallback ks): H200 proxy G88 still loses:
    pool m32 SK 21.1 | d0 23.2 | d16:1 24.4 | d32:1 25.0 | d64:1 26.8; qkv m32 14.5 | 15.4 | 17.1 | 17.5 | 21.7;
    m16 d16:2 / d32:4 +3.7..4.9. The stage-driven loop alone costs +1-2 us consumer-bound; each claim/flush more. DEAD.
  * PF (g24 SK, RR bit 9): a warp that finishes L2-prefetches (cp.async.bulk.prefetch.L2, once per warp) the unissued rest
    of each block-mate's range. H100 G132 +0.5..1.0 us (intra-block spread 2.85 -> 2.14 but median end 17.2 -> 18.0:
    H100 is HBM-bound, the prefetch only moves bytes earlier), G88 proxy +0.5..1.1 everywhere. DEAD.
- Lesson: in g24 the per-warp end spread is per-warp HBM share, and any rebalancing that adds claims, flushes or bytes in
  flight in the main phase costs more than the ~2 us tail it targets. Static SK / static r1 stay (tuner picks).
- More s55 probes (all in dev/s55/gemv25.cu's g24 kernel behind defines, off by default; harness dev/s55/g25.py,
  env NOCHK=1 skips the result check for timing-only probes). H200 proxy G88 and H100 G132, alternated A B A B:
  * m32 KC 1024 XB 1 (half the FP32 split-K red traffic; grid multiple of k/KC so SK block ranges align): tie
    (unp G132 20.0-20.2 vs layer cfg 20.1-20.2; pool G104/G112 23.0-24.0 vs 23.3; G108 does not align). DEAD.
  * G24_XPRE (x bulk-loaded BEFORE griddepcontrol.wait; timing upper bound only, not legal): m32 -0.4..-1.4, m16 -0.2..-0.6 us
    = the whole post-wait x arrival cost. G24_XPAR (x row copies issued by warp 0's 32 lanes instead of thread 0): tie. DEAD.
  * G24_ZPROBE (no out zeroing / no fence; timing only): -0.4..-1.7 us (qkv m32, pool m32 most). G24_LZ = lazy zero: plain
    zero stores + __syncthreads, fence + epoch arrive per warp after its first stage: v1 per-warp global atomics (2112 on one
    address; epoch in units of 4096/launch) +1..2 us; v2 (LZ2, on file) block-aggregated (smem count, last warp does the one
    global atomic, spinners read target from smem) +0.4..1.5 us. DEAD. (v1 first HUNG in the harness: the epoch target
    old % (G*WPB) breaks when configs with different WPB share a ctr; any per-launch-unit epoch needs a fixed unit.)

## s56 (2026-10-26 23:15-~00:30Z, sandbox okc-night4, session 13): g24 EZ/ZN (zero-ahead chain) for the layer at T=16
- dev/s56/diag_read.py: readout of a proj judge log's diag24/diag19d lines vs layer main's g24 picks (pool G108, rest G132).
- Idea (from s55's bound: zero+fence+epoch barrier 0.4-1.7 us/call): out arrives zeroed, the kernel skips zero, fence,
  epoch atomic and spin (EZ). Standalone dev/s56/gemv26.cu + g26.py (DEFS G24_EZ: the 3 us pred zeroes out; G24_ZN: the
  g24 zeroes a second m x n buffer at its warps' end; NOCHK=1 timing; ESCW PASS for EZ and EZ+ZN). Graph pred+gemm, us/call,
  2 rounds each, layer cfgs (r1 / r257): EZ alone G88 proxy -1.0..-1.5 on every op/m; H100 G132 -0.4..-1.4 (pool m32 SK
  +0.4). EZ+ZN (the chain element): G88 qkv/o/pool/unp m16 -0.5..-1.3, m32 qkv/o/pool -0.9..-1.5 but unp m32 +0.6..0.8;
  H100 G132 m16 -0.4..-1.4, unp m32 +0.9..1.5. ZN alone +0..0.3 (pool m32 r1 +0.6 at G88).
- Layer (copy /tmp/pj56/L of r-layer a2a2d41): kernel args zo/zn4 + npre bit 17 (EZ) / 18 (EZPF); _g24_sel() helper;
  G24_ZCHAIN qkv->o->pooler->unpool: the g24 of key K preallocates the next key's out (_G24_ZN, cleared at kernel_fn start)
  and zeroes it at its warps' end; the consumer pops it and runs EZ. dev/s56/ezcalls.py confirms: d16 qkv(ZN o) ->
  o(EZ, ZN pooler) -> pooler(EZ, ZN unpool) -> unpool(EZ); d32 pooler -> unpool; T=128/256 untouched.
- Safety (needs completion transitivity between the zeroing g24 and its consumer): d16 sl qkv->csl(PW, waits; PDL attr
  only with PW)->o; d16 full qkv->_gd_full (no launch_pdl = full edge)->_red_fin (waits)->o; o->_boundary (waits on SEP and
  non-SEP paths when PDL)->pooler; pooler->_rmsnorm_gain (waits when PDL)->event join->tk2f->moe_norm2->unpool. OK.
  Tests 13/13, tools/replay_test.py d16 40 inputs x seeds 11,7: 0/160 fails.
- H100 ab_flag (variants TUNE {}, 4-5 rounds x seeds 7,11,99), us vs off: chain on at T=16: d16 sl -0.3..-1.7 (3 runs,
  mean ~-1.1), d16 full -0.1..+0.2 (tie). T=32: d32 sl/full +2..+4 (LOSES): ZN stores on the pooler +0.6..1.5 and the EZ
  unpool +0.5..1 on top (not L2 eviction: EZPF L2 prefetch of out before the wait = tie). Short chain (no pooler->unpool)
  at d16: -0.3..-1.2 sl (less than the full chain). A/A noise floor of ab_flag here +-0.5 us.
  NB ab_flag stmts persist across variants in one namespace: "pass" after "X = set()" is an A/A run (my first try).
- Handoff dev/s56/g24z_layer.patch (git apply in r-layer a2a2d41): G24Z_TOKENS {16} on, TUNE[16] += {"g24z": False}.
- 00:00Z check 636c89fcd45e49a5 (a3c49ba = main with DIAG19D + DIAG24 incl. s54 SK twins): FAILED setup, "prepare() took
  longer than 300 s" (325.6 s). Not counted on the board; all diag lines printed. LESSON: the 300 s prepare limit; the SK
  twins doubled diag24. Main now has DIAG19D/DIAG24 off (ca7e557, local prepare 1.6 s).
- H200 diag24 (us after a 3 us PDL pred; dev/s56/diag_read.py; layer cfgs: pool G108, rest G132):
  pool m16: layer (2,4,16) G108 18.34 | (4,3,12) G108 15.36 | best SK (4,3,12) G132 15.33. pool m32: layer SK (2,4,16) G108
  21.15 | SK (4,3,12) G108 20.00 | static (2,4,16) G132 18.69. qkv m16: layer static 10.53 | SK 9.89 (G132/G108).
  qkv m32: SK G132 13.15 (diag19 lt pick 13.92). unp m16 15.58 = best (tie SK 4,3,12 15.55); unp m32 layer SK 17.38 |
  (4,3,12) r1 17.15. o m16 layer 6.34 = best (lt pick 6.30); o m32 SK 7.62 / r1 7.68.
  H200 diag19d (g19, us): TAG2 vs NOTM pool m32 -0.42/-0.13, m16 +0.03/-0.35, unp m32 +0.39/+0.20: noise, TAG2 DEAD.
  H200 pool m32 G132: main 19.17, NOW 16.26, NOW NOMMA 10.59 (consumer/decode bound as before).
  Standalone tune lines (cold, no pred): m16 pooler lt 22.21, qkv lt 16.93, unp g16 20.70; m32 pool lt 22.21, unp lt 22.37.
- Layer H100 ab_files (A = r-layer worktree 00:21Z incl. g24z, B = pooler T=16 (4,3,12) + T=32 SK (4,3,12); A B B A,
  4 rounds, TUNE {}): d16 sl -0.9, full -0.4; d32 sl +1.7, full +1.2. qkv T=16 SK (ab_flag): +0..0.6 on H100.
  -> handoff dev/s56/g24cfg_h200_layer.patch = pooler T=16 (4,3,12) only (applies on r-layer worktree, tests 13/13).
  NB ab_flag cannot change G24_CFG (packed at prepare): use ab_files for config changes.
- Check ed67748e9a684fe7 (ca7e557 = main, diags off): PASSED 1.3277x (3638.4 us), > board best 1.3110 (2480c53). Per-shape
  speedups: m1 qkv 1.41 o 1.08 pool 1.45 unp 1.84 router 2.92; m8 1.24/1.02/1.29/1.37/5.11; m16 1.11/1.07/1.15/1.09/2.77;
  m32 1.10/1.04/1.15/0.99/2.49; m128 1.14/1.04/1.23/1.01/1.84; m256 qkv 1.08 o 1.05 pool 1.12 unp 1.04.
- Submit 1ff4721e87c844c2 (ca7e557, same code as ed67748e): 1.3233x (3650.7 us), NEW BOARD BEST (was 1.3110), proj board #252. Budget 10-27: checks 2/20, scored 1/10.

## s57 (2026-10-27 01:15-02:25Z, sandbox okc-night4, session 14): gemv27 = no cross-block split-K o GEMM for the layer's T=16
- Standalone first-op qkv m32 g24 (H100, no pred; dev/s53/g24.py now takes PRED=-1): (4,3,12) r1 20.0-20.5 vs cuBLAS
  20.7-20.9 graph-us; NOW (no W) 15.5: consumer-bound as first op. The layer already has g24:qkv_gate T=32 (4,3,12) as an
  option (tuner never picked it): nothing new there.
- dev/s57/gemv27.cu + harness dev/s57/g27.py ("M,KQ,AP,NS,RGB[,OBF]"; env PRED (<0 = none), ESCW, G24=<g24 cfgs>, DEFS
  G27_TRACE / G27_XBULK / G27_XHALF / G27_XREP (probes), NOCHK). Block = RGB whole 16-row groups (all of k), WPB = RGB*KQ
  warps each own (row group, k/KQ slice) with a per-warp TMA ring (pack24 with kc = k/KQ, r 1); partials -> smem (aliasing
  the warp's ring) -> fixed-order block sum -> plain FP32/BF16 stores. No out zeroing, no fence, no epoch barrier, no
  atomics; deterministic; subnormal products kept (g24's red.add flushes them: the spike test accepts either).
  ns >= 100 = PD (gemm_pd): the warp's whole slice (ns-100 atoms) streamed AND decoded into registers before
  griddepcontrol.wait; after the wait only x (cp.async 16 B by all threads) + ldmatrix/mma + sum. ZN arg (zo) like g24.
- H100 after a 3 us PDL pred (us): o m16 PD 4.8-5.4 | non-PD 4,4,2,4 5.2-5.5 | g24 layer cfg 6.2-6.9 | cuBLAS 7.8;
  m32 PD 6.2-7.0 | g24 6.9-7.2 | cuBLAS 8.2-8.4. PRED 6000 same as 3000 (not W-bound); no pred: g27 loses to cuBLAS.
  unpool m16 (k 3072, non-PD): 20.1 vs g24 18.6: dead. KQ 8 / RGB 2 / NS 1 / AP 2 variants worse.
- Trace (o m16, H100, 0.256 us timer): x arrival after the wait ~1.8 us (m32 2.3-2.8), then ldsm+mma ~0.5-1, store 0.5.
  x arrival is NOT the bulk-copy issue (cp.async by all threads = same) but bytes: XHALF probe 1.8 -> 1.3 (m16), 2.8 -> 1.8
  (m32): ~1 us fixed + ~0.5 us per 16 KB/block with 128 blocks reading the same x. Untried: cluster TMA multicast of x.
- Check c7b586838394472b (fbec9e1, print-only _diag27; picks = ca7e557): 1.3190x (not a beat: board 1.3233). H200 diag27
  (us over the 3 us pred): o m16 cuBLAS 6.75 | lt pick 7.20 | g24 (512,2,4,16,2,1) 6.43 | g27 PD 5.25 | g27 non-PD 5.82;
  m32 cuBLAS 6.91 | F pick 6.43 | g24 (512,4,3,12,2,1) 7.17 | PD 6.69 | non-PD 7.30.
- Layer copy (/tmp/pj57/L = r-layer 66fda6e): G27_CFG {("o",16): (16,4,4,108,4,0)}, G27_ON, _g27_prepare (own load_inline,
  checked vs F.linear), _g27_sel (_g24_sel returns None for a g27 key, so qkv's ZN skips o), _g19 runs g27 first (pops any
  _G24_ZN for o, zeroes pooler's out as ZN), O_PF prefetches the g27 pack, TUNE[16] += {"g27:o": False}.
  Tests 13/13; replay_test d16 + d32 (g27 forced at T=32 too) seeds 11,23 0/160. ab_flag TUNE {} 4 rounds x seeds 7,11,99
  (A B A): d16 sl 157.24/152.34/189.93 vs g24 158.06/153.20/190.94 (-0.8/-0.9/-1.0), full -0.1/0/-0.2. T=32 o g27 +1.1..1.8
  (3 rounds x 7,11): T=32 left out. Handoff: dev/s57/g27_layer.patch (git apply on origin/main v105 f9b702a/66fda6e: clean).

## s58 (2026-10-27 02:05-03:00Z, sandbox okc-night4, session 15): g27 x loading (MC), main decl fix
- BUG found + fixed: 0cfca9f (s57's last commit, never judged) ran a regex that appended ", torch::Tensor" to 14 cpp_sources
  declarations in dev/sub_template.py (g10/g12/g13/g14/g15/g16/g19/g19_bf16/g20/g22/g24/...), not only g27_gemm: the main
  native libs (muse_proj_v14, g19a) failed to load ("undefined symbol") -> silent torch fallback. fbec9e1 (judged) was fine.
  Fixed in c2b541b (only g27_gemm keeps its zo Tensor). LESSON: after any template edit grep the prepare output for
  "build failed"; local runs must use a fresh TORCH_EXTENSIONS_DIR (/tmp/...) when sources change but names do not.
- dev/s58/gemv27.cu: G27_MC (default 1). 0 = old cp.async xload; 1 = x as cp.async.bulk copies, one per (row) [plain] or
  (row, k slice) [PD], spread over all threads, barriers armed (expect_tx) before griddepcontrol.wait; >1 = cluster of
  G27_MC CTAs, each issues 1/C of the rows with .multicast::cluster into all (cluster barrier before the wait).
- H100 standalone (dev/s58/g27.py, o, us over 3 us pred, 2-3 runs): m16 PD MC0 5.12-5.25 | MC1 5.12-5.26 | MC2 5.3-5.9;
  m32 PD MC0 7.49-7.52 | MC1 6.99-7.20 | MC2 7.25-7.58; non-PD m32 MC0 8.0-8.1 | MC1 7.55 | MC2 6.96-7.10. MC4 bad (+3..4).
- Trace (MC0/1/2): ~28 of 128 blocks start only when the pred's blocks exit (PD = 127 regs x 512 thr = whole RF: no
  co-residency); their W stream + decode after the wait = the tail. Lower-footprint PD shapes all lose: RGB 2 (256 blocks,
  x read twice: m16 8.2 us!), KQ 8 (1024 thr) 6.1. Same-x reads by all blocks are expensive (+4 MB x = +3 us).
- Layer H100 (/tmp/pj57/L copy of r-layer 66fda6e + s57 g27; ab_files, TUNE {} in all files, 8 rounds, seed 99):
  d16 sl MC0 190.26 | MC1 188.45 | MC2 189.01; d16 full 291.41 | 290.36 | 291.92 -> MC1 -1.8 / -1.05. (With the tuner on:
  -1.24/-0.25.) T=32 o via g27 MC1 vs g24: sl 192.24 vs 192.14, full 411.23 vs 411.91 = tie (s57 MC0 was +1.1..1.8).
  Layer copy /tmp/pj58/L (g27_src.cu = s58 file): bench test 13/13, replay_test d16 40 x seeds 11,23 0/160.
  Handoff dev/s58/g27mc_layer.patch (s57 + s58 on r-layer 5637814; or cp the .cu if s57 patch already applied).
- Local cold prepare (fresh TORCH_EXTENSIONS_DIR) with diag27 (2 g27 modules, PD cfgs only): ~200 s wall on the H100.
- H200 check a4962d70da9d40b4 (c2b541b, diag27 MC1 vs MC0; picks unchanged): 1.3116x (CI 1.29-1.33; not a beat of 1.3233).
  diag27 (us over a 3 us pred): o m16 PD mc1 4.42 | mc0 5.28 | g24 (2,4,16) 6.59 | lt pick 6.34 | cuBLAS 6.82;
  o m32 PD mc1 6.27 | mc0 6.88 | g24 (4,3,12) 7.33 | lt pick 6.72 | cuBLAS 6.98. MC1 = -0.86/-0.61 us on H200.
  Log: profiles/submission.a4962d70da9d40b4.json. Diag off again in main (DIAG27 = {}).
- Layer handoff dev/s58/g27mc_inc_layer.patch on r-layer d4fd834 (which already has s57 g27): g27_src.cu = s58 file,
  G27_CFG += ("o", 32), TUNE[32] += {"g27:o": True}. On d4fd834: tests 13/13, replay d16 0/40; d32 forced 0/160 (pj58/L).
- g24 XSP (dev/s58/gemv24.cu G24_XSP=1: x barriers armed pre-wait, x copies spread over threads instead of thread 0
  after the wait): dev/s58/g24.py H100, us: m16 qkv/pool/unp -0.25/-0.1/-0.06..+0.13 (tie); m32 (512,4,3,12,2) +0.13..
  +0.4 (worse; 2 runs). DEAD: g24 is W-stream-bound and its per-block x chunk is small. (Default in the s58 copy is 1;
  dev/s53/gemv24.cu, the one used everywhere, is unchanged.)

## s59 (2026-10-27 03:02-03:50Z, sandbox okc-night4, session 16): G24X = cross-call EZ for the layer's first-op qkv g24
- Why qkv (the layer graph's first op, T=16/32) is slow, H100 standalone, no pred (dev/s59/g24.py PRED=-1; us/graph):
  bare ld.global.v4 reader of qkv's 28.3 MB 12-bit bytes (dev/s59/rd.py) 11.7-12.4 | g24 layer cfg m16 18.5-19.1,
  m32 20.5-21.1 | cuBLAS 20.6-20.8 | consumer-only (G24_NOW) m16 14.5-14.8. Trace with t0 at kernel ENTRY (G24_TR_ENTRY):
  m16 (1024,2,4,16,2): 1152 tasks / 132 blocks = ~9 busy warps of 16; idle warps finish their setup (x issue, zero,
  __threadfence, epoch atomic) at 5.3 us (NOW 2.7): the setup's dependent memory round trips run behind the W stream.
  G24_NOZ (skip zero + fence + epoch, timing only) = -1.3 (m16) / -1.9 (m32) us. Deeper rings (NS 8-12, AP 4, WPB 8-12)
  and SK: no gain (not in-flight-bound; consumer ~= stream time, they add).
- G24X (layer patch dev/s59/g24x_layer.patch, on r-layer 85b5641 = v108): qkv's FP32 out at T in {16, 32} is a
  persistent zeroed buffer state["g24x"][T]; qkv g24 runs EZ on it (npre bit 17), the same call's g24 unpool zeroes it as
  its (free) ZN; if unpool is not g24 in that call, kernel_fn (now a wrapper around _kernel_fn) zero_()s it at the end.
  Invariant: zero between calls (nothing data-derived is kept; like the self-resetting counters). qkv's out is dead after
  attention (verified: qkv writes the buffer, unpool re-zeroes it, buffer 0 after every call). T=32 acts only if the
  tuner picks g24:qkv_gate (existing TUNE[32] option); TUNE[16] += {"g24x": False}.
- H100 layer ab_files TUNE {} A B B A: on 66fda6e-era main (8 rounds) d16 sl -1.56 full -0.68; on v108 (6 rounds)
  d16 sl 189.97 -> 188.11 (-1.86), full 291.21 -> 290.59 (-0.62). T=32 with g24 qkv forced: G24X -0.5 sl / -0.9 full vs
  g24 qkv without it, vs QSS default +0.4 / -1.2 (tuner option). Tests 13/13; replay d16 40 x seeds 11,23 0/160;
  g24 qkv forced at T=32 + g24sk qkv at T=16: d1/d16/d32 0/40 each.

## s60 (2026-10-27 03:30-04:20Z, sandbox okc-night4, session 17): wgmma for the layer's g24 m16/m32 = DEAD
- Why: layer H200 profiles (r-layer profiles/submission.7fda724c / 71f451bb): d32 g24 SK pool+unp 47-49 us for 2 calls
  (~24/call vs 12-bit floor ~11), d16 g24 x2 ~36 + pooler 18; H200 diag19d (s56): g19 pool m32 NOW 16.3 vs NOW NOMMA 10.6
  = the mma.sync path is ~1/3 of the consumer. Idea: warpgroup wgmma (A = decoded W from registers, RS) cuts tensor time.
- dev/s60/gemv24.cu = r-layer g24_src.cu (85b5641) + WG mode (RR bit 9 = 513): task = 4 row groups x KC per warpgroup
  (warp w % 4 = its row group, same r1 pack as the layer), x copied after the wait with __ldcg into the wgmma B
  core-matrix layout ((kk * M + q) * 16, LBO = M * 16, SBO = 128, no swizzle), wgmma m64n{16,32}k16 RS, accumulator and
  A-fragment layouts per warp = mma.m16n8k16's (epilogue + escape fixups unchanged). CORRECT (dev/s60/g24.py asserts:
  rel err < 1e-5 + exact escape probe), pool/unp/qkv m16/m32. Switches: G24_WG_W1 (3 A register sets, wait_group 1),
  G24_WG_A2 (2 accumulator chains), G24_WG_NOMMA / G24_WG_NOX / G24_WG_SS (timing probes, wrong results).
- H100, us over a 3 us PDL pred, G132 (dev/s60/g24.py; r1 = layer's mma.sync g24 with the same cfg):
  full: pool m32 WG 21.7-22.5 | r1 21.5; m16 20.1-20.4 | r1 ~20.4 (W-bound tie). G88 proxy: pool m16 WG 24.6 | r1 20.5 (loses).
  consumer-only (G24_NOW): pool m16 WG 13.97 (1 A set) / 13.5 (2 sets) / 13.2 (W1) / 13.3 (A2) / 12.0 (AP 4) | r1 11.2;
  m32 16.6 / 16.3 / 15.7 / 15.9 / 15.9 | r1 15.2; unp same picture. WG NOMMA 8.9 / 10.4 -> wgmma adds 4.4-5.5 us.
  SS wgmma (A from smem, probe) worse: 15.1 / 17.7. => at N = 16/32 wgmma (RS or SS) is NOT faster than mma.sync here.
- Trace (G24_TRACE, NOW, pool m32 r1): warps doing 1 task end at ~5.6 us, 2 tasks at ~11.3: per 16 x 512 task ~5.6 us =
  decode/skeleton ~2.4 + mma ~2.8-3.1, ADDITIVE (no overlap across the 4 warps/SMSP). mma.sync rate there ~1 per 10 clk
  per SMSP = saturated: pool m32 tensor time alone ~5.6 us (H100) per SM. The m32 consumer needs decode to overlap mma
  (mma.sync seems to hold issue), not a different MMA instruction.
- Compiler trap found (keep in mind for any wgmma code): with two accumulator arrays passed by reference to asm "+f"
  wgmma wrappers, nvcc DROPPED the final `acc += acc2` (PTX had 8 wgmmas, acc2 results never read; ptxas then deleted
  those wgmmas: SASS 4 HGMMA, rel err 0.707, while NOW timings looked faster = bogus). Fix: one local float aw[2][NT][4]
  + explicit sum after the last wait, plus empty-asm "+f" operand fences after every wait_group (CUTLASS style).
- Also dead (same harness): g24 R=2 at m32 (2 row groups per warp share x fragments): NOW 15.3 (8 warps) vs r1 15.2,
  full 22.7 vs 21.5.
- No judge runs this session (proj checks 4/20 used today). Nothing for layer.

## s61 (2026-10-27 04:05-05:20Z, sandbox okc-night4, session 18): where the g24 m32 consumer's time goes (clock64 split)
- ncu (G24_NOW pool m32 r1 (512,4,3,12,2) G132): issue slots 30% busy, 2.75 active / 0.53 eligible warps per scheduler,
  tensor pipe 22%: LATENCY-bound (top stalls wait 1.6, short_scoreboard 1.2, membar 0.7 per issue). Loop body is well
  scheduled already (G24_G=1/2/4 = atoms whose smem loads go together: all tie). WPB 16 at m32: NOW -0.6 but full +0.6 us.
- dev/s61/gemv24.cu (= s60 file + below) + dev/s61/g24.py (env EZ / XSP / DF bits; DEFS=G24_CLK prints per-warp clock64:
  stage loops, task ends (fix = escape fixups, bar = epoch spin), first-x wait, setup (wait -> after epoch arrive), pass@).
  H100 pool m32 full G132, kcyc median: setup 10.4 (zero + __threadfence + atomic, all warps in __syncthreads), loop 16,
  ends 8.3 (fix 1.7, bar spin 5.0), barrier seen passed at 22 kcyc after griddepcontrol.wait although every block arrived by
  11.4: dependent round trips (fence, atomic, ld.acquire spin) behind the saturated W stream cost ~3 us each.
  NOW: setup 4.1, bar 0.7, fix 1.8 (fixups ~0.7 kcyc per task end = ~7% of a warp).
- EZ probe (bit 17, no zeroing; timing only): setup -> 0.1 but the first x wait becomes 9.3 kcyc (G132) / 6.3 (G88):
  x arrives ~3.2 us after the wait REGARDLESS of W (NPRE 0: 5.5 kcyc; PRED 10 us: 6.1): thread 0's 64 serial bulk issues.
  XSP (x copies spread over threads, s58 idea): x wait 6.3 -> 2.0 kcyc, but the stage loop grows by the same (W-bound).
- Kernel times: G132 EZ upper bound m16 -0.6..-1.0, m32 0; G88 proxy pool m32 -1.65, unp m16 -0.9, pool m16 -0.3, unp m32 0.
  XSP / DF (bit 20: keep a task's scaled partials in registers while the barrier is pending, flush with the next) all tie
  on the H100 (G132 and G88 W-bound). Correct with ESCW adversarial escapes (all 4 combos, qkv/o/pool/unp m16/m32).
- => only the H200 (consumer-bound) can judge them: print-only _diag61 in submission (layer cfgs + grids, variants base |
  xsp | df | xsp+df | EZprobe).
- H200 check e25a1d4d199245ed (079f7b4, diag61 print-only): 1.2925x (not a beat of 1.3233; no submit). diag61 (us over a
  3 us pred; base | xsp | df | xsp+df | EZprobe): pool m16 (4,3,12) G108 15.14 | 15.17 | 15.01 | 15.14 | 14.30;
  unp m16 (2,4,16) G132 14.72 | 15.62 | 14.98 | 14.66 | 13.70; pool m32 (4,3,12) G108 20.99 | 21.18 | 21.38 | 21.54 | 20.03;
  pool m32 SK (2,4,16) G108 20.86 | 20.74 | 20.86 | 20.61 | 19.33; unp m32 (4,3,12) G132 16.93 | 17.47 | 16.93 | 16.77 | 17.02;
  unp m32 SK 17.28 | 17.31 | 17.50 | 17.34 | 17.18.
  => XSP and DF DEAD on the H200 too (ties +-0.3, xsp worse at unp). Removing zero/fence/epoch is worth at most -1.0..-1.5 us
  for the pooler at T=32 (unp m32 0): the layer's g24z chain (o zeroes pooler's out) at T=32 was judged on the H100 (where
  it cannot show); Request posted to layer: H200 TUNE option. Diag stays in main (small: cold local test 169 s).

## s62 (2026-10-27 04:45-05:20Z, sandbox okc-night4, session 19): layer's s72 HYBRID idea (BF16 + 12-bit rows in one g24) = DEAD
- Built: dev/s62/gemv24.cu = s61 file + HY mode (RR bit 10, r | 1024; runtime hyb = GB << 16 | NB): blocks [0, GB) stream
  row tiles [0, NB) as plain BF16 (pack: per atom [s][lane] uint4 = the mma A fragment, 2 ld.shared.v4, no decode/escapes,
  scale 1), blocks [GB, grid) the 12-bit rows NB*16.. (own pack + header). dev/s62/g24.py env HY="GB:NB;..." (packbf()).
  CORRECT: ESCW adversarial escapes, pool/unp m32 hy40:64 and hy20:32 (asserts pass). Smem: BF stage 4/3 of 12-bit, so
  (4,3,12) / (2,4,16) do not fit at m32; (4,2,12) does.
- H100, full (W-bound, BF dearer as expected): pool m32 r1 (4,2,12) 22.98 vs hy40:64 24.86, hy20:32 23.20; unp 21.60 vs 24.06/22.38.
- The decisive number is the CONSUMER (G24_NOW, what binds on the H200), G132 after a 3 us pred, pool:
  m32 base 15.18 | hy33:48 16.46, 33:64 15.57, 33:80 17.74, 99:192 15.44, 99:208 15.92, 99:224 17.50
  (minimum at rows share = blocks share, never below base: a BF16 row costs the SAME consumer time as a 12-bit row);
  m16 base 11.98 | 33:64 12.40, 33:80 12.11, 99:192 12.34, 99:208 11.54 (BF16 rows ~4-8% cheaper at most).
  unp m32 66:256 15.49 / 33:128 15.74 vs 15.07.
- => at m32 the g24 consumer is mma/ldsm/latency-bound, not decode-bound (consistent with s61 ncu: 30% issue, latency);
  BF16 rows add 33% bytes for no consumer saving. Model gain on H200 m16 <= ~0.7 us, likely 0: not worth a check. No judge
  runs this session (checks 5/20 today). Answered layer under Requests.

## s63 (2026-10-27 04:51Z-, sandbox okc-night4, session 20): pooler grid at T=32 (dead, known); G24P = pooler EZ at T=32 (TUNE option handoff)
- Re-read the s56 diag24 H200 lines: pool m32 G132 vs G108 standalone after a 3 us pred: static (2,4,16) 18.69 vs 21.50, SK
  19.04 vs 21.15 (-2.1..-2.8 us). In the layer it is DEAD: layer s18 (check ea9421fa, g19 era) H200 tuner T=32 grid
  108 232.6 / 120 243.8 / 132 245.7 us; H100 today (r-layer 445e18b copy, ab_flag 4 rounds) d32 sl 192.9 / 120 205.2 /
  132 203.5, full 407.1 / 421.0 / 419.7. The side-stream router_split (128 programs) + routing need the free SMs.
- H100 d32 sliding timeline (r-layer HEAD): routing ENDS AFTER the pooler (75.5 vs 74.3 us; rmsnorm starts at routing end):
  on the H100 the router, not the pooler, is critical at T=32. On the H200 router_split is ~6.5 us (layer NOTES s16), so
  the pooler is critical there: pooler-only gains can only be judged by the H200 tuner.
- G24P (dev/s63/g24p_layer.patch, applies to r-layer HEAD 445e18b and to its s73 worktree): the s61 Request built by proj.
  Pooler g24 at T=32 runs EZ on a persistent zero buffer state["g24p"][32] (no zero + __threadfence + epoch grid barrier);
  the call's g24 unpool zeroes it again as its ZN (its zo slot is free at T=32 unless G24X qkv is picked: G24X wins the
  slot, then kernel_fn's end zero_() restores the invariant). TUNE[32] += {g24p}, {g24p, tk2h 4}, {g24p, tk2h 4, lt:o};
  default off. Placed AFTER _SLIDING[0] is set (FYI layer: the G24X check in _kernel_fn runs BEFORE `_SLIDING[0] = ...`,
  so its _opt reads the previous call's sliding flag; harmless today since both T=16 tables keep g24x).
- Correct (forced variant): bench test 13/13, scored_test decode 32 x3 0 bad, replay_test decode 32 30 inputs x seeds
  11,23 0/120; dev/s63/g24p_inv.py: buffer absmax 0.0 after every call and the unpool (not the fallback) zeroed it.
  RMSnorm hides pure scale errors, so the invariant check matters more than the tolerance tests here.
- H100 (W-/router-bound, cannot show it): ab_flag 6 rounds x seeds 7,11 d32 sl +5.7/+3.2, full +1.7/+2.1 us; tuner lines
  g24p +1.5..2.4 vs default. H200 bound (diag61 EZprobe): pool m32 G108 -1.0..-1.5 us minus the unpool's 512 KB ZN tail.

## s64 (2026-10-27 05:05-06:15Z, sandbox okc-night4, session 21): gemv28 = x as wgmma A, W decoded into smem as wide-N B = DEAD (this form)
- Idea (new orientation; g20/g22/s60 all had W in registers as A with tokens as N): A = x TMA box [64 rows (OOB token rows
  zero-filled)][64 k] SW128, B = 12-bit W decoded by 2 decoder warpgroups into smem ([BN rows][64 k] BF16, SW128 K-major),
  wgmma m64nBNk16 SS by an MMA warpgroup (async, overlaps decode), WG3 = TMA producer; stream-K over (BN-row tile, 64-k)
  stages, red.global.add.v2 flush per run. dev/s64/gemv28.cu + g28.py "M,BN,NS,ND[,KA]" (pack = DEC3 units (row, 8 k),
  stage = [BN*8 x 8 B][BN*8 x 4 B], tile-major; no escapes yet: checked vs W' like s53). CORRECT first try (rel err < 1e-5).
- BN=256 does not compile at 512 threads (ptxas caps 128 regs; setmaxnreg does not lift the compile limit): needs 384 thr.
- H100 pool, us over a 3 us PDL pred (g28.py / g24.py back to back): m32 g28 (128,8,4) 20.94 | (128,6,3) 21.26 | KA2
  (128,3,2,2) 21.49 vs g24 layer cfg (512,2,4,16,2,257) 20.74; m16 g28 21.06 / KA2 19.76 vs g24 (1024,4,3,12,2) 18.46.
  Consumer-only (NOW): g28 m32 19.5-20.3, m16 20.5 vs g24 15.84 / 11.81 -> loses on the H200 (consumer-bound) too.
- Where the time goes (NOW pool m32 (128,8,4), ablations, us): full NOW 19.5 | no decode 17.0 | no wgmma 15.7 | neither
  14.3 | neither + no x TMA 12.5 | + no flush 12.9 (flush ~0.5) | empty kernel (return after init) 0.8. Trace (globaltimer,
  block 0): EVERY role advances ~0.33-0.36 us per stage regardless of NS/ND (4/8, 8/2, 8/4) and of try_wait vs test_wait
  (test_wait worse), proxy fence (none), wgmma fence/commit/wait (none: 13.1). MMA loop ALONE (no waits, other WGs exit,
  no setmaxnreg): 0.28 us/iter; + no wgmma ops 10.0 total; + no mbarrier arrives 6.0 => each mbarrier.arrive ~65 ns of
  issue-serialized time, and the 3-role handoff chain costs ~0.3 us per stage = about the H200 HBM time of one 12 KB stage
  per SM (~0.4 us). Bigger stages (KA=2) cut the handoffs per byte but did not help the full kernel (21.5).
- Traps: setmaxnreg.inc in a WG hangs if the other WGs exit without setmaxnreg.dec (G28_SOLO hang); ND=1 or NS=1
  deadlocks (MMA releases stage j-1 only after issuing j); now a host TORCH_CHECK.
- Verdict: dead in this form. A variant that might still work: ping-pong (each of 2 MMA warpgroups decodes its own B tile
  then issues async wgmma, no cross-WG mbarrier chain per stage; FA3-style) with KA>=2 and BN=256 at 384 threads.
  Not worth more before that skeleton is shown < ~0.1 us/stage in NOW mode on the H100.
- No judge runs this session.

## s65 (2026-10-27 06:15-07:10Z, sandbox okc-night4, session 22): g27 PD at M=8 (12-bit o at T=8 = v1) = DEAD in the layer
- Picked from the layer's H200 profile a0bcbbf4: v1 o is BF16 g12 (8.2 sl / 6.8 full us incl. its PDL wait); g27 PD had
  never been built for M=8. dev/s65/gemv27.cu = s58 file + NT == 1 in gemm_pd (one ldmatrix.x4 = both k16 halves of the
  atom's 8 x rows: xrow = (lane & 7) * XS + (lane >> 3) * 16). dev/s65/g27.py = s58 harness copy.
- Standalone H100, o (n 8192, k 1024), after a 3 us PDL pred, ESCW adversarial escapes all correct (us): m8 cuBLAS 7.84 |
  (4,4,108,4) 4.77 | (4,4,108,2) 8.59 | (4,4,108,8) 7.58 | (8,4,104,2) 8.42 | (8,4,104,4) 5.97; m16 (4,4,108,4) 6.08 vs 8.42.
- Layer copy /tmp/pj65/L (r-layer 6145e6d + g27_src.cu = s65 file + G27_CFG ("o", 8): (8,4,4,108,4,0)), ab_flag 4 rounds,
  seeds 99,7, G27_ON without/with ("o", 8): verify 1 sliding 141.56/115.87 -> 142.44/116.88 (+0.9/+1.0), full
  155.05/129.70 -> 158.85/132.70 (+3.8/+3.0); verify 16 untouched (tie). DEAD: the pre-wait W stream + whole-RF blocks
  hurt vtk / _attn_split+_combine more than o gains (same lesson as pr1 npf / PR1_BND). No handoff. No judge runs.
