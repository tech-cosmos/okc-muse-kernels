# okc: problem=muse_bf16_proj
"""Dense projections of Muse Spark Open: out = x @ W[:n].T (BF16; the router FP32).

Router (FP32 weights, FP32 output): up to 8 tokens a native true-FP32 split-K kernel (FP32 weights, FP32 FMA,
FP32 split partials), as SGLang's F.linear in FP32. Above 8 tokens SGLang's BF16 hi + lo split with FP32
accumulation: 9..32 tokens a native tensor-core kernel (dev/gemv6.cu: mma.sync with hi and lo tiles into one FP32
accumulator, split K reduced in FP32 over a thread-block cluster through distributed shared memory); above 32 one
cuBLAS GEMM against the stacked [hi; lo] weight (FP32 out) and an FP32 add of the two halves. BF16 projections: at 1 token a native grid-stride GEMV (dev/gemv10.cu: W read in place, row-major, the whole grid
sweeping one contiguous window; FP32 FMA and FP32 reductions, no split K), else at 8 tokens (pooler also 16) a native
tensor-core streaming GEMM (dev/gemv12.cu: mma.sync with x as A and W, repacked losslessly into 512-B fragment atoms, as B;
FP32 accumulators; split K reduced in FP32), else cuBLAS (as SGLang's bf16_gemm_dispatch on H200).
Native kernels are built once at import (load_inline); if the build fails everything falls back to torch.
"""

from typing import TypedDict

import torch
import torch.nn.functional as F


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [m, k]
    config: Config


_ROUTER_SRC = r'''@@ROUTER@@'''
_GEMV_SRC = r'''@@GEMV7@@'''
_G10_SRC = r'''@@GEMV10@@'''
_G12_SRC = r'''@@GEMV12@@'''
_G13_SRC = r'''@@GEMV13@@'''
_G14_SRC = r'''@@GEMV14@@'''
_G15_SRC = r'''@@GEMV15@@'''
_G16_SRC = r'''@@GEMV16@@'''
_G19_SRC = r'''@@GEMV19@@'''
_G24_SRC = r'''@@GEMV24@@'''
_G27_SRC = r'''@@GEMV27@@'''
_G24S61_SRC = r'''@@GEMV24S61@@'''
_G20_SRC = r'''@@GEMV20@@'''
_G22_SRC = r'''@@GEMV22@@'''
_TG2_SRC = r'''@@TG2@@'''
_G17_SRC = r'''@@GEMV17@@'''
_G21_SRC = r'''@@GEMV21@@'''
_R3_SRC = r'''@@ROUTER3@@'''
_LT_SRC = r'''@@LT@@'''
_G30_SRC = r'''@@G30@@'''

# Router FP32 kernel configs (mp, rows per block, threads), dev/router2.cu, chosen on the H100 (dev/r2.py).
ROUTER_F32 = [(1, (1, 2, 512)), (8, (8, 2, 512))]
# Router hi/lo one-kernel split-K GEMM (dev/router3.cu: tiles MT x NT, WM x WN warps, KC, stages), chosen on the H100
# (dev/r3.py): m9-16 6.9 (g7 8.5-8.9), m32 7.2-7.7 (g7 10.3), m128 9.6-10.9 (cuBLAS 3 kernels 12.2-12.6); m256 ties cuBLAS.
ROUTER_R3 = [(32, (16, 64, 1, 4, 256, 4)), (128, (128, 64, 4, 4, 256, 4))]
# router3 configs that prepare() times against the cuBLAS path per m (H100 m256: 12.9 vs 12.5-13.4, a tie).
ROUTER_R3_TUNE = {256: [(256, 32, 4, 2, 512, 4), (256, 64, 4, 2, 256, 4)],
                  # n17: H100 dev/n16/r3b.py m32 (32,32,2,2,256,4) 7.17-7.74 vs default 7.17-7.97 (GPU-shared noise +-0.5): let the
                  # judge's GPU pick (MT 32 reads W once at m32; NT 32 = 2x the CTAs at m16)
                  16: [(16, 32, 1, 2, 256, 4), (16, 32, 1, 2, 128, 4)], 32: [(32, 32, 2, 2, 256, 4), (32, 64, 1, 4, 256, 4)]}
# n16: router m >= 128 as one strided-batched cuBLAS GEMM over S K-slices (N = 512 = [hi; lo], FP32 out) + pj_rsum
# (sums slices and halves). H100 (dev/n16/bmm.py): m256 S4 11.1-11.4 vs 13.5-13.6 (torch.mm + add) / router3 13.6;
# m128 S4/S8 10.2-10.3 vs router3 10.4. Candidates ("bmm", S) in _tune_router, picked on the judge's GPU.
ROUTER_BMM_TUNE = {128: [4, 8], 256: [4, 8]}
# Router hi/lo tensor-core configs (n, mt, nwr, nwk, stages, na, cluster, token groups), chosen on the H100 (dev/g5.py).
ROUTER_HL = [(16, (256, 2, 2, 4, 8, 2, 8, 1)), (32, (256, 4, 2, 2, 8, 2, 8, 1))]
# BF16 grid-stride GEMV configs per layer: (m, k, u, nv, minb, pf, blocks), chosen on the H100 (dev/g10.py).
BF16_G10 = {"qkv_gate": [(1, 8192, 4, 4, 1, 1, 132)], "pooler": [(1, 8192, 4, 4, 1, 1, 132)],
            "o": [(1, 1024, 2, 4, 2, 1, 264)], "unpooler": [(1, 4096, 2, 4, 1, 1, 132)]}
# BF16 m=1 GEMV over losslessly compressed weights (dev/gemv13.cu: 12 bits per weight), (m, k, u, nv, minb, pf,
# blocks, vt), chosen on the H100 (dev/g13.py); prepare() times them against gemv10 and F.linear and keeps the fastest.
# vt 512 (512-thread virtual blocks, half the rows per round): qkv H100 13.7 -> 13.1 us (2304 rows: smaller tail).
BF16_G13 = {"qkv_gate": [(1, 8192, 2, 4, 1, 1, 132, 256), (1, 8192, 2, 2, 1, 2, 132, 256), (1, 8192, 1, 2, 1, 1, 132, 512)],
            "pooler": [(1, 8192, 2, 4, 1, 1, 132, 256), (1, 8192, 2, 2, 1, 2, 132, 256)],
            "unpooler": [(1, 4096, 2, 4, 1, 2, 132, 256)]}
# s16: gemv13 for o (k 1024) U2 V4 loses on the H100 tuner (11.3 vs gemv10 11.0: ~2 rounds per vb, latency-bound);
# s23: U2 V2 B2 on 264 blocks (512-thread blocks, 2/SM) wins on H100 g13.py (8.16-8.38 vs gemv10 8.53-8.58 us) but LOSES
# on the H200 (check 2722e8c4 tuner: g13 10.05 / U2V1B4 g528 10.11 vs gemv10 9.70): o m1 is latency-bound there. Dead.
# gemv13 built with G13_CPA: the weight chunks go through a cp.async smem ring (PF chunks in flight, no registers).
# H100 qkv m1 (2,2,1,3) 13.4 vs gemv13 14.4 us, pooler tie, unp1k 7.6 vs 7.7 (dev/g13.py XF=-DG13_CPA).
BF16_G13C_S16 = {"qkv_gate": [(1, 8192, 2, 2, 1, 3, 132, 256), (1, 8192, 2, 2, 1, 4, 132, 256)],
             "pooler": [(1, 8192, 2, 2, 1, 3, 132, 256), (1, 8192, 2, 2, 1, 4, 132, 256)],
             "unpooler": [(1, 4096, 2, 2, 1, 3, 132, 256), (1, 4096, 2, 2, 1, 4, 132, 256)]}
# s17: H200 diag (bytes in flight: H200 needs ~40% more than H100 at the same latency); H100 ties/loses.
BF16_G13C_S17 = {l: [(1, 8192, 2, 2, 1, 3, 132, 256), (1, 8192, 2, 2, 1, 4, 132, 256), (1, 8192, 2, 4, 1, 2, 132, 256),
                 (1, 8192, 1, 2, 1, 4, 132, 512)] for l in ("qkv_gate", "pooler")}
BF16_G13C = {}  # H200 check 29a2b16b: g13c loses every config (qkv 14.30+ vs g13 13.98, pooler 18.69+ vs 18.53)
G13_ESC = 8  # escapes per row (gemv13.cu / gemv14.cu ESC)
# BF16 tensor-core streaming GEMM over the same compressed weights (dev/gemv14.cu = gemv12 + decode), (m, kcu, wg, pf,
# bk, xp), chosen on the H100 (dev/g14.py); tuned in prepare() like the others.
# s16: qkv m8 without split-K (kcu 8, 96 blocks of 3 row groups): H100 15.2-15.6 vs (8, 2, 2, 4, 32, 1) 16.8 us
BF16_G14 = {"qkv_gate": [(8, 2, 2, 4, 32, 1), (8, 8, 8, 4, 96, 1), (8, 8, 8, 4, 96, 2), (8, 8, 8, 8, 96, 1)], "o": [(8, 1, 2, 4, 128, 1), (8, 1, 2, 8, 128, 1)],
            "pooler": [(8, 8, 8, 4, 128, 1), (8, 2, 2, 4, 32, 1), (16, 4, 4, 4, 64, 1)],
            "unpooler": [(8, 4, 8, 4, 32, 1), (8, 2, 4, 4, 32, 1)]}
# gemv14 with the split-K reduction over a thread-block cluster through DSMEM (dev/gemv15.cu), (m, kcu, wg, pf, runs, xp),
# H100 (dev/g15.py): pooler m16 25.1 vs cuBLAS 27.4; elsewhere no better (m32: x smem reloads per 8-row atom).
BF16_G15 = {"pooler": [(16, 4, 4, 4, 64, 1)]}
# gemv15 with W as the mma A operand (16 rows), x via ldmatrix and x staged by TMA bulk copies multicast over a cluster of
# mc blocks sharing the k-chunk (dev/gemv16.cu), (m, kcu, wg, pf, runs, xp, mc), H100 (dev/g16.py): unpooler m16 (S=1,
# no split-K) 23.0 vs cuBLAS 25.5; qkv m16 (S=2 + multicast: clusters of 4) 16.9 vs 18.1; pooler m16 (S=2, partials pushed
# over DSMEM) 25.2 vs 27.5. H200 (check ae0e03fa, unpooler only): 21.8 vs lt 21.4, not picked.
BF16_G16 = {"unpooler": [(16, 4, 4, 4, 128, 1, 2), (16, 4, 8, 2, 128, 1, 2)],
            "qkv_gate": [(16, 4, 4, 4, 48, 1, 2), (16, 4, 4, 4, 36, 1, 2)], "pooler": [(16, 4, 4, 4, 64, 1, 1)]}
# BF16 small-m GEMV with router2's structure (dev/gemv17.cu: R whole rows per block, x in registers, no split K),
# (m, R, T), H100 (dev/g17.py): unpooler-1k m1 5.66 vs 6.24 (gemv10), m8 7.65 vs 8.67; o loses (n = 8192).
# gemv21 (dev/gemv21.cu): m=1 over gemv13's 12-bit packing with the weight stream in a TMA ring (one row per stage),
# NG consumer groups of WG warps + a producer warp, rows by global ticket. (m, k, wg, ng, ns). H100 (dev/g21.py): slower
# than gemv13 (qkv 15.1 vs 13.7: HBM-bound there, per-row consumer latency ~0.8 us); built for the layer's QG overlap
# (few registers: csl fits beside it) and the H200, where gemv13 runs at ~2.6 TB/s.
# H200 (check 96165c87, first version: consumer-side finalize, static rows): qkv m1 17.7-18.3 vs gemv13 14.0, pooler 26.2
# vs 18.7, unp1k 11.2 vs g17 7.3 (tuner us): DEAD, not tuned any more (s13 NOTES).
BF16_G21 = {}
BF16_G17 = {"unpooler": [(1, 4, 512), (1, 4, 256), (8, 4, 256), (8, 4, 128)]}
# flashinfer 0.7.0 tinygemm_bf16 (TRT-LLM tinygemm2) ported to dev/tinygemm2.cu: TMA, 16 deep stages, n/16 blocks.
# H100 (dev/tg2.py, s26): unpooler-1k m8 6.53 vs 7.26 us (g17); loses elsewhere (m1 all, o/qkv/pooler m8, m>=16).
BF16_TG2 = {"unpooler": [(8,)]}
# gemv19 (dev/gemv19.cu, dev/g19_lib.py): TMA-ring streamer over the 12-bit weights, FP32 out zeroed in-kernel + red.add
# split-K partials, then a BF16 copy (g19_cvt). (m, kc, ap, ns, npre). Built for the layer (PDL W prefetch during the
# previous kernel); standalone (no predecessor) H100 it ties cuBLAS at m16 (pooler 24.8 vs 28.0) and loses at m32.
# (m, kc, ap, ns, npre, xb, ts, rg, rpw): xb x buffers, ts warps per row group, rg row groups per tile, rpw row groups per
# warp (dev/gemv19.cu). H200 diag (check 60ddb481): pooler m16 (1024, 8, 3, xb 1) 15.65 us vs pick 19.0 after a PDL predecessor.
BF16_G19 = {"qkv_gate": [(16, 1024, 8, 3, 2, 1, 1, 8, 1), (32, 1024, 8, 2, 2, 1, 1, 8, 1)],
            "pooler": [(16, 1024, 8, 3, 2, 1, 1, 8, 1), (32, 1024, 8, 2, 2, 1, 1, 8, 1)],
            "unpooler": [(16, 1024, 8, 3, 2, 1, 1, 8, 1), (32, 1024, 8, 2, 2, 1, 1, 8, 1)], "o": [(16, 512, 8, 3, 2, 1, 1, 8, 1)]}
# gemv20 (dev/gemv20.cu) = gemv19 with wgmma consumers (x by an SW128 tensor map). _diag20 only (print, no picks):
# variant a = 1 block/SM (grid #SMs), b = 2 blocks/SM (grid 2 x #SMs, single-buffered A fragments, 1 partial buffer).
# (m, kc, ap, ns) per layer; npre = ns, xb 1.
DIAG20_S12 = {"a": [(16, 1024, 4, 4), (16, 1024, 8, 3), (32, 1024, 4, 4), (32, 1024, 8, 2)],
          "b": [(16, 1024, 4, 2), (16, 512, 4, 2), (32, 512, 4, 2), (32, 512, 2, 3)],
          "c": [(16, 1024, 4, 2), (16, 512, 4, 2), (32, 512, 4, 2), (32, 512, 2, 3)]}  # c = gemv19 (mma.sync), 2 blocks/SM
DIAG20 = {}  # s12 diag done (DIAG20_S12: all lose on the H200); nothing built
# _diag19 only (print, no picks): H200 behaviour of other configs in the layer case
DIAG19 = {}  # s10 diag configs measured (checks 9b09c536/60ddb481/f1319777): see NOTES-proj
# s15: after the escape-fixup rework (gemv19 fixups on the stored partials): the layer's unpool m16 config and AP4 m32
DIAG19_S15 = {"unpooler": [(16, 1024, 4, 4, 4, 2, 1, 8, 1), (32, 1024, 4, 4, 4, 1, 1, 8, 1)],
              "pooler": [(32, 1024, 4, 4, 4, 1, 1, 8, 1)], "qkv_gate": [(16, 1024, 4, 4, 4, 2, 1, 8, 1), (32, 1024, 4, 4, 4, 1, 1, 8, 1)]}
DIAG19 = {}  # s15 diag done (check ce84da07, NOTES-proj s15)
# s15: _diag19q only (print): qkv with BF16 out written by gemv19 itself (mode 3: tile tickets, self-cleaning FP32
# workspace, tile-major stages), standalone as the layer's first op (no predecessor), vs the picked kernel
# H200 (check 4e5fad5b): m16 19.6 vs lt 16.9, m32 23.3 vs 17.5 us: DEAD (no predecessor to hide g19's startup)
DIAG19Q_S15 = {"qkv_gate": [(16, 1024, 4, 4, 4, 2, 1, 8, 1), (16, 1024, 8, 3, 3, 1, 1, 8, 1), (32, 1024, 8, 2, 2, 1, 1, 8, 1),
                            (32, 1024, 4, 4, 4, 1, 1, 8, 1)]}
DIAG19Q = {}
DIAG_SLAB = {}  # s22 diag done (check ccf41026: qkv m32 bmm2 16.03 vs lt 17.50 H200 us; NOTES s22); was {"qkv_gate": (16, 32), "o": (32,), "pooler": (32,)}
DIAG_EXT = {}  # s25 diag done (check 5abb71f1: DeepGEMM bf16_gemm_nt ties/loses vs lt everywhere); was {"qkv_gate": (16, 32, 128, 256), "o": (128, 256), "pooler": (32, 128, 256), "unpooler": (128,)}
# s50: _diag19d only (print): H200 decomposition of gemv19 at m32/m16 in the layer's case (3 us PDL predecessor, npre = ns),
# each config rebuilt with diag defines: "" (= main), G19_NOW (no W loads: consumer-only), +G19_NOMMA (no decode/mma),
# +G19_NOX (decode, no x/mma), G19_NOFIX (no escape fixups); grids 132 and 108 (the layer's pooler grid). H100 proxy
# (NOTES s16/s34): skeleton 13.8, decode +5.9, mma+ldsm +7.5 proxy-us at pool m32 G72; the H200 itself never measured.
DIAG19D = {"pooler": [(32, 1024, 8, 2, 2, 1, 1, 8, 1), (16, 1024, 8, 3, 2, 1, 1, 8, 1)], "unpooler": [(32, 1024, 8, 2, 2, 1, 1, 8, 1)]}
# s51: + NOTM (k-major order at compile time, = the layer's g19 copy) and NOTM+TAG2 (producer precomputes ks/st/tr per
# slot: no runtime div/mod in the consumers' loop top; bit-identical). H100 proxy G72: NOW mode m16 -1.0, m32 -0.1..-0.4;
# full kernel tie (W-bound on H100). The H200 decides whether it goes to the layer.
DIAG19D_DEFS = ["", "G19_NOW", "G19_NOW G19_NOMMA", "G19_NOW G19_NOX", "G19_NOFIX", "G19_NOTM", "G19_NOTM G19_TAG2"]
DIAG19D_GRIDS = (132, 108)
# s53: _diag24 only (print): gemv24 (dev/s53/gemv24.cu: per-warp TMA rings, no producer warp / stage barriers; pack19 with
# rg 1) vs diag19d's g19 main, same harness (3 us PDL predecessor, npre = ns), grids 132/108. (m, kc, ap, ns, wpb, xb, r).
# H100 G132 us (g24 | g19 layer-like): pool m16 18.6 | 19.1, m32 21.1 | 21.6, unp m16 18.8 | 19.7, m32 20.2 | 21.2.
# H200 proxy G72 (g24 best | g19 best): m32 pool 24.8 | 26.6, unp 23.9 | 24.7, o 8.7 | 9.9, qkv 16.1 | 16.4; m16 pool 20.4 | 19.1.
DIAG24 = {"pooler": [(16, 1024, 2, 4, 16, 2, 1), (16, 1024, 4, 3, 12, 2, 1), (32, 512, 2, 4, 16, 2, 1), (32, 512, 4, 3, 12, 2, 1)],
          "unpooler": [(16, 1024, 2, 4, 16, 2, 1), (16, 1024, 4, 3, 12, 2, 1), (32, 512, 2, 4, 16, 2, 1), (32, 512, 4, 3, 12, 2, 1)],
          "qkv_gate": [(16, 512, 2, 4, 16, 2, 1), (16, 1024, 2, 4, 16, 2, 1), (32, 512, 2, 4, 16, 2, 1)],
          "o": [(16, 512, 2, 4, 16, 2, 1), (32, 512, 2, 4, 16, 2, 1)]}
DIAG24 = {k: v + [c[:6] + (257,) for c in v] for k, v in DIAG24.items()}  # s54: + SK twins (r = 1 | 256: stream-K stages)
DIAG24_GRIDS = (132, 108)
DIAG19D = {}  # s56: diag19d/diag24 done (check 636c89fc FAILED setup: prepare() 325.6 s > 300 s with them on; lines in NOTES s56)
DIAG24 = {}
# s57: _diag27 only (print): gemv27 (dev/s57/gemv27.cu: no cross-block split-K; block = RGB whole row groups, warps split
# k, smem sum, plain stores: no zeroing / epoch barrier / atomics; ns >= 100 = PD: W decoded into registers BEFORE
# griddepcontrol.wait) for the layer's o at T=16/32 vs g24's layer cfg / cuBLAS / lt pick, after a 3 us PDL pred.
# H100 (us over pred): o m16 PD 5.1-5.4 | g24 6.7 | cuBLAS 7.9; m32 PD 7.0 | g24 7.2 | cuBLAS 8.4.
DIAG27 = {"o": [(16, 4, 4, 108, 4, 0), (16, 4, 4, 2, 4, 0), (32, 4, 4, 108, 4, 0), (32, 4, 4, 2, 4, 0)]}
DIAG27_G24 = {("o", 16): (16, 512, 2, 4, 16, 2, 1), ("o", 32): (32, 512, 4, 3, 12, 2, 1)}
# H200 check c7b58683: o m16 PD +5.25 | non-PD +5.82 | g24 +6.43 | cuBLAS +6.75 | lt +7.20; m32 PD +6.69 | non-PD +7.30 |
# g24 +7.17 | cuBLAS +6.43 us. -> layer handoff dev/s57/g27_layer.patch (T=16 o). Diag off now.
DIAG27 = {}
# s58: G27_MC (dev/s58/gemv27.cu): 1 = x by bulk copies spread over threads (new default; H100 layer d16 sl -1.8, full
# -1.05 us vs 0 = cp.async xload; standalone m32 -0.5, m16 tie), 2/4 = cluster multicast (H100 loses for PD). H200 A/B:
DIAG27 = {"o": [(16, 4, 4, 108, 4, 0), (32, 4, 4, 108, 4, 0)]}
DIAG27_MC = (1, 0)
# H200 check a4962d70 (us over pred): o m16 g27 PD mc1 +4.42 | mc0 +5.28 | g24 +6.59 | lt pick +6.34 | cuBLAS +6.82;
# m32 mc1 +6.27 | mc0 +6.88 | g24 +7.33 | lt pick +6.72 | cuBLAS +6.98. -> layer handoff dev/s58 (T=16 on, T=32 TUNE). Off now.
DIAG27 = {}
# s61: _diag61 only (print): the layer's g24 (dev/s61/gemv24.cu = r-layer g24_src + runtime npre bits) at the layer's
# T=16/32 pooler/unpool cfgs + grids, after a 3 us PDL pred, per variant: base | XSP (bit 19: x barriers armed pre-wait,
# x copies over threads) | DF (bit 20: defer a task's flush while the out-zero barrier is pending) | XSP+DF | EZ probe
# (bit 17, out not zeroed: timing upper bound for removing zero/fence/epoch). H100 clock64 split (pool m32 G88): setup
# (zero + fence + epoch) 3.5-7.5 kcyc + barrier spin 2-3 kcyc; with EZ the x wait is 6.3 kcyc (XSP: 2.0). H100 = W-bound tie.
DIAG61 = [("pooler", (16, 1024, 4, 3, 12, 2, 1), 108), ("unpooler", (16, 1024, 2, 4, 16, 2, 1), 132),
          ("pooler", (32, 512, 4, 3, 12, 2, 1), 108), ("pooler", (32, 512, 2, 4, 16, 2, 257), 108),
          ("unpooler", (32, 512, 4, 3, 12, 2, 1), 132), ("unpooler", (32, 512, 2, 4, 16, 2, 257), 132)]
DIAG61 = []  # s68: s61 diag done (check e25a1d4d: XSP/DF dead on the H200); keeps prepare() short
DIAG61_BITS = [("base", 0), ("xsp", 1 << 19), ("df", 1 << 20), ("xsp+df", 3 << 19), ("EZprobe", 1 << 17)]
DIAG19_STREAM = []  # s17 H200: 50 MB 13.89, 28 MB 8.51 us at (24576, 4) (1-kernel graph 4.6 us subtracted); was [(24576, 4), (12288, 8)]  # (stage bytes, stages): 50 MB and 28 MB bare streams
# gemv22 (dev/gemv22.cu, dev/g22.py): m = 128/256 swap-AB wgmma GEMM over the 12-bit weights, tile-major stream-K (one
# run per block), escapes packed per stage (E entries), FP32 out (zero kernel + red.add), then a BF16 copy.
# (m, ap, ns, E). H100: qkv m128 20.8 vs cuBLAS 20.6, others lose (H100 TMA stream ~2.9 TB/s); built for the H200.
# H200 (check fb2fe4ee): loses every m128/m256 shape (qkv m128 22.1 vs lt 18.8, pooler 29.6 vs 24.4; m256 +10..19 us):
# per-SM smem ingest (W + x box per stage, x/W = 1.33 at m128) caps it, not HBM. DEAD; candidates were:
BF16_G22_S16 = {l: [(128, 2, 6, 32), (128, 4, 3, 32), (256, 2, 4, 32), (256, 2, 3, 32)] for l in ("qkv_gate", "o", "pooler", "unpooler")}
BF16_G22 = {}
_SMS = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count if torch.cuda.is_available() else 132
_LAYER_K = {"qkv_gate": 8192, "o": 1024, "pooler": 8192, "unpooler": 4096}
# BF16 tensor-core streaming GEMM configs per layer: (m, kcu, wg, pf, bk, xp, xr), chosen on the H100 (dev/g12.py).
BF16_G12 = {"qkv_gate": [(8, 2, 2, 4, 32, 1, 1)], "pooler": [(8, 8, 8, 4, 128, 1, 1), (16, 8, 8, 4, 128, 8, 2)],
            "o": [(8, 1, 2, 8, 128, 1, 1)], "unpooler": [(8, 4, 4, 4, 32, 1, 1)]}  # unpooler m8: H200 5.7-7.3 (noisy)
# cuBLASLt configs to try per (layer, m), from an exhaustive H100 sweep of algo 66 (dev/lte.py): (algo, tile, stages,
# split k, reduction, swizzle, custom option, cluster). prepare() times them and F.linear on this GPU, cold L2, and keeps
# the fastest (or F.linear).
BF16_LT = @@LTCFG@@
LT_WS_BYTES = 32 << 20
# s100: g30 (layer's 12-bit W decoded into wgmma A registers, swap-AB; MTOK 16/32 = proj s95 g30n with a (1, S, 1) cluster
# split-K + DSMEM reduce; MTOK 128 = layer s99 g30). Per (layer, m): (S, NW, NS) tuner candidates; one build per (m, S, NW, NS).
# n19: (S, NW, NS, "G30_GX=1") = escapes out of the main loop (n5s6; H200 diag30e a670beb1 qkv m32 standalone 12.86 vs 13.09).
BF16_G30 = {("unpooler", 16): [(1, 16, 16)], ("pooler", 16): [(1, 16, 16), (2, 16, 16)],
            ("qkv_gate", 16): [(3, 16, 16), (2, 16, 16), (3, 16, 16, "G30_GX=1")],
            ("unpooler", 32): [(1, 16, 16)], ("pooler", 32): [(2, 16, 16)], ("qkv_gate", 32): [(3, 16, 16), (3, 16, 16, "G30_GX=1")],
            ("unpooler", 128): [(1, 10, 10)]}  # n5s4: o and the 20/8 rings dropped (H200 never picked them, checks 7ad5148c/7cbaa0fd)
# n5s4 _diag30k only (print, no picks): g30k (stream-K g30 on G CTAs; G30K_F32 1 = FP32 slabs, 0 = BF16 out via slab+ticket)
# vs g30 S-split, standalone (first op) and after a 3 us PDL predecessor: does qkv gain from 132 CTAs on the H200?
DIAG30K = {}  # n5s4 diag done (check 6d9063a1: qkv kF132 only -0.2 us vs g30 S3; pooler/unpool g30k lose 1-3 us; NOTES n5s4)
# n5s4 _diag30e only (print, no picks): g30 with the escape path compiled out (G30_ESC 0: WRONG output, timing only) vs g30,
# m32, standalone and after a 3 us PDL predecessor = the H200 upper bound of a cheaper escape format.
DIAG30E = {}  # n19: GX is a qkv tuner candidate now (diag30e a670beb1: GX standalone qkv -0.23, pooler +1.0, unp +0.6 us); was {"qkv_gate": 3, "pooler": 2, "unpooler": 1}. n5s6: G30_GX vs g30 (print only). n5s5 diag done (check b3ff96fb: G30_PL vs g30 m32 standalone -0.3/-0.4/-0.4, after pred -0.2/-0.2/-0.5 us qkv/pool/unp; was {"qkv_gate": 3, "pooler": 2, "unpooler": 1}). n5s4 diag done (check 198e7084: escapes cost H200 qkv/pool/unp m32 1.2/1.1/0.6 us standalone, 1.5/2.0/2.0 after a pred; NOTES n5s4); was {"qkv_gate": 3, "pooler": 2, "unpooler": 1}
DIAG30E_M = 32
DIAG30E_V = "G30_GX=1"  # n5s6: escapes out of the main loop (prepare() records, TMA x boxes, escape warps). n5s5 was "G30_PL=0": the variant timed against g30 ("noesc" column): escape keys in a per-thread smem list (0 = no escape path)
DIAG30K_S = {"qkv_gate": 3, "pooler": 2, "unpooler": 1}
_G7 = sorted({c[1:6] for _, c in ROUTER_HL})
_SRC = (_ROUTER_SRC.replace("PR2_CONFIGS", " ".join("PR2_CASE(%d,%d,%d)" % c for _, c in ROUTER_F32)) + "\n" +
        _GEMV_SRC.replace("G7_CONFIGS", " ".join("G7_CASE(%d,%d,%d,%d,%d)" % c for c in _G7)) + "\n" +
        _G10_SRC.replace("G10_CONFIGS", " ".join("G10_CASE(%d,%d,%d,%d,%d,%d)" % c[:6] for v in BF16_G10.values() for c in v)) + "\n" +
        _G12_SRC.replace("G12_CONFIGS", " ".join("G12_CASE(%d,%d,%d,%d,%d,%d)" % c for c in sorted({c[:4] + c[5:] for v in BF16_G12.values() for c in v}))) + "\n" +
        _G13_SRC.replace("G13_CONFIGS", " ".join("G13_CASE(%d,%d,%d,%d,%d,%d)" % (c[1:6] + c[7:8]) for c in sorted({c for v in BF16_G13.values() for c in v}))) + "\n" +
        _G14_SRC.replace("G14_CONFIGS", " ".join("G14_CASE(%d,%d,%d,%d,%d)" % (c[:4] + c[5:]) for c in sorted({c for v in BF16_G14.values() for c in v}))) + "\n" +
        _G15_SRC.replace("G15_CONFIGS", " ".join("G15_CASE(%d,%d,%d,%d,%d)" % (c[:4] + c[5:]) for c in sorted({c for v in BF16_G15.values() for c in v}))) + "\n" +
        _G16_SRC.replace("G16_CONFIGS", " ".join("G16_CASE(%d,%d,%d,%d,%d,%d)" % (c[:4] + c[5:]) for c in sorted({c for v in BF16_G16.values() for c in v}))) + "\n" +
        _G17_SRC.replace("G17_CONFIGS", " ".join("G17_CASE(%d,%d,%d,%d)" % c for c in sorted({c + (_LAYER_K[l],) for l, v in BF16_G17.items() for c in v}))) + "\n" +
        _G21_SRC.replace("G21_CONFIGS", " ".join("G21_CASE(%d,%d,%d,%d)" % c[1:] for c in sorted({c for v in BF16_G21.values() for c in v}))) + "\n" +
        _R3_SRC.replace("PR3_CONFIGS", " ".join("PR3_CASE(%d,%d,%d,%d,%d,%d)" % c for c in [c for _, c in ROUTER_R3] + [c for v in ROUTER_R3_TUNE.values() for c in v])))
try:
    from torch.utils.cpp_extension import load_inline
    _mod = load_inline(
        name="muse_proj_v15",
        cpp_sources="void pj_router2(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t);\n"
                    "void g7_gemv(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, "
                    "int64_t, int64_t, int64_t);\n"
                    "void g10_gemv(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t);\n"
                    "void g12_gemm(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, "
                    "int64_t, int64_t, int64_t, int64_t, int64_t);\n"
                    "void g13_gemv(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, int64_t, "
                    "int64_t, int64_t, int64_t, int64_t, int64_t);\n"
                    "void g14_gemm(torch::Tensor, torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor, "
                    "torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);\n"
                    "void g15_gemm(torch::Tensor, torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor, "
                    "int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);\n"
                    "void g16_gemm(torch::Tensor, torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor, "
                    "int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);\n"
                    "void g17_gemv(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t);\n"
                    "void g21_gemv(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, int64_t, "
                    "int64_t, int64_t, int64_t, torch::Tensor, int64_t, torch::Tensor);\n"
                    "void pj_router3(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, "
                    "int64_t, int64_t);\n"
                    "void pj_rsum(torch::Tensor, torch::Tensor);",
        cuda_sources=_SRC, functions=["pj_router2", "g7_gemv", "g10_gemv", "g12_gemm", "g13_gemv", "g14_gemm", "g15_gemm",
                                      "g16_gemm", "g17_gemv", "g21_gemv", "pj_router3", "pj_rsum"],
        extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"], verbose=False)
except Exception as _e:  # noqa: BLE001
    print("muse_proj native build failed, using torch:", _e)
    _mod = None

try:
    _g19 = load_inline(
        name="muse_proj_g19a",
        cpp_sources="void g19_gemm(torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, "
                    "int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t); void g19_cvt(torch::Tensor, torch::Tensor); "
                    "void g19_gemm_bf16(torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, "
                    "int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t); "
                    "void g19_pred(torch::Tensor, torch::Tensor, int64_t); void g19_stream(torch::Tensor, int64_t, int64_t, torch::Tensor);",
        cuda_sources=("#define G19_ONLY23\n" if DIAG19Q else "#define G19_ONLY2\n") + _G19_SRC.replace("G19_CONFIGS", " ".join(
            "G19_CASE(%d,%d,%d,%d,1,%d,%d,%d,%d)" % (c[:4] + c[5:]) for c in sorted({c for v in list(BF16_G19.values()) + list(DIAG19.values()) + list(DIAG19Q.values()) for c in v}))),
        functions=["g19_gemm", "g19_cvt", "g19_pred", "g19_stream", "g19_gemm_bf16"], extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"])
except Exception as _e:  # noqa: BLE001
    print("muse_proj gemv19 build failed:", _e)
    _g19 = None

try:
    _g13c = load_inline(
        name="muse_proj_g13c",
        cpp_sources="void g13_gemv(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, int64_t, "
                    "int64_t, int64_t, int64_t, int64_t, int64_t);",
        cuda_sources="#define G13_CPA\n" + _G13_SRC.replace("G13_CONFIGS", " ".join("G13_CASE(%d,%d,%d,%d,%d,%d)" % (c[1:6] + c[7:8]) for c in sorted({c for v in BF16_G13C.values() for c in v}))),
        functions=["g13_gemv"], extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"]) if BF16_G13C else None
except Exception as _e:  # noqa: BLE001
    print("muse_proj gemv13 cp.async build failed:", _e)
    _g13c = None

try:
    _g22 = load_inline(
        name="muse_proj_g22a",
        cpp_sources="void g22_gemm(torch::Tensor, double, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);",
        cuda_sources=_G22_SRC.replace("G22_CONFIGS", " ".join("G22_CASE(%d,%d,%d,%d)" % c for c in sorted({c for v in BF16_G22.values() for c in v}))),
        functions=["g22_gemm"], extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"]) if BF16_G22 else None
except Exception as _e:  # noqa: BLE001
    print("muse_proj gemv22 build failed:", _e)
    _g22 = None

_g20 = {}
for _v, _defs in [v for v in (("a", ""), ("b", "#define G20_MINB 2\n#define G20_SB\n#define G20_NPB 1\n"), ("c", "")) if v[0] in DIAG20]:
    try:
        if _v == "c":  # gemv19 with 2 blocks per SM, exposed as g20_gemm
            _g20[_v] = load_inline(
                name="muse_proj_g19c",
                cpp_sources="void g19_gemm(torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, "
                            "int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);",
                cuda_sources="#define G19_ONLY2\n#define G19_MINB 2\n#define G19_NPB 1\n" + _G19_SRC.replace("G19_CONFIGS", " ".join(
                    "G19_CASE(%d,%d,%d,%d,1,1,1,8,1)" % c for c in sorted(set(DIAG20[_v])))),
                functions=["g19_gemm"], extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"])
            _g20[_v].g20_gemm = _g20[_v].g19_gemm
            continue
        _g20[_v] = load_inline(
            name="muse_proj_g20" + _v,
            cpp_sources="void g20_gemm(torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, "
                        "int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);",
            cuda_sources="#define G20_ONLY2\n" + _defs + _G20_SRC.replace("G20_CONFIGS", " ".join(
                "G20_CASE(%d,%d,%d,%d,1,1,1,8,1)" % c for c in sorted(set(DIAG20[_v])))),
            functions=["g20_gemm"], extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"])
    except Exception as _e:  # noqa: BLE001
        print("muse_proj gemv20", _v, "build failed:", _e)

try:
    _tg2 = load_inline(
        name="muse_proj_tg2a", cpp_sources="void tg2(torch::Tensor, torch::Tensor, torch::Tensor, bool);",
        cuda_sources=_TG2_SRC, functions=["tg2"], extra_ldflags=["-lcuda"],
        extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"]) if BF16_TG2 else None
except Exception as _e:  # noqa: BLE001
    print("muse_proj tinygemm2 build failed:", _e)
    _tg2 = None

try:
    _lt = load_inline(
        name="muse_proj_lt1",
        cpp_sources="int64_t lt_gemm(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, std::vector<int64_t>);\n"
                    "torch::Tensor lt_cands(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);",
        cuda_sources=_LT_SRC, functions=["lt_gemm", "lt_cands"], extra_ldflags=["-lcublasLt"],
        extra_cuda_cflags=["-O3"], verbose=False)
except Exception as _e:  # noqa: BLE001
    print("muse_proj cuBLASLt build failed, using F.linear:", _e)
    _lt = None

_G30 = {}


def _g30_build(c):
    """c = (MTOK, S, NW, NS[, ESC]) -> g30 lib (None on failure)."""
    try:
        import hashlib
        dfs = [f"-DG30_MTOK={c[0]}", f"-DG30_S={c[1]}", f"-DG30_NW={c[2]}", f"-DG30_NS={c[3]}"] + [f"-D{e}" if isinstance(e, str) else f"-DG30_ESC={e}" for e in c[4:]]
        return load_inline(
            name="muse_proj_g30_" + hashlib.sha256((_G30_SRC + " ".join(dfs)).encode()).hexdigest()[:10],
            cpp_sources="void g30_gemm(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, int64_t);",
            cuda_sources=_G30_SRC, functions=["g30_gemm"],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", *dfs], extra_ldflags=["-lcuda"])
    except Exception as ex:  # noqa: BLE001
        print("muse_proj g30 build failed:", c, ex)
        return None


_G30K = {}


def _g30k_build(c):
    """c = (MTOK, F32) -> g30k lib (NW16 NS16, up to 6 pieces per tile) for _diag30k (None on failure)."""
    try:
        import hashlib
        dfs = ["-DG30K_ON", f"-DG30K_F32={c[1]}", "-DG30K_MAXP=6", f"-DG30_MTOK={c[0]}", "-DG30_NW=16", "-DG30_NS=16"]
        return load_inline(
            name="muse_proj_g30k_" + hashlib.sha256((_G30_SRC + " ".join(dfs)).encode()).hexdigest()[:10],
            cpp_sources="void g30k_gemm(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, int64_t, "
                        "torch::Tensor, torch::Tensor, int64_t);",
            cuda_sources=_G30_SRC, functions=["g30k_gemm"],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", *dfs], extra_ldflags=["-lcuda"])
    except Exception as ex:  # noqa: BLE001
        print("muse_proj g30k build failed:", c, ex)
        return None


if BF16_G30:
    from concurrent.futures import ThreadPoolExecutor
    _cs = sorted({(m, *c) for (_l, m), v in BF16_G30.items() for c in v})
    _ks = [(m, f) for m in (16, 32) for f in (1, 0)] if DIAG30K else []
    _es = sorted({(DIAG30E_M, S, 16, 16, DIAG30E_V) for S in DIAG30E.values()})
    with ThreadPoolExecutor(len(_cs) + len(_ks) + len(_es)) as _ex:
        _fk = [_ex.submit(_g30k_build, c) for c in _ks]
        _fe = [_ex.submit(_g30_build, c) for c in _es]
        _G30.update(zip(_cs, _ex.map(_g30_build, _cs)))
        _G30K.update(zip(_ks, (f.result() for f in _fk)))
        _G30.update(zip(_es, (f.result() for f in _fe)))

ROUTER_FP32_MAX_TOKENS = 8


def _pack(w):
    n, k = w.shape
    return w.view(n // 16, 2, 8, k // 16, 2, 4, 2).permute(0, 3, 2, 5, 4, 1, 6).reshape(n // 16, k // 16, 256)


def _pack12(w, kc):
    # [k-chunk][8-row group][kc/32 atoms][lane][8]: atom = 8 rows x 32 k, lane l = row l/4, k (l%4)*8..+8
    n, k = w.shape
    return w.view(n // 8, 8, k // kc, kc // 32, 4, 8).permute(2, 0, 3, 1, 4, 5).contiguous()


def _compress(w):
    """Lossless 12-bit BF16 (gemv13/gemv14): sign|mantissa byte and 4-bit exponent code c = e - base per weight
    (base = max exponent - 15; decoded exponent 112 | c, so sums are rescaled by scale = 2^(base - 112)); weights below
    the window (zeros, tiny values) are escapes: code 0, byte 0 (decodes to 2^(base - 127)) plus a per-row header of
    G13_ESC (col, true / scale) entries (exact; kernels subtract 2^-15 = 2^(base - 127) / scale). None if a row has more than G13_ESC escapes."""
    n, k = w.shape
    b = w.contiguous().view(torch.int16).int() & 0xFFFF
    e = (b >> 7) & 0xFF
    base = int(e.max()) - 15
    if base < 1:
        return None
    code = e - base
    esc = code < 0
    if int(esc.sum(1).max()) > G13_ESC:
        return None
    code = torch.where(esc, 0, code)
    sm = torch.where(esc, 0, ((b >> 8) & 0x80) | (b & 0x7F))
    scale = 2.0 ** (base - 112)
    # empty slots: col 0, value 2^-15 (the kernels add x[col] * (value - 2^-15): exactly 0)
    hdr = torch.zeros(n, G13_ESC, 2, dtype=torch.float32, device=w.device)
    hdr[:, :, 1] = 2.0 ** -15
    rows, cols = esc.nonzero(as_tuple=True)  # row-major order
    if len(rows):
        slot = torch.arange(len(rows), device=w.device) - torch.searchsorted(rows, rows)
        hdr[rows, slot, 0] = cols.int().view(torch.float32)
        hdr[rows, slot, 1] = w[rows, cols].float() / scale  # BF16 over a power of two: exact in FP32
    # round trip: every weight decodes to its original bits (else no compression for this weight)
    dec = torch.where(esc, 0, (((sm & 0x80) << 8) | ((code + base) << 7) | (sm & 0x7F)))
    dec = dec.to(torch.int32).where(dec < 32768, dec - 65536).to(torch.int16).view(torch.bfloat16)
    if len(rows):
        fix = (hdr[rows, slot, 1] * scale).to(torch.bfloat16)
        if not torch.equal(fix.view(torch.int16), w[rows, cols].contiguous().view(torch.int16)):
            return None
        dec[rows, cols] = fix
        if not torch.equal(hdr[rows, slot, 1] * scale, w[rows, cols].float()):
            return None
    if not torch.equal(dec.view(torch.int16), w.contiguous().view(torch.int16)):
        return None
    return code, sm, hdr, scale


def _nib(code):
    """Codes [..., 8] -> int32 words: weight 2p + h at bits 4p + 16h."""
    sh = torch.tensor([4 * (i // 2) + 16 * (i % 2) for i in range(8)], device=code.device)
    v = (code.long() << sh).sum(-1)
    return torch.where(v >= 2 ** 31, v - 2 ** 32, v).int()


def _dec3(sm, code):
    """s35 DEC3 layout over groups of 8 weights (last dim = the kernels' dec8 unit: 8 bytes + 1 word): byte = c0 << 7 |
    mantissa; word: weight 2P + h's sign, c1, c2, c3 at bits (15, 8, 9, 10) + 16h + (0, -4, -8, 4)[P] (mod 32). Kernels
    (decw in dev/gemv*.cu): prmt with 0x38 for the exponent's top bits + one lop3 per BF16 pair. Checked lossless here."""
    sm, code = sm.long(), code.long()
    i = torch.arange(8, device=sm.device)
    off = 16 * (i % 2) + torch.tensor([0, -4, -8, 4], device=sm.device)[i // 2]
    pos = [(b + off) % 32 for b in (15, 8, 9, 10)]
    bits = [(sm >> 7) & 1, (code >> 1) & 1, (code >> 2) & 1, (code >> 3) & 1]
    v = sum(b << q for b, q in zip(bits, pos)).sum(-1)
    b3 = ((code & 1) << 7) | (sm & 0x7F)
    back = [(v.unsqueeze(-1) >> q) & 1 for q in pos]  # round trip: every bit lands in its own place
    if not (torch.equal(back[0], bits[0]) and torch.equal(back[1] | back[2] << 1 | back[3] << 2, code >> 1)
            and torch.equal(b3 >> 7, code & 1) and torch.equal(b3 & 0x7F, sm & 0x7F)):
        raise RuntimeError("DEC3 packing round trip failed")
    return b3.to(torch.uint8), torch.where(v >= 2 ** 31, v - 2 ** 32, v).int()


def _pack30(code, sm):
    """[N, K] codes / sign|mantissa bytes -> [N/64][K/64][6144 B] g30 stages: DEC3 bytes [2][128 thr][16], words [128][4]
    (thread = (w, gq, q) of the wgmma A fragment; layer pack30)."""
    n, k = code.shape
    G, KS = n // 64, k // 64
    # row = 16 w + 8 rr + gq ; k = 16 j + 8 kh + 2 q + h ; element e = (kh, rr, h) ; thread = (w, gq, q)
    perm = lambda t: t.view(G, 4, 2, 8, KS, 4, 2, 4, 2).permute(0, 4, 1, 3, 7, 5, 6, 2, 8).reshape(G, KS, 128, 4, 8)  # noqa: E731
    b3, v = _dec3(perm(sm), perm(code))
    Sb = b3.view(G, KS, 128, 2, 16).permute(0, 1, 3, 2, 4).reshape(G, KS, 4096)
    Nb = v.contiguous().view(torch.uint8).view(G, KS, 2048)
    return torch.cat([Sb, Nb], 2).contiguous()


def _pack13(w):
    """gemv13 layout: flat byte plane, code words over 8 consecutive weights."""
    c = _compress(w)
    if c is None:
        return None
    code, sm, hdr, scale = c
    s3, n3 = _dec3(sm.flatten().view(-1, 8), code.flatten().view(-1, 8))
    return s3.flatten().contiguous(), n3.contiguous(), hdr, scale


def _pack14(w, kc):
    """gemv14 layout: gemv12's atom order [ks][rg][atom][lane][8] for bytes and codes."""
    c = _compress(w)
    if c is None:
        return None
    code, sm, hdr, scale = c
    n, k = w.shape
    perm = lambda t: t.view(n // 8, 8, k // kc, kc // 32, 4, 8).permute(2, 0, 3, 1, 4, 5).contiguous()
    s3, n3 = _dec3(perm(sm), perm(code))
    return s3, n3.contiguous(), hdr, scale


def _pack16(w, kc):
    """gemv16 layout: [ks][16-row group][atom = 16 rows x 32 k][lane][16] bytes, codes as 2 words per lane; lane (g, c)
    holds the two mma A fragments (k-steps s) of rows g, g + 8 and k 2c, 2c + 1, 2c + 8, 2c + 9."""
    c = _compress(w)
    if c is None:
        return None
    code, sm, hdr, scale = c
    n, k = w.shape
    # [rg, h, g, ks, atom, s, hk, c, e] -> [ks, rg, atom, g, c, s, hk, h, e]
    perm = lambda t: t.view(n // 16, 2, 8, k // kc, kc // 32, 2, 2, 4, 2).permute(3, 0, 4, 2, 7, 5, 6, 1, 8).contiguous()
    s3, nw = _dec3(perm(sm).reshape(k // kc, n // 16, kc // 32, 32, 2, 8), perm(code).reshape(k // kc, n // 16, kc // 32, 32, 2, 8))
    sw = s3.reshape(k // kc, n // 16, kc // 32, 32, 16).contiguous()
    nw = nw.contiguous()
    return sw, nw, hdr, scale


def _pack19(c, kc, ap, rg=8, tm=0):
    """_compress output -> gemv19 stages: stage (ks, 128-row tile, st) = [8 row groups x ap atoms x 32 lanes x 16 B of
    sign|mantissa][same x 8 B of codes], one contiguous block (one bulk copy)."""
    code, sm, hdr, scale = c
    n, k = code.shape
    perm = lambda t: t.view(n // 16, 2, 8, k // kc, kc // 32, 2, 2, 4, 2).permute(3, 0, 4, 2, 7, 5, 6, 1, 8).contiguous()
    S, N = _dec3(perm(sm).reshape(k // kc, n // 16, kc // 32, 32, 2, 8), perm(code).reshape(k // kc, n // 16, kc // 32, 32, 2, 8))
    S = S.reshape(k // kc, n // 16, kc // 32, 32, 16)
    N = N.contiguous()
    ks, ntr, spk = k // kc, n // (16 * rg), kc // 32 // ap
    pm = (1, 0, 3, 2, 4) if tm else (0, 1, 3, 2, 4)  # tm: tile-major stages (tile, ks, st) for mode 3
    S = S.view(ks, ntr, rg, spk, ap * 512).permute(*pm).reshape(ks * ntr * spk, rg * ap * 512)
    N = N.view(torch.uint8).view(ks, ntr, rg, spk, ap * 256).permute(*pm).reshape(ks * ntr * spk, rg * ap * 256)
    return torch.cat([S, N], 1).contiguous(), hdr, scale


def _pack22(c, ap, E):
    """_compress output -> gemv22 stages, tile-major (u = 128-row tile * S + ks, S = k / 32ap): [8 row groups x ap atoms x
    32 lanes x 16 B sign|mantissa][same x 8 B codes][E escapes: int32 (row in tile | k in stage << 8, true / scale bits),
    padding (-1, 0)]. None if a stage has more than E escapes."""
    code, sm, hdr, scale = c
    n, k = code.shape
    kc = 32 * ap
    P, _, _ = _pack19(c, kc, ap, 8, 1)  # tm: (tile, ks, st), one stage per k-chunk
    ks, ntr = k // kc, n // 128
    EV = torch.full((ks * ntr, E, 2), -1, dtype=torch.int32, device=code.device)
    EV[:, :, 1] = 0
    rows, slots = (hdr[:, :, 1] != 2.0 ** -15).nonzero(as_tuple=True)
    if len(rows):
        col = hdr[rows, slots, 0].view(torch.int32).long()
        u = (rows // 128) * ks + col // kc
        o = torch.argsort(u * (1 << 20) + rows * 8 + slots)
        rows, col, u, hy = rows[o], col[o], u[o], hdr[rows[o], slots[o], 1]
        rank = torch.arange(len(u), device=code.device) - torch.searchsorted(u, u)
        if int(rank.max()) >= E:
            return None
        EV[u, rank, 0] = ((rows % 128) | ((col % kc) << 8)).int()
        EV[u, rank, 1] = hy.view(torch.int32)
    return torch.cat([P, EV.view(torch.uint8).view(ks * ntr, -1)], 1).contiguous(), scale


def prepare(weights):
    st = {name: w.contiguous() for name, w in weights.items()}
    if _g22 is not None:
        for name, cfgs in BF16_G22.items():
            c = _compress(weights[name])
            if c is None:
                continue
            for cf in cfgs:
                if ("g22", name, cf[1], cf[3]) not in st:
                    pk = _pack22(c, cf[1], cf[3])
                    if pk is not None:
                        st[("g22", name, cf[1], cf[3])] = pk
        st["ws22"] = torch.zeros(256 * 8192, dtype=torch.float32, device=weights["o"].device)
    if _g19 is not None:
        for name in BF16_G19:
            cfgs = BF16_G19[name] + DIAG19.get(name, [])
            c = _compress(weights[name])
            if c is None:
                continue
            for cf in cfgs:
                if ("g19", name, cf[1], cf[2], cf[7]) not in st:
                    st[("g19", name, cf[1], cf[2], cf[7])] = _pack19(c, cf[1], cf[2], cf[7])
            for cf in [c_ for v_ in DIAG20.values() for c_ in v_] if _g20 else []:
                if ("g19", name, cf[1], cf[2], 8) not in st and weights[name].shape[1] % cf[1] == 0:
                    st[("g19", name, cf[1], cf[2], 8)] = _pack19(c, cf[1], cf[2], 8)
        st["ws19"] = torch.zeros(32 * 8192, dtype=torch.float32, device=weights["o"].device)
        st["ctr19"] = torch.zeros(2, dtype=torch.int32, device=weights["o"].device)  # [1] = gemv19's epoch counter
    for name, cfgs in BF16_G12.items():
        for c in cfgs:
            if (name, c[1]) not in st:
                st[(name, c[1])] = _pack12(weights[name], 1024 * c[1])
    if _mod is not None:
        for name, cfgs in list(BF16_G14.items()) + list(BF16_G15.items()):
            for c in cfgs:
                rows = 1024 if name == "unpooler" and c[0] <= 8 else weights[name].shape[0]
                if ("g14", name, c[1], rows) not in st:
                    pk = _pack14(weights[name][:rows], 1024 * c[1])
                    if pk is not None:
                        st[("g14", name, c[1], rows)] = pk
        for name, cfgs in BF16_G16.items():
            for c in cfgs:
                if ("g16", name, c[1]) not in st:
                    pk = _pack16(weights[name], 1024 * c[1])
                    if pk is not None:
                        st[("g16", name, c[1])] = pk
        for name in BF16_G13:
            pk = _pack13(weights[name][:1024] if name == "unpooler" else weights[name])
            if pk is not None:
                st[("g13", name)] = pk
    r = weights["router"]
    hi = r.to(torch.bfloat16)
    lo = (r - hi.float()).to(torch.bfloat16)
    st["router_hi_t"] = hi.t().contiguous()
    st["router_lo_t"] = lo.t().contiguous()
    st["router_hl_t"] = torch.cat([hi, lo], 0).t().contiguous()  # [k, 512]: one GEMM gives both halves
    # [stripe][k-tile][hi/lo][lane][8]: a 16x16 tile's mma A fragments, 512 contiguous bytes
    st["router_hl_p"] = torch.stack([_pack(hi), _pack(lo)], 2).contiguous()
    st["router_hl2"] = torch.stack([hi, lo]).contiguous()  # [2, 256, k] for router3
    # Split-K workspace and per-row-block counters: always left zeroed by the kernels.
    st["ws"] = torch.zeros(32 * 8192, dtype=torch.float32, device=r.device)
    st["cnt"] = torch.zeros(1024, dtype=torch.int32, device=r.device)
    # gemv12's own: FP32 split-K partials (overwritten each call, not zeroed) and per-rg counters (left zeroed)
    st["ws12"] = torch.empty(8 * 32 * 8192, dtype=torch.float32, device=r.device)
    st["cnt12"] = torch.zeros(1024, dtype=torch.int32, device=r.device)
    st["ltws"] = torch.empty(LT_WS_BYTES, dtype=torch.uint8, device=r.device)
    if _G30:
        for name in sorted({l for l, _m in BF16_G30}):
            n, k = weights[name].shape
            if n % 64 or k % 128:
                continue
            c = _compress(weights[name])
            if c is None:
                continue
            code, sm, hdr, scale = c
            v = hdr[:, :, 1]
            if not torch.equal(v.bfloat16().float(), v) or bool(((v != 0) & (v.abs() < 2.0 ** -126)).any()):
                print("g30 dropped (escape value not exact in BF16):", name)  # g30 patches escapes as BF16 A halves
                continue
            st[("g30", name)] = (_pack30(code, sm), hdr.contiguous(), scale)
            for (l_, _m), cs in BF16_G30.items():
                for c_ in cs:
                    if l_ == name and len(c_) > 3 and ("g30gx", name, c_[0]) not in st:
                        hb = _g30_gx_tables(hdr.contiguous(), k, c_[0])
                        if hb is not None:
                            st[("g30gx", name, c_[0])] = hb
                        else:
                            print("g30 GX records do not fit:", name, c_[0])
    st["pick"] = _tune(st)
    _diag30k(st)
    _diag30e(st)
    _diag19(st)
    _diag19q(st)
    _diag20(st)
    _diag19_stream()
    _diag19d(st)
    _diag24(st)
    _diag27(st)
    _diag61(st)
    _diag_slab(st)
    _diag_ext(st)
    _diag_lte(st)
    st["rpick"] = _tune_router(st)
    return st


def _run(st, layer, x, out, cand):
    """One BF16 projection by candidate: None = F.linear, ("g12", cfg) = dev/gemv12.cu, ("lt", cfg) = cuBLASLt."""
    if cand is None:
        return F.linear(x, st[layer][:out.shape[1]])
    kind, c = cand
    if kind == "g10":
        _mod.g10_gemv(st[layer], x, out, *c[2:])  # W[:n] is the first n rows, in place
    elif kind == "g13c":
        sm, nib, hdr, scale = st[("g13", layer)]
        _g13c.g13_gemv(sm, nib, hdr, x, out, scale, *c[2:])
    elif kind == "g13":
        sm, nib, hdr, scale = st[("g13", layer)]
        _mod.g13_gemv(sm, nib, hdr, x, out, scale, *c[2:])
    elif kind == "g21":
        sm, nib, hdr, scale = st[("g13", layer)]
        _mod.g21_gemv(sm, nib, hdr, x, out, scale, *c[2:], _SMS, st["cnt"][:0], 0, st[("tk21", layer, c)])
    elif kind == "g14":
        rows = out.shape[1]
        sw, nw, hdr, scale = st[("g14", layer, c[1], rows)]
        _mod.g14_gemm(sw, nw, hdr, scale, x, out, st["ws12"], st["cnt12"], rows // 8, *c[1:])
    elif kind == "g15":
        rows = out.shape[1]
        sw, nw, hdr, scale = st[("g14", layer, c[1], rows)]  # same layout as gemv14
        _mod.g15_gemm(sw, nw, hdr, scale, x, out, rows // 8, *c[1:])
    elif kind == "g16":
        sw, nw, hdr, scale = st[("g16", layer, c[1])]
        _mod.g16_gemm(sw, nw, hdr, scale, x, out, out.shape[1] // 16, *c[1:])
    elif kind == "g19":
        P, hdr, scale = st[("g19", layer, c[1], c[2], c[7])]
        o32 = st["ws19"][:out.numel()].view(out.shape)
        _g19.g19_gemm(P, hdr, scale, x, o32, st["ctr19"], c[1], c[2], c[3], 1, _SMS, 0, 2, c[4], *c[5:])
        _g19.g19_cvt(o32, out)
    elif kind == "g22":
        P, scale = st[("g22", layer, c[1], c[3])]
        o32 = st["ws22"][:out.numel()].view(out.shape)
        _g22.g22_gemm(P, scale, x, o32, c[1], c[2], c[3], _SMS, 0, 0)
        _g19.g19_cvt(o32, out) if _g19 is not None else out.copy_(o32)
    elif kind == "g17":
        _mod.g17_gemv(st[layer], x, out, *c)  # W[:n] prefix rows, in place
    elif kind == "tg2":
        _tg2.tg2(x, st[layer][:out.shape[1]], out, False)  # W[:n] prefix rows, BF16 out
    elif kind == "g30":
        P, hdr, scale = st[("g30", layer)]
        if len(c) > 4:
            hdr = st[("g30gx", layer, c[1])]
        _G30[c].g30_gemm(x, P, hdr, out, scale, 0)
    elif kind == "g12":
        _mod.g12_gemm(st[(layer, c[1])], x, out, st["ws12"], st["cnt12"], st[layer].shape[0] // 8, *c[1:])
    else:
        _lt.lt_gemm(st[layer], x, out, st["ltws"], list(c))
    return out


def _tune(st, reps=25):
    """Per (layer, m): time F.linear, the native gemv10/gemv13 (m=1) and gemv12 configs and each BF16_LT config as CUDA-graph replays with a
    cold L2 (read flush, as the judge) on random x; keep the fastest. Only shapes and weights are used."""
    picks = {}
    flush = torch.zeros(256 << 20, dtype=torch.uint8, device=st["qkv_gate"].device).view(torch.int64)
    sink = torch.zeros((), dtype=torch.int64, device=flush.device)
    g = torch.Generator(device=flush.device)
    g.manual_seed(0)
    shapes = {}
    if _mod is not None:
        for layer, cfgs in BF16_G10.items():
            for c in cfgs:
                shapes.setdefault((layer, c[0]), []).append(("g10", c))
        for layer, cfgs in BF16_G13C.items() if _g13c is not None else []:
            if ("g13", layer) in st:
                for c in cfgs:
                    shapes.setdefault((layer, c[0]), []).append(("g13c", c))
        for layer, cfgs in BF16_G13.items():
            if ("g13", layer) in st:
                for c in cfgs:
                    shapes.setdefault((layer, c[0]), []).append(("g13", c))
        for layer, cfgs in BF16_G14.items():
            for c in cfgs:
                rows = 1024 if layer == "unpooler" and c[0] <= 8 else st[layer].shape[0]
                if ("g14", layer, c[1], rows) in st:
                    shapes.setdefault((layer, c[0]), []).append(("g14", c))
        for layer, cfgs in BF16_G15.items():
            for c in cfgs:
                rows = 1024 if layer == "unpooler" and c[0] <= 8 else st[layer].shape[0]
                if ("g14", layer, c[1], rows) in st:
                    shapes.setdefault((layer, c[0]), []).append(("g15", c))
        for layer, cfgs in BF16_G16.items():
            for c in cfgs:
                if ("g16", layer, c[1]) in st:
                    shapes.setdefault((layer, c[0]), []).append(("g16", c))
        for layer, cfgs in BF16_G12.items():
            for c in cfgs:
                shapes.setdefault((layer, c[0]), []).append(("g12", c))
        for layer, cfgs in BF16_G17.items():
            for c in cfgs:
                shapes.setdefault((layer, c[0]), []).append(("g17", c))
        for layer, cfgs in BF16_G21.items():
            if ("g13", layer) in st:
                for c in cfgs:
                    st[("tk21", layer, c)] = torch.zeros(1, dtype=torch.int64, device=st[layer].device)
                    shapes.setdefault((layer, c[0]), []).append(("g21", c))
    if _g19 is not None:
        for layer, cfgs in BF16_G19.items():
            for c in cfgs:
                if ("g19", layer, c[1], c[2], c[7]) in st:
                    shapes.setdefault((layer, c[0]), []).append(("g19", c))
    if _g22 is not None:
        for layer, cfgs in BF16_G22.items():
            for c in cfgs:
                if ("g22", layer, c[1], c[3]) in st:
                    shapes.setdefault((layer, c[0]), []).append(("g22", c))
    if _tg2 is not None:
        for layer, cfgs in BF16_TG2.items():
            for c in cfgs:
                shapes.setdefault((layer, c[0]), []).append(("tg2", c))
    for (layer, m), cfgs in BF16_G30.items():
        if ("g30", layer) in st:
            k = st[layer].shape[1]
            for c in cfgs:
                if _G30.get((m, *c)) is not None and k // 64 >= 4 * c[0] and (len(c) < 4 or ("g30gx", layer, c[0]) in st):
                    shapes.setdefault((layer, m), []).append(("g30", (m, *c)))
    if _lt is not None:
        for key, cfgs in BF16_LT.items():
            shapes.setdefault(key, []).extend(("lt", c) for c in cfgs)
    for (layer, m), cfgs in sorted(shapes.items()):
        w = st[layer]
        k = w.shape[1]
        n = 1024 if layer == "unpooler" and m <= 8 else w.shape[0]
        x = torch.randn(m, k, generator=g, device=w.device).to(torch.bfloat16)
        ref = F.linear(x, w[:n]).float()
        out = torch.empty(m, n, dtype=torch.bfloat16, device=w.device)
        cands = [None] if m > 1 else []  # m=1: gemv10 always beat cuBLAS on the judge; ties picked F.linear by noise
        for c in cfgs:
            if c[0] == "lt" and _lt.lt_gemm(w, x, out, st["ltws"], list(c[1])) != 0:
                continue
            _run(st, layer, x, out, c)
            torch.cuda.synchronize()
            if ((out.float() - ref).norm() / ref.norm()).item() < 1e-3:
                cands.append(c)
        if not cands:
            continue
        graphs = []
        for c in cands:  # one graph per candidate: times the kernel as the judge replays it, no host overhead
            gr = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                with torch.cuda.graph(gr, stream=s):
                    _run(st, layer, x, out, c)
            torch.cuda.current_stream().wait_stream(s)
            graphs.append(gr)
        torch.cuda.synchronize()
        ev = [[(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in cands]
              for _ in range(reps)]
        for r in range(reps):
            for i in range(len(cands)):
                torch.sum(flush, 0, out=sink)  # cold L2 by reading, as the judge does
                ev[r][i][0].record()
                graphs[i].replay()
                ev[r][i][1].record()
        torch.cuda.synchronize()
        med = [sorted(ev[r][i][0].elapsed_time(ev[r][i][1]) for r in range(reps))[reps // 2] for i in range(len(cands))]
        best = min(range(len(cands)), key=med.__getitem__)
        # n16: near-ties (within 4%) re-timed with 4x the reps: the judge's 4 runs flipped picks by noise (o m8/m16)
        near = sorted((i for i in range(len(cands)) if med[i] <= 1.04 * med[best]), key=med.__getitem__)[:4]
        if len(near) > 1:
            r2 = 4 * reps
            ev2 = [[(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in near]
                   for _ in range(r2)]
            for r in range(r2):
                for j, i in enumerate(near):
                    torch.sum(flush, 0, out=sink)
                    ev2[r][j][0].record()
                    graphs[i].replay()
                    ev2[r][j][1].record()
            torch.cuda.synchronize()
            for j, i in enumerate(near):
                med[i] = sorted(ev2[r][j][0].elapsed_time(ev2[r][j][1]) for r in range(r2))[r2 // 2]
            best = min(near, key=med.__getitem__)
        if cands[best] is not None:
            picks[(layer, m)] = cands[best]
        print(f"tune {layer} m{m}: best {cands[best]} {med[best] * 1e3:.2f} us; "
              + " ".join(f"{'F' if c is None else c[0] + ('' if c[0] != 'g30' else 'S%dW%dN%d' % c[1][1:4] + ('GX' if len(c[1]) > 4 else ''))}:{t * 1e3:.2f}"
                         for c, t in zip(cands, med)))
    del flush
    return picks


def _g30_gx_tables(hdr, K, S, gxc=64, gxr=512):
    """hdr [N][ESC] float32 pairs (col bits, v) as built by _compress; returns int32 [N / 64 * S][rlen] on hdr's
    device or None if a record does not fit (more than gxc distinct escape columns in one CTA: use the non-GX g30)."""
    h = hdr.detach().cpu()
    N, E = h.shape[0], h.shape[1]
    cols = h[:, :, 0].contiguous().view(torch.int32)
    vb = h[:, :, 1].contiguous().view(torch.int32)
    valid = h[:, :, 1] != 2.0 ** -15
    KS = K // 64
    recs = []
    for g in range(N // 64):
        for y in range(S):
            i0, i1 = 2 * (y * (KS // 2) // S), 2 * ((y + 1) * (KS // 2) // S)
            c = cols[g * 64:(g + 1) * 64]
            ok = valid[g * 64:(g + 1) * 64] & (c >= i0 * 64) & (c < i1 * 64)
            rows, es = ok.nonzero(as_tuple=True)  # row-major = header order
            cc = c[rows, es]
            uc = torch.unique(cc)  # sorted
            if len(uc) > gxc:
                return None
            slot = torch.searchsorted(uc, cc)
            starts = [int((rows < 16 * w).sum()) for w in range(4)] + [len(rows)]
            last = torch.ones_like(rows, dtype=torch.bool)
            last[:-1] = rows[1:] != rows[:-1]
            ent = torch.stack([slot | (rows << 16) | (last.int() << 22) | ((cc & 7) << 28), vb[g * 64:(g + 1) * 64][rows, es]], 1).reshape(-1)
            recs.append((uc, starts, ent))
    emax = max(len(r[2]) for r in recs)
    rlen = (8 + gxc + emax + 3) // 4 * 4
    if rlen > gxr:
        return None
    out = torch.zeros(len(recs), rlen, dtype=torch.int32)
    for i, (uc, starts, ent) in enumerate(recs):
        out[i, 0] = len(uc)
        out[i, 1:6] = torch.tensor(starts, dtype=torch.int32)
        out[i, 8:8 + len(uc)] = uc
        out[i, 8 + gxc:8 + gxc + len(ent)] = ent
    return out.to(hdr.device)


def _diag30e(st):
    """Print only: g30 vs g30 without the escape path (wrong output, timing only), standalone and after a 3 us predecessor."""
    if _g19 is None:
        return
    g = torch.Generator(device="cuda")
    g.manual_seed(8)
    m = DIAG30E_M
    for layer, S in DIAG30E.items():
        a, b = _G30.get((m, S, 16, 16)), _G30.get((m, S, 16, 16, DIAG30E_V))
        if ("g30", layer) not in st or a is None or b is None:
            continue
        P_, hdr, scale = st[("g30", layer)]
        n, k = st[layer].shape
        hb = _g30_gx_tables(hdr, k, S) if "G30_GX=1" in DIAG30E_V else hdr
        if hb is None:
            print(f"diag30e {layer}: GX records do not fit")
            continue
        xs = torch.randn(m, k, generator=g, device="cuda").to(torch.bfloat16)
        x = xs.clone()
        out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
        ref = F.linear(x, st[layer]).float()
        b.g30_gemm(x, P_, hb, out, scale, 0)
        eb = ((out.float() - ref).norm() / ref.norm()).item()
        a.g30_gemm(x, P_, hdr, out, scale, 0)
        ea = ((out.float() - ref).norm() / ref.norm()).item()
        print(f"diag30e {layer} m{m} S{S}: rel err g30 {ea:.2e} {DIAG30E_V} {eb:.2e}")
        fns = [lambda pdl: a.g30_gemm(x, P_, hdr, out, scale, pdl), lambda pdl: b.g30_gemm(x, P_, hb, out, scale, pdl)]
        med0 = _graph_medians([lambda: None] + [lambda fn=fn: fn(0) for fn in fns] * 2)
        medp = _graph_medians([lambda: _g19.g19_pred(xs, x, 3000)] + [lambda fn=fn: (_g19.g19_pred(xs, x, 3000), fn(1)) for fn in fns] * 2)
        print(f"diag30e {layer} m{m} S{S}: standalone (+us over an empty graph {med0[0] * 1e3:.2f}) g30 "
              + "/".join(f"{(t - med0[0]) * 1e3:.2f}" for t in med0[1::2]) + f" {DIAG30E_V} " + "/".join(f"{(t - med0[0]) * 1e3:.2f}" for t in med0[2::2])
              + f" || after 3 us pred (+us over {medp[0] * 1e3:.2f}) g30 " + "/".join(f"{(t - medp[0]) * 1e3:.2f}" for t in medp[1::2])
              + f" {DIAG30E_V} " + "/".join(f"{(t - medp[0]) * 1e3:.2f}" for t in medp[2::2]))


def _diag30k(st):
    """Print only: g30k (FP32 slabs / BF16 via ticket) per grid vs g30 S-split, standalone and after a 3 us predecessor."""
    if not _G30K or _g19 is None:
        return
    g = torch.Generator(device="cuda")
    g.manual_seed(7)
    for layer, grids in DIAG30K.items():
        if ("g30", layer) not in st:
            continue
        P_, hdr, scale = st[("g30", layer)]
        n, k = st[layer].shape
        KP = k // 128
        tick = torch.zeros(n // 64, dtype=torch.int32, device="cuda")
        slab = torch.empty(n // 64 * 6 * 16 * 128, dtype=torch.float32, device="cuda")
        noz = torch.empty(0, dtype=torch.float32, device="cuda")
        for m in (16, 32):
            xs = torch.randn(m, k, generator=g, device="cuda").to(torch.bfloat16)
            x = xs.clone()
            out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
            ref = F.linear(x, st[layer]).float()
            gs = [G for G in grids if (n // 64 * KP) // G >= 2 and KP // ((n // 64 * KP) // G) + 2 <= 6]
            o32 = {G: torch.zeros(KP // ((n // 64 * KP) // G) + 2, m, n, dtype=torch.float32, device="cuda") for G in gs}
            fns, names = [], []
            S = DIAG30K_S[layer]
            lib = _G30.get((m, S, 16, 16))
            if lib is not None:
                fns.append(lambda pdl, lib=lib: lib.g30_gemm(x, P_, hdr, out, scale, pdl)); names.append(f"g30S{S}")
            for G in gs:
                if _G30K.get((m, 1)) is not None:
                    fns.append(lambda pdl, G=G: _G30K[(m, 1)].g30k_gemm(x, P_, hdr, o32[G], scale, pdl, noz, noz, G)); names.append(f"kF{G}")
                if _G30K.get((m, 0)) is not None:
                    fns.append(lambda pdl, G=G: _G30K[(m, 0)].g30k_gemm(x, P_, hdr, out, scale, pdl, slab, tick, G)); names.append(f"kB{G}")
            ok = []
            for nm, fn in zip(names, fns):
                fn(0)
                torch.cuda.synchronize()
                r = (o32[int(nm[2:])].sum(0) if nm.startswith("kF") else out).bfloat16().float()
                ok.append(((r - ref).norm() / ref.norm()).item() < 1e-3)
            pick = st["pick"].get((layer, m))
            med0 = _graph_medians([lambda: _run(st, layer, x, out, pick)] + [lambda fn=fn: fn(0) for fn in fns])
            medp = _graph_medians([lambda: _g19.g19_pred(xs, x, 3000)] + [lambda fn=fn: (_g19.g19_pred(xs, x, 3000), fn(1)) for fn in fns])
            print(f"diag30k {layer} m{m}: standalone pick {pick} {med0[0] * 1e3:.2f} | "
                  + " ".join(f"{nm}:{t * 1e3:.2f}{'' if o else '(BAD)'}" for nm, t, o in zip(names, med0[1:], ok))
                  + f" || after 3 us pred (+us over {medp[0] * 1e3:.2f}): "
                  + " ".join(f"{nm}:{(t - medp[0]) * 1e3:.2f}" for nm, t in zip(names, medp[1:])))
            del o32


def _diag19(st):
    """Print only (no effect on picks): the layer's case, a 3 us predecessor that writes x and lets its PDL dependent
    launch at once, then gemv19 (FP32 out, npre = ns: W streams during the predecessor) vs the picked kernel."""
    if _g19 is None:
        return
    g = torch.Generator(device="cuda")
    g.manual_seed(3)
    for layer, cfgs in BF16_G19.items():
        for c in cfgs + DIAG19.get(layer, []):
            if ("g19", layer, c[1], c[2], c[7]) not in st:
                continue
            m, w = c[0], st[layer]
            xs = torch.randn(m, w.shape[1], generator=g, device="cuda").to(torch.bfloat16)
            x = xs.clone()
            out = torch.empty(m, w.shape[0], dtype=torch.bfloat16, device="cuda")
            P, hdr, scale = st[("g19", layer, c[1], c[2], c[7])]
            o32 = st["ws19"][:out.numel()].view(out.shape)
            pick = st["pick"].get((layer, m))
            med = _graph_medians([lambda: _g19.g19_pred(xs, x, 3000),
                                  lambda: (_g19.g19_pred(xs, x, 3000), _run(st, layer, x, out, pick)),
                                  lambda: (_g19.g19_pred(xs, x, 3000), _g19.g19_gemm(P, hdr, scale, x, o32, st["ctr19"], c[1], c[2], c[3], 1, _SMS, 0, 2, c[3], *c[5:]))])
            if c not in cfgs:
                med[1] = float("nan")
            print(f"diag19 {layer} m{m} {c}: after a 3 us PDL predecessor ({med[0] * 1e3:.2f}): pick {pick} +{(med[1] - med[0]) * 1e3:.2f} us, "
                  f"g19 FP32 out +{(med[2] - med[0]) * 1e3:.2f} us")


def _diag19d(st):
    """Print only: per DIAG19D config and grid, us over the 3 us predecessor for each DIAG19D_DEFS build."""
    if _g19 is None or not DIAG19D:
        return
    mods = []
    for i, defs in enumerate(DIAG19D_DEFS):
        try:
            mods.append(load_inline(
                name="muse_proj_g19d%d" % i, cpp_sources="void g19_gemm(torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor, "
                "torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);",
                cuda_sources="".join("#define %s\n" % d for d in defs.split()) + "#define G19_ONLY2\n" + _G19_SRC.replace("G19_CONFIGS", " ".join(
                    "G19_CASE(%d,%d,%d,%d,1,%d,%d,%d,%d)" % (c[:4] + c[5:]) for c in sorted({c for v in DIAG19D.values() for c in v}))),
                functions=["g19_gemm"], extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"]))
        except Exception as e:  # noqa: BLE001
            print("diag19d build failed:", defs, e)
            mods.append(None)
    g = torch.Generator(device="cuda")
    g.manual_seed(5)
    for layer, cfgs in DIAG19D.items():
        for c in cfgs:
            if ("g19", layer, c[1], c[2], c[7]) not in st:
                continue
            m, w = c[0], st[layer]
            xs = torch.randn(m, w.shape[1], generator=g, device="cuda").to(torch.bfloat16)
            x = xs.clone()
            P, hdr, scale = st[("g19", layer, c[1], c[2], c[7])]
            o32 = st["ws19"][:m * w.shape[0]].view(m, w.shape[0])
            for grid in DIAG19D_GRIDS:
                fns, names = [lambda: _g19.g19_pred(xs, x, 3000)], []
                for defs, mod in zip(DIAG19D_DEFS, mods):
                    if mod is None:
                        continue
                    ctr = torch.zeros(2, dtype=torch.int32, device="cuda")  # one epoch counter per (build, grid)
                    fns.append(lambda mod=mod, ctr=ctr, grid=grid: (_g19.g19_pred(xs, x, 3000), mod.g19_gemm(
                        P, hdr, scale, x, o32, ctr, c[1], c[2], c[3], 1, grid, 0, 2, c[3], *c[5:])))
                    names.append(defs.replace("G19_", "") or "main")
                try:
                    med = _graph_medians(fns)
                except Exception as e:  # noqa: BLE001
                    print("diag19d", layer, m, grid, "failed:", e)
                    continue
                print(f"diag19d {layer} m{m} {c} G{grid} (pred {med[0] * 1e3:.2f}):", " | ".join(
                    f"{nm_} +{(t - med[0]) * 1e3:.2f}" for nm_, t in zip(names, med[1:])))


def _diag24(st):
    """Print only: gemv24 per DIAG24 config and grid, us over the 3 us predecessor (as _diag19d), + max rel err."""
    if _g19 is None or not DIAG24:
        return
    try:
        mod = load_inline(name="muse_proj_g24", cpp_sources="void g24_gemm(torch::Tensor, torch::Tensor, double, torch::Tensor, "
                          "torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);",
                          cuda_sources=_G24_SRC.replace("G24_CONFIGS", " ".join("G24_CASE(%d,%d,%d,%d,%d,%d,%d)" % c for c in sorted(
                              {c for v in DIAG24.values() for c in v}))),
                          functions=["g24_gemm"], extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"])
    except Exception as e:  # noqa: BLE001
        print("diag24 build failed:", e)
        return
    g = torch.Generator(device="cuda")
    g.manual_seed(5)
    for layer, cfgs in DIAG24.items():
        w = st[layer]
        c_ = _compress(w)
        if c_ is None:
            continue
        for c in cfgs:
            m, kc, ap, ns, wpb, xb, r = c
            if w.shape[1] % kc:
                continue
            xs = torch.randn(m, w.shape[1], generator=g, device="cuda").to(torch.bfloat16)
            x = xs.clone()
            P, hdr, scale = _pack19(c_, kc, ap, r & 0xff)
            o32 = torch.empty(m, w.shape[0], dtype=torch.float32, device="cuda")
            try:
                ref = x.float() @ w.float().T
                ctr = torch.zeros(2, dtype=torch.int32, device="cuda")
                mod.g24_gemm(P, hdr, scale, x, o32, ctr, kc, ap, ns, wpb, xb, _SMS, ns, r)
                torch.cuda.synchronize()
                err = ((o32 - ref).norm() / ref.norm()).item()
                fns = [lambda: _g19.g19_pred(xs, x, 3000)]
                for grid in DIAG24_GRIDS:
                    ctr = torch.zeros(2, dtype=torch.int32, device="cuda")  # one epoch counter per grid
                    fns.append(lambda ctr=ctr, grid=grid: (_g19.g19_pred(xs, x, 3000), mod.g24_gemm(P, hdr, scale, x, o32, ctr, kc, ap, ns, wpb, xb, grid, ns, r)))
                med = _graph_medians(fns)
            except Exception as e:  # noqa: BLE001
                print("diag24", layer, c, "failed:", e)
                continue
            print(f"diag24 {layer} m{m} {c} (pred {med[0] * 1e3:.2f}, rel err {err:.1e}):", " | ".join(
                f"G{grid} +{(t - med[0]) * 1e3:.2f}" for grid, t in zip(DIAG24_GRIDS, med[1:])))


def _diag61(st):
    """Print only (s61): g24 layer cfgs x DIAG61_BITS variants, us over a 3 us PDL predecessor, + rel err per variant."""
    if _g19 is None or not DIAG61:
        return
    try:
        mod = load_inline(name="muse_proj_g24s61", cpp_sources="void g24_gemm(torch::Tensor, torch::Tensor, double, torch::Tensor, "
                          "torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, torch::Tensor);",
                          cuda_sources=_G24S61_SRC.replace("G24_CONFIGS", " ".join("G24_CASE(%d,%d,%d,%d,%d,%d,%d)" % c for c in sorted({c for _, c, _ in DIAG61}))),
                          functions=["g24_gemm"], extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"])
    except Exception as e:  # noqa: BLE001
        print("diag61 build failed:", e)
        return
    g = torch.Generator(device="cuda")
    g.manual_seed(61)
    zo = torch.empty(0, device="cuda")
    for layer, c, grid in DIAG61:
        w = st[layer]
        c_ = _compress(w)
        if c_ is None:
            continue
        m, kc, ap, ns, wpb, xb, r = c
        xs = torch.randn(m, w.shape[1], generator=g, device="cuda").to(torch.bfloat16)
        x = xs.clone()
        P, hdr, scale = _pack19(c_, kc, ap, r & 0xff)
        ref = x.float() @ w.float().T
        fns, errs = [lambda: _g19.g19_pred(xs, x, 3000)], []
        try:
            for nm, bits in DIAG61_BITS:
                o32 = torch.empty(m, w.shape[0], dtype=torch.float32, device="cuda")
                ctr = torch.zeros(2, dtype=torch.int32, device="cuda")
                a_ = (kc, ap, ns, wpb, xb, grid, ns | bits, r)
                mod.g24_gemm(P, hdr, scale, x, o32, ctr, *a_, zo)
                torch.cuda.synchronize()
                errs.append(((o32 - ref).norm() / ref.norm()).item())
                fns.append(lambda o32=o32, ctr=ctr, a_=a_: (_g19.g19_pred(xs, x, 3000), mod.g24_gemm(P, hdr, scale, x, o32, ctr, *a_, zo)))
            med = _graph_medians(fns)
        except Exception as e:  # noqa: BLE001
            print("diag61", layer, c, "failed:", e)
            continue
        print(f"diag61 {layer} m{m} {c} G{grid} (pred {med[0] * 1e3:.2f}):", " | ".join(
            f"{nm} +{(t - med[0]) * 1e3:.2f} e{er:.0e}" for (nm, _), t, er in zip(DIAG61_BITS, med[1:], errs)))


def _diag27(st):
    """Print only: gemv27 per DIAG27 config vs gemv24 (DIAG27_G24), F.linear and the pick, us over a 3 us PDL predecessor."""
    if _g19 is None or not DIAG27:
        return
    try:
        c24 = sorted(set(DIAG27_G24.values()))
        s27 = _G27_SRC.replace("G27_CONFIGS", " ".join("G27_CASE(%d,%d,%d,%d,%d,%s)" % (c[:5] + ("true" if c[5] else "false",)) for c in sorted(
            {c for v in DIAG27.values() for c in v})))
        mods = {}
        for i, mc in enumerate(DIAG27_MC):  # s58: one module per G27_MC; the first also carries g24
            mods[mc] = load_inline(name="muse_proj_g27_mc%d" % mc, cpp_sources=("void g24_gemm(torch::Tensor, torch::Tensor, double, torch::Tensor, "
                                   "torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t); " if i == 0 else "") +
                                   "void g27_gemm(torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, torch::Tensor);",
                                   cuda_sources=(_G24_SRC.replace("G24_CONFIGS", " ".join("G24_CASE(%d,%d,%d,%d,%d,%d,%d)" % c for c in c24)) + "\n" if i == 0 else "") +
                                   "#define G27_MC %d\n" % mc + s27,
                                   functions=(["g24_gemm"] if i == 0 else []) + ["g27_gemm"], extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"])
        mod = mods[DIAG27_MC[0]]
    except Exception as e:  # noqa: BLE001
        print("diag27 build failed:", e)
        return
    g = torch.Generator(device="cuda")
    g.manual_seed(6)
    for layer, cfgs in DIAG27.items():
        w = st[layer]
        n, k = w.shape
        c_ = _compress(w)
        if c_ is None:
            continue
        for m in sorted({c[0] for c in cfgs}):
            xs = torch.randn(m, k, generator=g, device="cuda").to(torch.bfloat16)
            x = xs.clone()
            ref = x.float() @ w.float().T
            out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
            pick = st["pick"].get((layer, m))
            fns = [lambda: _g19.g19_pred(xs, x, 3000), lambda: (_g19.g19_pred(xs, x, 3000), F.linear(x, w)),
                   lambda: (_g19.g19_pred(xs, x, 3000), _run(st, layer, x, out, pick))]
            names, errs = ["cublas", f"pick {pick[0] if pick else None}"], ["", ""]
            c4 = DIAG27_G24.get((layer, m))
            try:
                if c4:
                    P4, h4, s4 = _pack19(c_, c4[1], c4[2], c4[6] & 0xff)
                    o4 = torch.empty(m, n, dtype=torch.float32, device="cuda")
                    ctr = torch.zeros(2, dtype=torch.int32, device="cuda")
                    a4 = c4[1:6] + (_SMS, c4[3], c4[6])
                    mod.g24_gemm(P4, h4, s4, x, o4, ctr, *a4)
                    torch.cuda.synchronize()
                    errs.append(f"{((o4 - ref).norm() / ref.norm()).item():.1e}")
                    fns.append(lambda: (_g19.g19_pred(xs, x, 3000), mod.g24_gemm(P4, h4, s4, x, o4, ctr, *a4)))
                    names.append(f"g24 {c4[1:]}")
                for c in cfgs:
                    if c[0] != m:
                        continue
                    kq, ap, ns, rgb, obf = c[1:]
                    P, hdr, scale = _pack19(c_, k // kq, ap, 1)
                    a27 = (kq, ap, ns, rgb, ns if ns < 100 else 99)
                    for mc, md in mods.items():
                        o27 = torch.empty(m, n, dtype=torch.bfloat16 if obf else torch.float32, device="cuda")
                        md.g27_gemm(P, hdr, scale, x, o27, *a27, torch.empty(0, device="cuda"))
                        torch.cuda.synchronize()
                        errs.append(f"{((o27.float() - ref).norm() / ref.norm()).item():.1e}")
                        fns.append(lambda P=P, hdr=hdr, scale=scale, o27=o27, a27=a27, md=md: (_g19.g19_pred(xs, x, 3000), md.g27_gemm(P, hdr, scale, x, o27, *a27, torch.empty(0, device="cuda"))))
                        names.append(f"g27 {c[1:]} mc{mc}")
                med = _graph_medians(fns)
            except Exception as e:  # noqa: BLE001
                print("diag27", layer, m, "failed:", e)
                continue
            print(f"diag27 {layer} m{m} (pred {med[0] * 1e3:.2f}):", " | ".join(
                f"{nm} +{(t - med[0]) * 1e3:.2f}{' err ' + er if er else ''}" for nm, t, er in zip(names, med[1:], errs)))


def _diag19q(st):
    """Print only: qkv by gemv19 mode 3 (BF16 out, no predecessor, tm 0/1) vs the picked kernel; plus max abs diff."""
    if _g19 is None:
        return
    g = torch.Generator(device="cuda")
    g.manual_seed(5)
    for layer, cfgs in DIAG19Q.items():
        c_ = _compress(st[layer])
        if c_ is None:
            continue
        w = st[layer]
        ctr = torch.zeros(4096, dtype=torch.int32, device="cuda")
        for c in cfgs:
            m = c[0]
            x = torch.randn(m, w.shape[1], generator=g, device="cuda").to(torch.bfloat16)
            out = torch.empty(m, w.shape[0], dtype=torch.bfloat16, device="cuda")
            ob = torch.empty_like(out)
            ws = torch.zeros(m, w.shape[0], dtype=torch.float32, device="cuda")
            pick = st["pick"].get((layer, m))
            fns = [lambda: _run(st, layer, x, out, pick)]
            for tm in (0, 1):
                P, hdr, scale = _pack19(c_, c[1], c[2], c[7], tm)
                fns.append(lambda P=P, hdr=hdr, scale=scale, tm=tm: _g19.g19_gemm_bf16(P, hdr, scale, x, ws, ob, ctr, c[1], c[2], c[3], _SMS, c[4], *c[5:], tm))
            med = _graph_medians(fns)
            fns[0](); fns[2](); torch.cuda.synchronize()
            d = (ob.float() - out.float()).abs().max().item()
            print(f"diag19q {layer} m{m} {c}: standalone: pick {pick} {med[0] * 1e3:.2f} us, g19 BF16 out tm0 {med[1] * 1e3:.2f} "
                  f"tm1 {med[2] * 1e3:.2f} us (max |diff| vs pick {d:.3g}, ws clean {int(torch.count_nonzero(ws)) == 0})")
            del P


def _diag20(st):
    """Print only: gemv20 variants vs gemv19 (BF16_G19 configs) in the layer's case (3 us PDL predecessor), H200 us."""
    if _g19 is None or not _g20:
        return
    g = torch.Generator(device="cuda")
    g.manual_seed(5)
    for layer, cfgs19 in BF16_G19.items():
        for m in (16, 32):
            w = st[layer]
            xs = torch.randn(m, w.shape[1], generator=g, device="cuda").to(torch.bfloat16)
            x = xs.clone()
            o32 = st["ws19"][:m * w.shape[0]].view(m, w.shape[0])
            ref = F.linear(xs.float(), w.float())
            fns, names = [lambda: _g19.g19_pred(xs, x, 3000)], []
            for c in cfgs19:
                if c[0] == m and ("g19", layer, c[1], c[2], c[7]) in st:
                    P, hdr, scale = st[("g19", layer, c[1], c[2], c[7])]
                    fns.append(lambda P=P, hdr=hdr, scale=scale, c=c: (_g19.g19_pred(xs, x, 3000), _g19.g19_gemm(P, hdr, scale, x, o32, st["ctr19"], c[1], c[2], c[3], 1, _SMS, 0, 2, c[3], *c[5:])))
                    names.append("g19%s" % (c[1:4],))
            for v, cfgs in DIAG20.items():
                if v not in _g20:
                    continue
                grid = _SMS * (2 if v in "bc" else 1)
                ctr = st.setdefault(("ctr20", v), torch.zeros(2, dtype=torch.int32, device="cuda"))
                for c in cfgs:
                    if c[0] != m or ("g19", layer, c[1], c[2], 8) not in st:
                        continue
                    P, hdr, scale = st[("g19", layer, c[1], c[2], 8)]
                    run = lambda P=P, hdr=hdr, scale=scale, c=c, v=v, grid=grid, ctr=ctr: _g20[v].g20_gemm(P, hdr, scale, x, o32, ctr, c[1], c[2], c[3], 1, grid, 0, 2, c[3], 1, 1, 8, 1)
                    try:
                        o32.fill_(7.0)
                        torch.cuda.synchronize()
                        run()
                        torch.cuda.synchronize()
                        err = ((o32 - ref).norm() / ref.norm()).item()
                    except Exception as e:  # noqa: BLE001
                        print("diag20", layer, m, v, c, "failed:", e)
                        continue
                    fns.append(lambda run=run: (_g19.g19_pred(xs, x, 3000), run()))
                    names.append("g20%s%s err %.1e" % (v, c[1:], err))
            med = _graph_medians(fns)
            print(f"diag20 {layer} m{m} (pred {med[0] * 1e3:.2f}):", " | ".join(f"{nm_} +{(t - med[0]) * 1e3:.2f}" for nm_, t in zip(names, med[1:])))


def _diag_slab(st):
    """Print only: the layer's FP32-consumer option, x @ W^T split over k into ns FP32 slabs by one strided batched
    GEMM (the consumer sums the slabs; no split-K reduce kernel) vs this shape's pick (BF16 out)."""
    g = torch.Generator(device="cuda")
    g.manual_seed(5)
    for layer, ms in DIAG_SLAB.items():
        w = st[layer]
        n, k = w.shape
        for m in ms:
            x = torch.randn(m, k, generator=g, device=w.device).to(torch.bfloat16)
            out = torch.empty(m, n, dtype=torch.bfloat16, device=w.device)
            pick = st["pick"].get((layer, m))
            fns = [lambda: _run(st, layer, x, out, pick) if pick is not None else torch.mm(x, w.t(), out=out),
                   lambda: torch.mm(x, w.t(), out_dtype=torch.float32)]
            nss = [ns for ns in (2, 3, 4, 8) if k % ns == 0]
            for ns in nss:
                kc = k // ns
                fns.append(lambda ns=ns, kc=kc: torch.bmm(x.view(m, ns, kc).permute(1, 0, 2), w.view(n, ns, kc).permute(1, 2, 0),
                                                          out_dtype=torch.float32))
            med = _graph_medians(fns)
            print(f"diagslab {layer} m{m}: pick {pick} {med[0] * 1e3:.2f} mm32 {med[1] * 1e3:.2f}",
                  " ".join(f"bmm{ns}:{t * 1e3:.2f}" for ns, t in zip(nss, med[2:])))


def _diag_ext(st):
    """Print only: which GEMM entry points the judge image's libraries export, and DeepGEMM's BF16 GEMM (if present)
    vs this shape's pick, as graph replays after the cold-L2 flush (+ max abs diff to the pick's output)."""
    import importlib
    import time
    t0 = time.time()
    libs = {}
    for name in ("deep_gemm", "flashinfer", "flashinfer.gemm", "sgl_kernel"):
        try:
            mod = importlib.import_module(name)
            libs[name] = mod
            names = sorted(a for a in dir(mod) if any(s_ in a.lower() for s_ in ("bf16", "gemm", "mm", "sms")))
            print(f"diagext {name} {getattr(mod, '__version__', '?')}: {' '.join(names)[:2000]}")
        except Exception as e:
            print(f"diagext {name} import failed: {type(e).__name__}: {str(e)[:200]}")
    fn = getattr(libs.get("deep_gemm"), "bf16_gemm_nt", None)
    if fn is None or not DIAG_EXT:
        return
    g = torch.Generator(device="cuda")
    g.manual_seed(7)
    for layer, ms in DIAG_EXT.items():
        for m in ms:
            if time.time() - t0 > 240:
                print("diagext: time cap")
                return
            try:
                w = st[layer]
                n, k = w.shape
                x = torch.randn(m, k, generator=g, device=w.device).to(torch.bfloat16)
                out = torch.empty(m, n, dtype=torch.bfloat16, device=w.device)
                ref = torch.empty(m, n, dtype=torch.bfloat16, device=w.device)
                pick = st["pick"].get((layer, m))
                runp = (lambda: _run(st, layer, x, ref, pick)) if pick is not None else (lambda: torch.mm(x, w.t(), out=ref))
                runp()
                torch.cuda.synchronize()
                t1 = time.time()
                fn(x, w, out)
                torch.cuda.synchronize()
                jit = time.time() - t1
                err = (out.float() - ref.float()).abs().max().item()
                med = _graph_medians([runp, lambda: fn(x, w, out)])
                print(f"diagext {layer} m{m}: pick {pick} {med[0] * 1e3:.2f} deep_gemm bf16_gemm_nt {med[1] * 1e3:.2f} us "
                      f"(first call {jit:.1f} s, maxdiff {err:.3g})")
            except Exception as e:
                print(f"diagext {layer} m{m} failed: {type(e).__name__}: {str(e)[:300]}")


# s68: _diag_lte only (print): H200 screen of EVERY algo-66 cuBLASLt config (lt_cands: tiles x cluster x split-K, custom 0,
# workspace <= LT_WS_BYTES 32 MB; the layer has 4 MB) per shape, eager after a read flush with the GPU kept behind the host; the top
# DIAG_LTE_TOP get all custom options, the best of those are re-timed as graph replays (like _tune) vs F.linear and
# this shape's lt_cfg.txt candidates. The H100 sweep (dev/lte.py) chose lt_cfg.txt; the H200 ranks configs differently.
DIAG_LTE = [("pooler", 128), ("unpooler", 128), ("qkv_gate", 128), ("o", 128), ("pooler", 256), ("unpooler", 256),
            ("qkv_gate", 256), ("o", 256), ("qkv_gate", 32), ("o", 32)]
DIAG_LTE = []  # s68 diag done (check f45aa794, NOTES s68: only pooler m128 (66, 124, .., 4) -1.2 us; rest +-0.2)
DIAG_LTE_TOP = 12
DIAG_LTE_CAP = 110.0  # s


def _diag_lte(st):
    import time
    if _lt is None or not DIAG_LTE or not hasattr(_lt, "lt_cands"):
        return
    t0 = time.time()
    flush = torch.zeros(256 << 20, dtype=torch.uint8, device="cuda").view(torch.int64)
    sink = torch.zeros((), dtype=torch.int64, device="cuda")
    ws = st["ltws"]
    g = torch.Generator(device="cuda")
    g.manual_seed(11)

    def screen(w, x, out, cfgs):
        for c in cfgs:  # untimed first launch of each kernel (lazy module load costs ms on the first call)
            _lt.lt_gemm(w, x, out, ws, c)
        times = {}
        for b in range(0, len(cfgs), 48):
            torch.cuda.synchronize()
            torch.cuda._sleep(100_000_000)  # ~55 ms > 48 host-side plan setups: no launch gap is timed
            evs = []
            for c in cfgs[b:b + 48]:
                torch.sum(flush, 0, out=sink)
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                ok = _lt.lt_gemm(w, x, out, ws, c) == 0
                e1.record()
                evs.append((tuple(c), ok, e0, e1))
            torch.cuda.synchronize()
            for c, ok, e0, e1 in evs:
                if ok:
                    times[c] = e0.elapsed_time(e1) * 1e3
        return times

    for layer, m in DIAG_LTE:
        if time.time() - t0 > DIAG_LTE_CAP:
            print("diaglte: time cap")
            break
        try:
            t1 = time.time()
            w = st[layer]
            n, k = w.shape
            x = torch.randn(m, k, generator=g, device=w.device).to(torch.bfloat16)
            out = torch.empty(m, n, dtype=torch.bfloat16, device=w.device)
            ref = F.linear(x, w).float()
            cfgs = _lt.lt_cands(m, n, k, ws.numel(), 0, 0).tolist()
            ta = screen(w, x, out, cfgs)
            top = sorted(ta, key=ta.get)[:DIAG_LTE_TOP]
            tb = screen(w, x, out, [list(c[:6]) + [cu, c[7]] for c in top for cu in range(8)])
            tb.update({c: ta[c] for c in top})
            best = []
            for c in sorted(tb, key=tb.get):
                if len(best) >= 8:
                    break
                if any(c[1] == b[1] and c[3] == b[3] and c[7] == b[7] for b in best):
                    continue  # one custom option per (tile, split, cluster)
                if _lt.lt_gemm(w, x, out, ws, list(c)) != 0:
                    continue
                torch.cuda.synchronize()
                if ((out.float() - ref).norm() / ref.norm()).item() < 1e-3:
                    best.append(c)
            mine = [tuple(c) for c in BF16_LT.get((layer, m), []) if tuple(c) not in best]
            fns = [lambda: torch.mm(x, w.t(), out=out)]  # = F.linear's cuBLAS path
            fns += [(lambda c=c: _lt.lt_gemm(w, x, out, ws, list(c))) for c in best + mine]
            med = _graph_medians(fns)
            print(f"diaglte {layer} m{m}: {len(cfgs)} cfgs, {time.time() - t1:.1f} s; F {med[0] * 1e3:.2f} | new "
                  + " ".join(f"{c}:{t * 1e3:.2f}/{tb[c]:.1f}" for c, t in zip(best, med[1:]))
                  + " | lt_cfg " + " ".join(f"{c}:{t * 1e3:.2f}" for c, t in zip(mine, med[1 + len(best):])))
        except Exception as e:  # noqa: BLE001
            print(f"diaglte {layer} m{m} failed: {type(e).__name__}: {str(e)[:300]}")
    del flush


def _diag19_stream():
    if _g19 is None:
        return
    for mb in (50, 28):
        w = torch.zeros(mb << 20, dtype=torch.uint8, device="cuda")
        o = torch.zeros(1, device="cuda")
        med = _graph_medians([lambda: o.zero_()] + [(lambda c=c: _g19.g19_stream(w, c[0], c[1], o)) for c in DIAG19_STREAM])
        print(f"diag19 bare TMA stream {mb} MB, 132 blocks:", " ".join(f"{c}:{(t - med[0]) * 1e3:.2f}" for c, t in zip(DIAG19_STREAM, med[1:])),
              f"(a 1-kernel graph, {med[0] * 1e3:.2f} us, subtracted)")
        del w


def _graph_medians(fns, reps=25):
    """Median time (ms) of each fn as a CUDA-graph replay after a 256 MB read flush (cold L2, as the judge)."""
    flush = torch.zeros(256 << 20, dtype=torch.uint8, device="cuda").view(torch.int64)
    sink = torch.zeros((), dtype=torch.int64, device="cuda")
    graphs = []
    for fn in fns:
        gr = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            with torch.cuda.graph(gr, stream=s):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        graphs.append(gr)
    torch.cuda.synchronize()
    ev = [[(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in fns] for _ in range(reps)]
    for r in range(reps):
        for i, gr in enumerate(graphs):
            torch.sum(flush, 0, out=sink)
            ev[r][i][0].record()
            gr.replay()
            ev[r][i][1].record()
    torch.cuda.synchronize()
    del flush
    return [sorted(ev[r][i][0].elapsed_time(ev[r][i][1]) for r in range(reps))[reps // 2] for i in range(len(fns))]


def _tune_router(st):
    """Per m in ROUTER_R3_TUNE: router3 configs vs the cuBLAS hi/lo path, timed on random x; keep the fastest."""
    picks = {}
    if _mod is None:
        return picks
    g = torch.Generator(device="cuda")
    g.manual_seed(1)
    for m in sorted(set(ROUTER_R3_TUNE) | set(ROUTER_BMM_TUNE)):
        x = torch.randn(m, st["router"].shape[1], generator=g, device="cuda").to(torch.bfloat16)
        ref = _router_cublas(x, st)
        out = torch.empty((m, 256), dtype=torch.float32, device="cuda")
        r3 = [c for mmax, c in ROUTER_R3 if m <= mmax][:1]  # the fixed config this m runs without a pick
        cands = [None] + r3 + ROUTER_R3_TUNE.get(m, []) + [("bmm", S) for S in ROUTER_BMM_TUNE.get(m, [])]
        ok = []
        for c in cands:
            if c is not None:
                o = _router_run(x, st, c)
                torch.cuda.synchronize()
                if not ((o - ref).norm() / ref.norm()).item() < 1e-5:
                    continue
            ok.append(c)
        cands = ok
        med = _graph_medians([(lambda c=c: _router_cublas(x, st) if c is None else _router_run(x, st, c)) for c in cands])
        best = min(range(len(cands)), key=med.__getitem__)
        picks[m] = cands[best]
        print(f"tune router m{m}: best {cands[best]} {med[best] * 1e3:.2f} us; "
              + " ".join(f"{'cublas' if c is None else c}:{t * 1e3:.2f}" for c, t in zip(cands, med)))
    return picks


def _router_run(x, st, c):
    m = x.shape[0]
    out = torch.empty((m, 256), dtype=torch.float32, device=x.device)
    if c[0] == "bmm":
        S = c[1]
        ws = torch.bmm(x.contiguous().view(m, S, -1).transpose(0, 1), st["router_hl_t"].view(S, -1, 512), out_dtype=torch.float32)
        _mod.pj_rsum(ws, out)
    else:
        _mod.pj_router3(st["router_hl2"], x.contiguous(), out, *c)
    return out


def _router_cublas(x, st):
    acc = torch.mm(x, st["router_hl_t"], out_dtype=torch.float32)
    return acc[:, :256] + acc[:, 256:]


def _router(x, st, m):
    if _mod is not None:
        if m in st["rpick"]:
            cfg = st["rpick"][m]
            return _router_cublas(x, st) if cfg is None else _router_run(x, st, cfg)
        for mmax, cfg in ROUTER_F32:
            if m <= mmax:
                out = torch.empty((m, 256), dtype=torch.float32, device=x.device)
                _mod.pj_router2(st["router"], x, out, *cfg)
                return out
        for mmax, cfg in ROUTER_R3:
            if m <= mmax:
                out = torch.empty((m, 256), dtype=torch.float32, device=x.device)
                _mod.pj_router3(st["router_hl2"], x.contiguous(), out, *cfg)
                return out
        for mmax, cfg in ROUTER_HL:
            if m <= mmax:
                out = torch.empty((m, 256), dtype=torch.float32, device=x.device)
                _mod.g7_gemv(st["router_hl_p"], x, out, *cfg)
                return out
    if m <= ROUTER_FP32_MAX_TOKENS:
        return F.linear(x.float(), st["router"])
    return _router_cublas(x, st)


def kernel_fn(data, st):
    x, cfg = data["x"], data["config"]
    layer = cfg["layer"]
    if layer == "router":
        return _router(x, st, cfg["m"])
    m, n, k = cfg["m"], cfg["n"], cfg["k"]
    c = st["pick"].get((layer, m))
    if c is None:
        return F.linear(x, st[layer][:n])
    return _run(st, layer, x, torch.empty((m, n), dtype=torch.bfloat16, device=x.device), c)
