# STATE-layer (muse_decode_layer) — short current picture; NOTES-layer.md is the long log

Updated 2026-10-25 ~19:50Z (session 1).

## Board
- Best scored: **53d9d188 = 224f7fd (SGLang baseline modules as submission) 1.000x, 22,609 us/step** (rank: first correct run).
- Board #1 colinlikescode 2.285x (9,900 us). Overall-board priority = get this lane to ~2x.

## SGLang per-shape H200 times (us, run 53d9d188) and floors
| shape | sglang | floor |
|---|---|---|
| decode 1 sliding / full | 172.72 / 183.10 | 40.0 / 44.2 |
| decode 16 sliding / full | 265.55 / 437.74 | 76.7 / 137.5 |
| decode 32 sliding / full | 261.33 / 612.56 | 84.5 / 208.1 |
| verify 1 sliding / full | 266.83 / 272.30 | 62.3 / 66.4 |
| verify 16 sliding / full | 506.48 / 692.02 | 126.6 / 188.7 |
| verify 32 sliding / full | 723.60 / 1087.28 | 143.8 / 267.2 |
Weights per load: sliding 46, full 16 (sliding dominates every load).

## SGLang time split (judge profile 53d9d188, top-12 kernels only)
- decode 1 sliding 175: Marlin x2 40.7, cuBLAS qkv+pooler (128x8 splitK) 32.0, FA3 9.9, o/unpool GEMMs ~12, router FP32 gemv 5.1,
  splitK reduces 5.3, rms_norm 5.7, ~24 small kernels (casts, store_cache, align, maps, norms) = ~60 us of glue.
- decode 16/32 full: FA3 191.6 / 368.7 (ours: 87.7 / 168.2 on the attention board), Marlin 148 / 124.
- verify 32 full: Marlin 537, FA3 379. verify 1 sliding: Marlin 74.6, GEMMs 33, router FP32 (sm80 xmma) 13+9.
- So: experts (Marlin) are the biggest slice at T>=16; attention on full layers; glue at small T.

## Code (muse_decode_layer/)
- layer.py = the layer source; attn_src.py = copy of ~/attn/submission.py @82866c6; `python build.py` -> submission.py
  (inlines attention, renames its kernel_fn attn_kernel_fn, patches triton.next_power_of_2 in a jit body: SGLang's
  import breaks it in the judge image, check 7e3e63a1 failed 2 tests on that).
- bench.py: `test` (13 judge tests vs reference.py), `bench`, `prof`; tools/micro*.py: glue-kernel microbenches.
  Locally no SGLang: experts use a dequant-everything fallback (correctness only; timings of whole layer meaningless).
- main 2a21d76 = v2: cuBLAS GEMMs (F.linear), our attention, own Triton boundary x2, router (FP32 FMA split-K <= 8
  tokens, BF16 hi/lo tl.dot above), routing (top-8, local ids), pre-expert norm; SGLang Marlin experts (judge only).

## Queued / next
1. Read check of 2a21d76 (v2). If it beats 1.000x clearly, submit.
2. Glue H100 T=1 (us): boundary 5.8 (single CTA per row: make multi-CTA), router 5.9 + routing 5.5 (fuse via ticket),
   prenorm 3.7 (fold into experts prologue later). One tiny kernel in a graph = 2.4 us on H100.
3. Watch COORD Handoffs: moe's experts (replace Marlin + align + silu + wsum), proj's GEMMs (qkv/o/pooler/unpooler).
4. Bigger fusion: o-GEMM epilogue emits per-row partial sums (sum a^2, (beta a)^2, alpha s beta a, (alpha s)^2) so
   boundary 1 folds into the pooler/router prologue; same for boundary 2 via unpooler epilogue.

## Dead ends
- (none yet)
