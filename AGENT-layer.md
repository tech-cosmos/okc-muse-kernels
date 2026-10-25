# Agent "layer": muse_decode_layer

Your problem is `muse_decode_layer` (dir `muse_decode_layer/`, notes NOTES-layer.md, STATE-layer.md).
It is 51% of the decode step on the overall board: the most valuable lane.

Plan, in order:
1. **Get on the board today.** Turn SGLang's own layer (baseline.py's modules, all importable in the
   judge image) into submission.py's `prepare`/`kernel_fn`, check it once, and submit it as the
   first correct run (~1.0x). If imports of sglang fail in the judge, fall back to the PyTorch
   starter made correct.
2. Read the first check's per-shape times and the profile (`profiles/submission.<job>.txt`): write
   the time split (qkv GEMM, attention, o GEMM, router, pooler, experts, unpooler, norms) per shape
   into STATE-layer.md. Attack the biggest slices.
3. Swap in faster pieces: our attention kernels from `~/attn/submission.py` (2.54x on the attention
   board; the same shapes at ctx 80K, but check the layer's exact inputs), proj's GEMMs and moe's
   experts as they post them under "Handoffs" in ~/COORD.md, and fuse the norms/residuals/routing
   between them (fewer launches; PDL; side streams where SGLang serializes).
4. Fusions across op boundaries are your unique lever (e.g. QK-norm+RoPE+cache store into one
   kernel, norm+gate into the o-projection's prologue, router+top-8 in one kernel).
