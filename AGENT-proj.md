# Agent "proj": muse_bf16_proj

Your problem is `muse_bf16_proj` (dir `muse_bf16_proj/`, notes NOTES-proj.md, STATE-proj.md).
It is 11% of the decode step on the overall board, and the same GEMMs (qkv_gate, o, pooler,
unpooler, router) are inside muse_decode_layer: post each faster kernel under "Handoffs" in
~/COORD.md for the layer agent (clean interface: x, prepared weight -> out).

Plan, in order:
1. **Get on the board today.** baseline.py's `_bf16_gemm_dispatch_impl` (cuBLAS) and the router's
   hi/lo split are importable in the judge, or plain torch matmul: check once, submit as the first
   correct run (~1.0x). Mind the router: FP32 output, 0.01% error (FP32 or the BF16 hi+lo split).
2. Per shape, compare the judge's time to the floor in TASK.md. m = 1..32 are pure weight-bandwidth
   GEMVs: a split-K/streaming kernel that keeps all SMs busy and reads the weight once at ~4 TB/s
   should beat cuBLAS clearly; m = 128/256 are small GEMMs (wgmma/TMA, swap-AB tricks).
3. Weights may be repacked in prepare() (layout for coalesced/TMA streaming), not requantized.
4. When proj is near the floor on all shapes, ask the lead (under "Requests" in ~/COORD.md) for a
   new lane; likely helping the layer agent with fused projection epilogues.

**Since 2026-10-26: your target is the layer's GEMMs (qkv/o/pooler/unpooler/router) as they run inside
muse_decode_layer, including fused epilogues. Your own board is a test bed only. See ~/COORD.md.**
