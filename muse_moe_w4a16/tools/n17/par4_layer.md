# n17 TK2G_PAR4 -> layer moe_src.py (2 hunks + 1 define; device code only, no python change)
Why: tk2h's weighted split (TK2H_M W > 0 = the layer's "tk2h 4" at T=32) computes 4 row bounds with tk2g_bound on
warp 0, serially: a latency chain (smem loads, shuffles, 64-bit division) that held EVERY warp (producer too) at the
following __syncthreads: H100 timeline (tools/n17/mktl_d.py) consumer start 4.4 us vs 2.3 for gs8 / W 0.
Fix: warp w < 4 computes bound w (b + (w & 1), rps (w & 2) ? 128 : 256); p in 32-bit (W * b < 2^28). Same bounds:
bitwise 138/138 (tools/n17/par4_t.py, W 2/3/4, layer d16/d32 routings + random T 9..32). H100 W3 consumer start
-> 2.6 us; moe-only totals d16/d32 -0.4..-1.5 us.
Port: apply the 3 hunks of tools/n17/par4.diff to moe_src.py (anchors: "#ifndef TK2G_PAR", the
"const int p = (int)((long long)W * b / gridDim.x);" line INSIDE tk2g_bound (moe_src.py ~3971; NOT the tk2w_bound one
at ~3940), and the "if (warp == 0) { const int v0 = tk2g_bound(" block).
