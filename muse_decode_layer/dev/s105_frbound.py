"""s105: H100 bound for a one-launch pooler+router g30 at T=128: g30 time vs N rows (extra rows = pooler copies,
12-bit; raw-BF16 lo rows approximated by 1.33x as many 12-bit rows) vs today's FR cuBLAS torch.mm [pooler; hi; lo]."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F
import bench, reference
L = bench.load_sub("submission.py")
L._g30_build()
lib = L._G30["lib"]
W = reference.generate_weights(0)
p = W["pooler"]
M = 128
x = torch.randn(M, p.shape[1], device="cuda").bfloat16()
e0 = bench.empty_cost()
fr = torch.cat([p, p[:512]])  # FR today: pooler + router hi/lo as BF16, FP32 out
tfr = bench.time_graph(lambda: torch.mm(x, fr.t(), out_dtype=torch.float32)) - e0
tp = bench.time_graph(lambda: F.linear(x, p)) - e0
print(f"cuBLAS FR [{fr.shape[0]}] fp32 out {tfr:.2f} us | cuBLAS pooler only {tp:.2f} us")
for extra in (0, 256, 512, 640, 768):
    w = torch.cat([p, p[:extra]]) if extra else p
    code, sm, hdr, scale = L._compress(w)
    wp = L._pack30(code, sm)
    out = torch.empty(M, w.shape[0], device="cuda", dtype=torch.bfloat16)
    t = bench.time_graph(lambda: lib.g30_gemm(x, wp, hdr, out, scale, 0)) - e0
    print(f"g30 N={w.shape[0]} (+{extra}) {t:.2f} us")
