"""s49: exponent statistics of the judge's weights (randn/sqrt(k) -> BF16): entropy bound for 'fewer bytes' at m=1,
and the escape rate of fixed-rate exponent windows (current 12-bit = 16-value window, ~1e-5 escapes)."""
import math, torch
k, n = 8192, 4096
w = (torch.randn(n, k, device="cuda") / math.sqrt(k)).bfloat16()
e = (w.view(torch.int16).int() >> 7) & 0xFF
mx = int(e.max())
h = torch.bincount((mx - e).flatten().clamp(max=40)).float() / e.numel()
p = h[h > 0]
print("exponent offset below max: mass", [round(x, 5) for x in h[:20].tolist()])
print("exponent entropy %.3f bits -> %.2f bits/weight (8 sign+mantissa bits are uniform)" % (
    float(-(p * p.log2()).sum()), 8 + float(-(p * p.log2()).sum())))
for W in (4, 6, 8, 16):
    print("window %2d codes: escape fraction %.2e = %.1f per %d-row" % (W, float(h[W:].sum()), float(h[W:].sum()) * k, k))
