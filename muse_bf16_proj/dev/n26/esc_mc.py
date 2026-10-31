"""n26: Monte Carlo of the 12-bit compressor's fallback (any row > 8 escapes) for reference-style weights randn(n,k)/sqrt(k) -> BF16."""
import math, sys, torch
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
g = torch.Generator(device="cuda").manual_seed(12345)
for name, (n, k) in {"pooler": (4096, 8192), "qkv_gate": (2304, 8192), "unpooler": (8192, 4096), "o": (8192, 1024)}.items():
    fb = fb2 = 0; hist = {}; maxes = []
    for i in range(N):
        t = (torch.randn((n, k), generator=g, device="cuda") / math.sqrt(k)).bfloat16()
        e = (t.view(torch.int16).int() >> 7) & 0xFF
        emax = int(e.max()); mx = int((e < emax - 15).sum(1).max())
        c16 = e - (emax - 16); mx2 = min(mx, int(((c16 < 0) | (c16 > 15)).sum(1).max()))  # n26 rule: best of two bases
        hist[emax] = hist.get(emax, 0) + 1; maxes.append(mx); fb += mx > 8; fb2 += mx2 > 8
    print(f"{name} {n}x{k}: fallback main {fb}/{N} n26 {fb2}/{N}, emax hist {hist}, max-escapes hist {dict(sorted({m: maxes.count(m) for m in set(maxes)}.items()))}", flush=True)
