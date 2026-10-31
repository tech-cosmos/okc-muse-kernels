"""n26: escapes per row for base = emax-15 (main) vs emax-16/-17 (top-exponent outliers become escapes too), seed 5353 weights."""
import sys, torch
sys.path.insert(0, sys.argv[1])
import reference
w = reference.generate_weights(int(sys.argv[2]) if len(sys.argv) > 2 else reference.WEIGHTS_SEED)
for k in ("qkv_gate", "o", "pooler", "unpooler"):
    t = w[k]; e = (t.contiguous().view(torch.int16).int() >> 7) & 0xFF; emax = int(e.max())
    s = []
    for d in (15, 16, 17):
        c = e - (emax - d); esc = (c < 0) | (c > 15); r = esc.sum(1)
        s.append(f"base emax-{d}: top {int((c > 15).sum())} total {int(esc.sum())} mean {r.float().mean():.3f} max {int(r.max())}")
    print(k, tuple(t.shape), " | ".join(s))
