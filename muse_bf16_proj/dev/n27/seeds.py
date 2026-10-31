"""n27: per weight seed, which 12-bit base the n26 rule picks per proj layer, and old-rule fallbacks (rows > 8 escapes)."""
import sys, torch
sys.path.insert(0, ".")
import reference
for seed in [int(a) for a in sys.argv[1:]] or range(40):
    w = reference.generate_weights(seed); out = []
    for name in ["qkv_gate", "o", "pooler", "unpooler"]:
        e = (w[name].view(torch.int16).int() >> 7) & 0xFF; em = int(e.max())
        r15 = int((e < em - 15).sum(1).max()); c = e - (em - 16); r16 = int(((c < 0) | (c > 15)).sum(1).max())
        t15 = int((e < em - 15).sum()); t16 = int(((c < 0) | (c > 15)).sum())
        pick = 16 if (r16 <= 8 and (t16 < t15 or r15 > 8)) else 15
        out.append(f"{name}:em{em} r{r15}/{r16} {'B16' if pick == 16 else ''}{' OLDFB' if r15 > 8 else ''}")
    print(seed, " | ".join(out), flush=True)
