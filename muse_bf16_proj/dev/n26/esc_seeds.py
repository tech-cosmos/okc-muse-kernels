"""n26: per weight seed, the 12-bit compressor's per-row escape max for each BF16 2D layer weight (fallback if > 8)."""
import sys, torch
sys.path.insert(0, sys.argv[1])
import reference
seeds = [int(s) for s in sys.argv[2].split(",")]
for sd in seeds:
    w = reference.generate_weights(sd)
    out = []
    for k, t in sorted(w.items()):
        if not (torch.is_tensor(t) and t.dtype == torch.bfloat16 and t.dim() == 2 and min(t.shape) >= 512):
            continue
        b = t.contiguous().view(torch.int16).int() & 0xFFFF
        e = (b >> 7) & 0xFF
        emax = int(e.max()); base = emax - 15
        esc = (e - base) < 0
        mx = int(esc.sum(1).max())
        out.append(f"{k}{tuple(t.shape)} emax {emax} mean {esc.sum(1).float().mean():.2f} max {mx}{' FALLBACK' if mx > 8 else ''}")
    print(f"seed {sd}: " + " | ".join(out), flush=True)
    del w; torch.cuda.empty_cache()
