# n5s36: per-expert token counts on the layer's v16/v32 sliding routings -> twm pass structure, dq repeats, modeled consumer cycles
import torch, numpy as np, sys
r = torch.load(sys.argv[1])
# H200-proxy cycles per 8 KB weight stage (RT2, G88, s28 NOTES l.753), stream included
tab = {8: 593, 16: 633, 32: 695, 40: 720, 64: 821, 72: 895, 96: 997, 104: 1017, 128: 1143}
ks = sorted(tab); c = lambda n: float(np.interp(n, ks, [tab[k] for k in ks]))
for load, hot in (("verify 16", False), ("verify 32", True)):
    for seed in (99, 7, 11, 13, 17):
        loc = r[(load, seed)].flatten(); loc = loc[loc >= 0]
        cnt = torch.bincount(loc.long(), minlength=32).tolist(); cnt = sorted([n for n in cnt if n > 0], reverse=True)
        now = ideal = 0.0; extra_dq = 0
        for n in cnt:
            g = lambda m: 8 * -(-m // 8)
            if n <= 32: p = [g(n)]
            elif n <= 64: p = [g(n)]
            elif hot and n <= 128: p = [128]
            else: p = [64] * (n // 64) + ([g(n % 64)] if n % 64 else [])
            now += sum(c(x) for x in p); extra_dq += len(p) - 1
            ideal += c(g(n)) if n <= 128 else c(128) * n / 128
        print(f"{load} s{seed}: E={len(cnt)} max={cnt[0]} top5={cnt[:5]} >64:{sum(n>64 for n in cnt)} 33-64:{sum(32<n<=64 for n in cnt)} "
              f"extra dq passes={extra_dq}  cyc/8KB-stage summed now {now:.0f} ideal-dq-once {ideal:.0f} ({100*(ideal/now-1):+.1f}%)")
