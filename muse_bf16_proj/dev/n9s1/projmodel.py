# n9s1: what per-shape times give the proj board's #1 (1.650)? Model "near-floor" team: per shape
# t = min(ours, fixed + MB_eff/rate) with MB_eff = 12-bit bytes (router FP32), and at m>=128 max(that, FLOP/eff_tf + fixed).
import json, math, sys
d = json.load(open('/home/daytona/r-proj/muse_bf16_proj/profiles/submission.eb166fc0a787491d.json'))
NK = dict(qkv_gate=2304 * 8192, o=8192 * 1024, pooler=4096 * 8192, unpooler=8192 * 4096, router=256 * 8192)
def model(layer, m, fixed, rate, tf):
    nk = NK[layer] if not (layer == 'unpooler' and m <= 8) else 1024 * 4096
    mb = nk * (4 if layer == 'router' else 1.5) / 1e6
    fl = 2 * m * nk * (2 if layer == 'router' else 1)
    return fixed + max(mb / rate, fl / tf / 1e6)
for fixed, rate, tf in [(2.5, 4.3, 750), (3.0, 4.0, 700), (3.5, 3.8, 650), (2.0, 4.3, 800)]:
    loads = {}
    for b in d['benchmarks']:
        c = b['case']; o = b['median_ns'] / 1e3; s = b['baseline_ns'] / 1e3
        t = min(o, model(c['layer'], c['m'], fixed, rate, tf))
        L = loads.setdefault(c['load'], [0, 0, 0]); L[0] += s; L[1] += o; L[2] += t
    g = math.exp(sum(math.log(v[0] / v[2]) for v in loads.values()) / len(loads))
    print(f'fixed {fixed} rate {rate} tf {tf}: geomean {g:.3f} |', ' '.join(f'{k}:{v[0]/v[2]:.2f}' for k, v in loads.items()))
