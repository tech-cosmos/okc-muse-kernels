"""n7s4: fold nsys GPU-metrics DRAM samples (tools/hbm_run.py under nsys, 200 kHz) onto one replay of the layer graph.
python tools/hbm_fold.py <report.sqlite> [reps] [peak_GBps=3352] [bin_us=0.5]
Each sample = mean DRAM read+write throughput % over its ~5 us period ending at its timestamp; samples from all replays
are aligned on the replay's first kernel start and a smooth per-bin profile is fitted by regularized least squares.
Prints the profile (GB/s per bin), per-kernel median start/end, and HBM-busy summaries."""
import sqlite3, sys, numpy as np
db = sys.argv[1]; reps = int(sys.argv[2]) if len(sys.argv) > 2 else 300
PEAK = float(sys.argv[3]) if len(sys.argv) > 3 else 3352.0; B = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
c = sqlite3.connect(db)
K = c.execute("select k.start, k.end, s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id = k.demangledName"
              " order by k.start").fetchall()
red = [i for i, k in enumerate(K) if k[2].startswith("void at::native::reduce_kernel")][-reps:]
runs = []
for j, i in enumerate(red):
    nxt = red[j + 1] if j + 1 < len(red) else len(K)
    ks = K[i + 2:nxt]  # i+1 = the sleep kernel
    assert "sleep" in K[i + 1][2] or "spin" in K[i + 1][2], K[i + 1][2]
    if ks:
        runs.append(ks)
import os
OFF = float(os.environ.get("OFF_US", "0")) * 1e3  # sample timestamp shift (calibrated on the flush end, ALIGN=flush)
if os.environ.get("ALIGN") == "fstart":
    t0s = np.array([K[i][0] for i in red], dtype=np.float64); runs = [[(K[i][0], K[i][1], "flush")] for i in red]
elif os.environ.get("ALIGN") == "flush":
    t0s = np.array([K[i][1] for i in red], dtype=np.float64); runs = [[(K[i][1], K[i + 1][1], "spin")] for i in red]
else:
    t0s = np.array([r[0][0] for r in runs], dtype=np.float64)
T = np.median([max(k[1] for k in r) - r[0][0] for r in runs]) / 1e3
lo, hi = float(os.environ.get("LO", "-10")), T + 10.0
nb = int((hi - lo) / B)
S = {m: np.array(c.execute(f"select timestamp, value from GPU_METRICS where metricId = {m} order by timestamp").fetchall(),
                 dtype=np.float64) for m in (18, 19)}
ts = S[18][:, 0] + OFF; v = S[18][:, 1] + S[19][:, 1]
P = np.median(np.diff(ts))
rows, ys = [], []
for t0 in t0s:
    a = np.searchsorted(ts, t0 + lo * 1e3); b = np.searchsorted(ts, t0 + hi * 1e3 + P)
    for s in range(a, b):
        e = (ts[s] - t0) / 1e3; st = e - P / 1e3
        if st < lo or e > hi:
            continue
        row = np.zeros(nb)
        for q in range(nb):
            x0, x1 = lo + q * B, lo + (q + 1) * B
            ov = max(0.0, min(x1, e) - max(x0, st))
            row[q] = ov / (P / 1e3)
        rows.append(row); ys.append(v[s])
A = np.array(rows); y = np.array(ys) * PEAK / 100.0
D = np.diff(np.eye(nb), 2, axis=0) * 3.0  # smoothness
x = np.linalg.lstsq(np.vstack([A, D]), np.concatenate([y, np.zeros(nb - 2)]), rcond=None)[0]
print(f"replays {len(runs)}, samples {len(y)}, period {P/1e3:.2f} us, median replay {T:.1f} us, fit rms {np.sqrt(np.mean((A @ x - y)**2)):.0f} GB/s")
# kernel medians
names = {}
for r in runs:
    for k in r:
        nm = k[2].split("(")[0][:48] + k[2].split(">(")[0][-12:] if "<" in k[2] else k[2][:48]
        names.setdefault(nm, []).append(((k[0] - r[0][0]) / 1e3, (k[1] - r[0][0]) / 1e3))
print("kernels (median start/end us):")
for nm, l in sorted(names.items(), key=lambda kv: np.median([p[0] for p in kv[1]])):
    a = np.array(l); n = len(l) // len(runs)
    print(f"  {np.median(a[:,0]):7.1f} {np.median(a[:,1]):7.1f}  x{n}  {nm}")
print("profile (us: GB/s), 1-us bins:")
w = int(round(1 / B))
line = []
for q in range(0, nb - w + 1, w):
    line.append(f"{lo + q * B:6.1f}:{x[q:q + w].mean():5.0f}")
for i in range(0, len(line), 8):
    print("  " + " ".join(line[i:i + 8]))
tt = lo + (np.arange(nb) + 0.5) * B
inr = (tt >= 0) & (tt <= T)
tot = x[inr].sum() * B * 1e-3  # MB (GB/s * us = KB) -> x*B us * GB/s = 1e3 B... : GB/s*us = 1e-6 GB = kB
print(f"bytes moved in [0, T]: {x[inr].sum() * B / 1e3:.1f} MB; mean BW {x[inr].mean():.0f} GB/s = {x[inr].mean()/PEAK*100:.0f}% of peak")
for thr in (0.5, 0.7, 0.8):
    busy = (x[inr] >= thr * PEAK).sum() * B
    print(f"  time at >= {int(thr*100)}% of peak: {busy:.1f} us of {T:.1f}")
