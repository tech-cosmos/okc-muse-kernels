"""fresh x12: run a stamp.py copy on a scored shape (graph replay, cold L2), print per-CTA stamp medians.
python tools/fresh/cw/strun.py <stamped submission.py> [shape filter, default 'decode 1 sliding'] [reps]"""
import sys, fcntl, statistics, torch
sys.path.insert(0, ".")
import bench
sub = bench.load_sub(sys.argv[1])
flt = sys.argv[2] if len(sys.argv) > 2 else "decode 1 sliding"
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 20
w, s = bench.weights_state(sub)
(name, c), = [(f"{l} {c['layer']}", c) for l, c in bench.SCORED if f"{l} {c['layer']}" == flt]
d = bench.make_input(c, w, 99)
sub.kernel_fn(d, s); torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    sub.kernel_fn(d, s)
cnt = s["moe"]["cnt"]
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
rows = []
with open("/tmp/okc-gpu.lock", "w") as lk:
    fcntl.flock(lk, fcntl.LOCK_EX)
    for r in range(reps):
        flush.zero_(); cnt[256:].zero_(); torch.cuda.synchronize()
        g.replay(); torch.cuda.synchronize()
        cc = cnt[256 + 132 * 32:].view(torch.int64).cpu() if cnt.numel() > 256 + 132 * 32 else None
        rows.append(cnt[256:256 + 132 * 32].view(torch.int64).view(132, 16)[:, :13].cpu().clone())
K = {0: "entry", 1: "g13 start", 5: "g2 entry", 6: "wait ok", 7: "ch3 ok", 8: "end", 9: "helper last", 12: "h first arr", 10: "h last arr", 11: "h last fenced"}
for k, nm in K.items():
    meds = []
    for t in rows:
        t0 = t[:, 0].min().item()
        v = t[:, k]
        v = v[v > 0]
        if len(v) == 0: break
        v = (v - t0).double() / 1000.0
        meds.append((v.min().item(), v.median().item(), v.max().item()))
    if meds:
        print(f"{nm:12s} min {statistics.median(m[0] for m in meds):6.2f}  med {statistics.median(m[1] for m in meds):6.2f}  max {statistics.median(m[2] for m in meds):6.2f} us")
t = rows[-1]; t0 = t[:, 0].min().item()
if cc is not None:
    print("chunk complete (us), seg x chunk:", " | ".join(" ".join(f"{(cc[j * 4 + q].item() - t0) / 1000:5.1f}" if cc[j * 4 + q] > 0 else "  -  " for q in range(4)) for j in range(8)))
    sa = s["moe"]["cnt"]
print("last rep per CTA (us): entry / g13 / g2 / waitok / ch3 / end, first 12 and slowest-end 6")
order = sorted(range(132), key=lambda b: -t[b, 8].item())
for b in list(range(12)) + order[:6]:
    print(b, " ".join(f"{(t[b, k].item() - t0) / 1000:6.2f}" if t[b, k] > 0 else "   -  " for k in (0, 1, 5, 6, 7, 8)))
