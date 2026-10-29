"""n8s1: read nv::g_st stamps of the v16-sliding nvs kernel: python tools/n8/stamp_rd.py <variant> [reps]"""
import sys, fcntl, statistics, torch
sys.path.insert(0, ".")
import bench
sub = bench.load_sub(sys.argv[1]); reps = int(sys.argv[2]) if len(sys.argv) > 2 else 20
w, st = bench.weights_state(sub)
c = next(c for l, c in bench.SCORED if f"{l} {c['layer']}" == "verify 16 sliding")
data = bench.make_input(c, w, 99)
g = bench.capture(lambda: sub.kernel_fn(data, st))
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda"); sink = torch.zeros((), dtype=torch.int64, device="cuda")
buf = torch.zeros(512 * 16, dtype=torch.int64, device="cuda")
NC = 128
rows = {k: [] for k in range(13)}; last = []
with open("/tmp/okc-gpu.lock", "w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    for _ in range(reps):
        torch.sum(flush.view(torch.int64), 0, out=sink); g.replay(); torch.cuda.synchronize()
        sub._NVS.nvs_dbg(buf); torch.cuda.synchronize()
        t = buf.view(512, 16)[:NC].cpu()
        t0 = int(t[:, 0].min())
        for k in range(13):
            v = [(int(x) - t0) / 1000 for x in t[:, k] if int(x) > t0 - 10**6]
            if v: rows[k].append((statistics.median(v), min(v), max(v)))
        last.append(max((int(x) - t0) / 1000 for x in t[:, 11]))
names = {0: "entry", 1: "pt staged", 10: "cons0 Q done", 12: "cons0 tile0 seen", 11: "cons0 end"}
for k in [0, 1] + list(range(2, 10)) + [10, 12, 11]:
    if rows[k]:
        r = rows[k]
        print(f"{names.get(k, f'tile {k-2} ready'):18s} med {statistics.median(x[0] for x in r):6.2f}  min {statistics.median(x[1] for x in r):6.2f}  max {statistics.median(x[2] for x in r):6.2f}")
print("last CTA end", statistics.median(last))
