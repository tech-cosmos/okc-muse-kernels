import sys, fcntl, torch
sys.path.insert(0, ".")
import bench
path, flt = sys.argv[1], sys.argv[2]; seeds = [int(x) for x in sys.argv[3].split(",")]
sub = bench.load_sub(path); W, st = bench.weights_state(sub)
lib = sub._moe__lib(); glib = sub._g10_lib()
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
o = torch.zeros(1024 * 16, dtype=torch.int64); op = torch.zeros(4096, dtype=torch.int64)
for load, c in bench.SCORED:
    if flt not in f"{load} {c['layer']}":
        continue
    for seed in seeds:
        data = bench.make_input(c, W, seed)
        sub.kernel_fn(data, st); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            sub.kernel_fn(data, st)
        g.replay(); torch.cuda.synchronize()
        rows, tot, acc = [], [], []
        with open("/tmp/okc-gpu.lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            for _ in range(15):
                flush.add_(1)
                a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                a.record(); g.replay(); b.record(); torch.cuda.synchronize()
                tot.append(a.elapsed_time(b) * 1e3)
                lib.moe_tdbg(o); glib.pr1_tdbg(op)
                t = o.view(1024, 16)[:, :10].double(); t = t[t[:, 0] > 0]; t0 = t[:, 0].min()
                f = lambda k: (t[:, k] - t0) / 1000
                fin = [(float(op[j]) - t0) / 1000 for j in (4020, 4021, 4022, 4023)] + [(float(op[4030:4038].max()) - t0) / 1000]
                acc.append((t - t0) / 1000)
                rows.append(fin + [f(0).median(), f(0).max(), f(1).median(), f(1).max(), f(4).max(), f(8).max()])
            fcntl.flock(lk, fcntl.LOCK_UN)
        n = min(x.shape[0] for x in acc); Tm = torch.stack([x[:n] for x in acc]).median(0).values
        names = ["entry", "routed", "prodTMA", "pastwait", "ph1data", "ph1end", "cntok", "ph2data", "end"]
        print("   tk2f CTAs", n, "  ".join(f"{nm} {Tm[:, k].min():.2f}/{Tm[:, k].median():.2f}/{Tm[:, k].max():.2f}" for k, nm in enumerate(names)))
        R = torch.tensor(rows).median(0).values.tolist()
        print(f"{load} {c['layer']} s{seed} graph {sorted(tot)[7]:.1f} | pr8 finish (us vs tk2f first entry): entry {R[0]:.2f} pastwait {R[1]:.2f} items done {R[2]:.2f} summed {R[3]:.2f} routed(8 warps max) {R[4]:.2f} | tk2f entry med/max {R[5]:.2f}/{R[6]:.2f} routed med/max {R[7]:.2f}/{R[8]:.2f} ph1data max {R[9]:.2f} end {R[10]:.2f}", flush=True)
