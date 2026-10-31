"""n2u per-CTA stamps vs tk2f last-CTA end. python /tmp/fx9/run_n2u.py <sub> <filter> <seeds>"""
import sys, fcntl, torch
sys.path.insert(0, ".")
import bench
path, flt = sys.argv[1], sys.argv[2]
seeds = [int(x) for x in sys.argv[3].split(",")]
sub = bench.load_sub(path)
W, st = bench.weights_state(sub)
lib = sub._moe__lib(); glib = sub._g10_lib()
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
o = torch.zeros(1024 * 16, dtype=torch.int64); nt = torch.zeros(16384, dtype=torch.int64)
names = ["entry", "pastwait", "ids", "Yss", "sync1", "unpool", "end"]
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
        acc, tot, tk = [], [], []
        with open("/tmp/okc-gpu.lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            for _ in range(15):
                nt.zero_(); glib.n2u_tdbg(nt)
                o.zero_(); lib.moe_tdbg(o)
                flush.add_(1)
                a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                a.record(); g.replay(); b.record(); torch.cuda.synchronize()
                tot.append(a.elapsed_time(b) * 1e3)
                lib.moe_tdbg(o); glib.n2u_tdbg(nt)
                t = o.view(1024, 16)[:, :10].double(); t = t[t[:, 0] > 0]
                tend = t[:, 8].max(); tk.append(((t[:, 8].median() - tend) / 1000).item())
                n = nt.view(2048, 8)[:, :7].double(); n = n[n[:, 0] > 0]
                acc.append((n - tend) / 1000)
            fcntl.flock(lk, fcntl.LOCK_UN)
        m = min(x.shape[0] for x in acc)
        Tm = torch.stack([x[:m] for x in acc]).median(0).values
        print(f"{load} {c['layer']} s{seed} n2u CTAs {m} graph {sorted(tot)[7]:.1f} (us vs tk2f last CTA end; tk2f median CTA end {sorted(tk)[7]:.2f}): " + "  ".join(
            f"{nm} {Tm[:, k].min():.2f}/{Tm[:, k].median():.2f}/{Tm[:, k].max():.2f}" for k, nm in enumerate(names)), flush=True)
