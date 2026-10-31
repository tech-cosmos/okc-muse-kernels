"""fresh x8 (b): in-layer tk2f per-CTA phases. python tools/fresh/span/run.py [sub_st.py] [filter e.g. 'decode 1 sliding'] [seeds]
Prints medians over 15 cold-L2 replays of: per-stamp min/median/max over CTAs (us from the first tk2f CTA entry)."""
import sys, fcntl, torch
sys.path.insert(0, ".")
import bench, reference as R
path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fx8/sub_st.py"
flt = sys.argv[2] if len(sys.argv) > 2 else "decode 1 sliding"
seeds = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "7,11,13,17,99").split(",")]
sub = bench.load_sub(path)
W, st = bench.weights_state(sub)
lib = sub._moe__lib()
glib = sub._g10_lib()
op = torch.zeros(4096, dtype=torch.int64)
names = ["entry", "routed", "prodTMA", "pastwait", "ph1data", "ph1end", "cntok", "ph2data", "end", "rmsdone"]
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
o = torch.zeros(1024 * 16, dtype=torch.int64)
for load, c in bench.SCORED:
    if flt not in f"{load} {c['layer']}":
        continue
    for seed in seeds:
        data = bench.make_input(c, W, seed)
        ids = None
        out = sub.kernel_fn(data, st); torch.cuda.synchronize()
        ids = out[2].long(); ls = int(data["local_start"])
        E = int(torch.unique(ids[(ids - ls) % 256 < 32]).numel())
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            sub.kernel_fn(data, st)
        g.replay(); torch.cuda.synchronize()
        acc, tot, pacc = [], [], []
        with open("/tmp/okc-gpu.lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            for _ in range(15):
                o.zero_(); lib.moe_tdbg(o)  # (reads; stamps overwritten each replay)
                flush.add_(1)
                a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                a.record(); g.replay(); b.record(); torch.cuda.synchronize()
                tot.append(a.elapsed_time(b) * 1e3)
                lib.moe_tdbg(o); glib.pr1_tdbg(op)
                t = o.view(1024, 16)[:, :10].double()
                live = t[:, 0] > 0
                t = t[live]
                t0 = t[:, 0].min()
                acc.append((t - t0) / 1000)
                try:
                    pt = op[:4000].view(1000, 4).double()
                    pt = pt[(pt[:, 0] > t0 - 200e3) & (pt[:, 0] < t0 + 50e3)]
                    rt, pl = pt[pt[:, 2] > 0], pt[pt[:, 2] == 0]
                    pacc.append(torch.tensor([pt[:, 0].min(), rt[:, 1].median(), rt[:, 1].max(), float(op[4000]),
                                              pl[:, 1].min(), pl[:, 1].median(), pl[:, 1].max(), float(op[4001]), float(op[4003]), float(op[4004]), rt[:, 3].max(), len(rt), len(pl)]).double() - torch.tensor([t0] * 11 + [0, 0]).double())
                except Exception:
                    pass
            fcntl.flock(lk, fcntl.LOCK_UN)
        n = min(x.shape[0] for x in acc)
        Tm = torch.stack([x[:n] for x in acc]).median(0).values
        print(f"{load} {c['layer']} s{seed} E{E} CTAs {n} graph {sorted(tot)[7]:.1f}: " + "  ".join(
            f"{nm} {Tm[:, k].min():.2f}/{Tm[:, k].median():.2f}/{Tm[:, k].max():.2f}" for k, nm in enumerate(names)), flush=True)
        if not pacc: continue
        P = torch.stack(pacc).median(0).values
        P[:11] /= 1000
        print(f"   pr1 (us vs tk2f first entry): entry {P[0]:.2f} | router blocks ({int(P[11])}) end med {P[1]:.2f} max {P[2]:.2f} | routing written {P[3]:.2f} | pooler blocks ({int(P[12])}) end min {P[4]:.2f} med {P[5]:.2f} max {P[6]:.2f} | routing: last pre-ticket {P[10]:.2f} post-ticket {P[7]:.2f} scores loaded {P[8]:.2f} top8 {P[9]:.2f}", flush=True)
