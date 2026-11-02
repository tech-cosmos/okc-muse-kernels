"""fresh x42 (lead 2026-11-02T03:45Z, MEASUREMENT ONLY): routing slack ON vs OFF on v189 + moe/slack + layer/slack1.
python tools/fresh/x42/slack_ab.py <off.py> <on.py> <router seeds a,b,..> <rounds> <judge n> [time filter] [judge filter] [timed inputs per seed]
Per router seed: prepare OFF and ON (TUNE pinned in both files), time every scored shape (bench.time_graph: cold L2,
median of 40 replays) interleaved OFF/ON (ON first on odd rounds) with inputs 500+10rs+j (j < ni; per-seed value = mean over inputs of the round medians), then the slack_judge routing
stress (fresh inputs copied into a captured graph, L2 flushed, check_routing vs reference.ref_front) on the ON state for
the judge-filter shapes. Prints one line per seed/shape, then per shape the median over seeds of the per-seed medians,
and the score delta (46 sl + 16 full per load, geomean over the loads present)."""
import sys, math, statistics, torch
sys.path.insert(0, ".")
import bench, reference

off_p, on_p = sys.argv[1], sys.argv[2]
seeds = [int(s) for s in sys.argv[3].split(",")]
rounds, nj = int(sys.argv[4]), int(sys.argv[5])
tflt = sys.argv[6] if len(sys.argv) > 6 else ""
jflt = sys.argv[7].split("|") if len(sys.argv) > 7 and sys.argv[7] else ["decode 1 ", "verify 16", "verify 32"]
ni = int(sys.argv[8]) if len(sys.argv) > 8 else 1  # timed inputs per router seed and shape (mean of their medians)
subs = [bench.load_sub(off_p), bench.load_sub(on_p)]
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
sink = torch.zeros((), dtype=torch.int64, device="cuda")
e0 = bench.empty_cost()
med = {}  # name -> [[off per seed], [on per seed]]
jtot = [0, 0, 0.0, 0.0]  # inputs, fails, distinct saved sum, worst drop-lead


def margins(ids, biased, local_start):
    best = biased.topk(9, dim=-1)
    top8, ninth = best.values[:, :8], best.values[:, 8:9]
    gap = float((top8[:, 7:8] - biased.gather(1, ids.long())).max())
    kept = (best.indices[:, :8, None] == ids.long()[:, None, :]).any(-1)
    local = (best.indices[:, :8] - local_start) % reference.N_EXPERTS < reference.LOCAL_EXPERTS
    lead = (top8 - ninth).masked_fill(~(local & ~kept), -1.0)
    return gap, float(lead.max())


for rs in seeds:
    w = reference.generate_weights(reference.WEIGHTS_SEED, router_seed=rs)
    sts = [m.prepare(w) for m in subs]
    for load, c in bench.SCORED:
        name = f"{load} {c['layer']}"
        if tflt in name:
            m = [0.0, 0.0]
            for j in range(ni):
                d = bench.make_input(c, w, 500 + 10 * rs + j)
                res = [[], []]
                for r in range(rounds):
                    for i in ((0, 1) if r % 2 == 0 else (1, 0)):
                        res[i].append(bench.time_graph(lambda: subs[i].kernel_fn(d, sts[i])) - e0)
                for i in range(2):
                    m[i] += statistics.median(res[i]) / ni
                del d
            med.setdefault(name, [[], []])
            med[name][0].append(m[0]); med[name][1].append(m[1])
            print(f"rs{rs} {name.replace('decode ', 'd').replace('verify ', 'v')[:6]} {m[0]:7.2f} {m[1]:7.2f} {m[1] - m[0]:+6.2f}", flush=True)
        if nj > 0 and any((name + " ").startswith(f) for f in jflt):
            sub, st = subs[1], sts[1]
            data = bench.make_input(c, w, 7000)
            sub.kernel_fn(data, st)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                sub.kernel_fn(data, st)
                with torch.cuda.graph(g, stream=s):
                    out = sub.kernel_fn(data, st)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            fails, wg, wl, sv, msgs = 0, 0.0, -1.0, 0.0, []
            for i in range(nj):
                dd = bench.make_input(c, w, 9001 + 17 * i + 1000 * rs)
                for k, v in dd.items():
                    if isinstance(v, torch.Tensor):
                        data[k].copy_(v)
                torch.sum(flush.view(torch.int64), 0, out=sink)
                g.replay()
                torch.cuda.synchronize()
                front = reference.ref_front(dd, w)
                ls = front["local_start"]
                err = reference.check_routing(out[2], front["biased"], ls)
                if err:
                    fails += 1; msgs.append(f"input {i}: {err}")
                gp, ld = margins(out[2], front["biased"], ls)
                wg, wl = max(wg, gp), max(wl, ld)
                r0, d0 = reference._local_load(front["biased"].topk(8, dim=-1).indices, ls)
                r1, d1 = reference._local_load(out[2].long(), ls)
                sv += d0 - d1
                del dd, front
            jtot[0] += nj; jtot[1] += fails; jtot[2] += sv; jtot[3] = max(jtot[3], wl)
            print(f"rs{rs} {name.replace('decode ', 'd').replace('verify ', 'v')[:6]} JUDGE fails {fails}/{nj} gap {wg:.4f} drop-lead {wl:.4f} distinct saved {sv / nj:.2f}", flush=True)
            for mm in msgs[:3]:
                print("   ", mm, flush=True)
            del g, data, out
        getattr(reference.generate_input, "cache_clear", lambda: None)()
        torch.cuda.empty_cache()
    del sts, w
    torch.cuda.empty_cache()

print("\nshape              off(med)  on(med)   on-off  (median over router seeds of per-seed medians)")
for name, (a, b) in med.items():
    print(f"{name:18s} {statistics.median(a):8.2f} {statistics.median(b):8.2f} {statistics.median(b) - statistics.median(a):+7.2f}")
loads = sorted({n.rsplit(" ", 1)[0] for n in med if f"{n.rsplit(' ', 1)[0]} full" in med and f"{n.rsplit(' ', 1)[0]} sliding" in med})
if loads:
    lg = [math.log((46 * statistics.median(med[l + " sliding"][0]) + 16 * statistics.median(med[l + " full"][0]))
                   / (46 * statistics.median(med[l + " sliding"][1]) + 16 * statistics.median(med[l + " full"][1]))) for l in loads]
    for l, x in zip(loads, lg):
        print(f"  {l:10s} load gain {100 * x:+.2f}%")
    print(f"score delta (geomean over {len(loads)} loads): {100 * (math.exp(sum(lg) / len(lg)) - 1):+.2f}%  (6-load board: x{len(loads)}/6)")
print(f"JUDGE total: {jtot[1]} fails / {jtot[0]} slack-on inputs, distinct saved {jtot[2] / max(jtot[0], 1):.2f}/input, worst drop-lead {jtot[3]:.4f}")
