"""Per-seed time vs local routing load (one process, one prepare = pinned tuner).
python tools/n24/seedspread.py <file> "<filter>" <seeds a-b> [rounds]
Prints per shape+seed: routes, distinct local experts, max tokens on one expert, median us over rounds."""
import sys, statistics
sys.path.insert(0, ".")
import torch, bench, reference
sub = bench.load_sub(sys.argv[1]); flt = sys.argv[2]
a, b = map(int, sys.argv[3].split("-")); rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 3
w, st = bench.weights_state(sub)
e0 = bench.empty_cost()
for n, c in [(f"{l} {c['layer']}", c) for l, c in bench.SCORED if flt in f"{l} {c['layer']}"]:
    rows = []
    for sd in range(a, b + 1):
        data = bench.make_input(c, w, sd)
        ls = int(data["local_start"])
        ids = reference.ref_front(data, w)["biased"].topk(reference.TOP_K, dim=-1).indices
        loc = ids[(ids - ls) % reference.N_EXPERTS < reference.LOCAL_EXPERTS]
        routes, distinct = reference._local_load(ids.long(), ls)
        mx = int(torch.bincount(loc.flatten()).max()) if loc.numel() else 0
        ts = [bench.time_graph(lambda: sub.kernel_fn(data, st)) - e0 for _ in range(rounds)]
        rows.append((sd, routes, distinct, mx, statistics.median(ts), min(ts)))
        del data; torch.cuda.empty_cache()
    for r in sorted(rows, key=lambda r: r[2]):
        print(f"{n:16s} s{r[0]:<3d} routes {r[1]:3d} distinct {r[2]:2d} max {r[3]:3d}  {r[4]:7.2f} ({r[5]:7.2f})", flush=True)
