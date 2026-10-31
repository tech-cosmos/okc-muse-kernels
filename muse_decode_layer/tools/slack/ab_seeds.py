"""moe/slack: paired A/B of submission files over several input seeds per scored shape (one process, interleaved).
python tools/slack/ab_seeds.py <filter> <rounds> <seeds a,b,c> <file A> <file B> ...
Per shape and seed: capture each file, time (bench.time_graph: cold L2, median of 40 replays) per round, files
interleaved; prints per shape the mean over seeds of the per-seed medians, and B - A."""
import sys, statistics, importlib.util, torch
sys.path.insert(0, ".")
import bench, reference
flt = sys.argv[1]; rounds = int(sys.argv[2]); seeds = [int(s) for s in sys.argv[3].split(",")]; files = sys.argv[4:]
w = reference.generate_weights(reference.WEIGHTS_SEED)
subs = []
for i, f in enumerate(files):
    spec = importlib.util.spec_from_file_location(f"sub{i}", f)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    subs.append((m, m.prepare(w)))
e0 = bench.empty_cost()
for l, c in bench.SCORED:
    n = f"{l} {c['layer']}"
    if flt not in n:
        continue
    per = [[] for _ in files]
    for sd in seeds:
        d = bench.make_input(c, w, sd)
        res = [[] for _ in files]
        for r in range(rounds):
            for i, (m, st) in enumerate(subs):
                res[i].append(bench.time_graph(lambda: m.kernel_fn(d, st)) - e0)
        for i in range(len(files)):
            per[i].append(statistics.median(res[i]))
        del d
        getattr(reference.generate_input, "cache_clear", lambda: None)()
        torch.cuda.empty_cache()
    mean = [sum(p) / len(p) for p in per]
    print(f"{n:18s}", "  ".join(f"[{i}] {mean[i]:7.2f}" for i in range(len(files))),
          " | B-A per seed", " ".join(f"{b - a:+6.2f}" for a, b in zip(per[0], per[1])) if len(files) > 1 else "",
          f" mean {mean[1] - mean[0]:+6.2f}" if len(files) > 1 else "", flush=True)
