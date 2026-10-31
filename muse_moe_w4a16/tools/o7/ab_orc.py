"""moe o7: paired A/B base vs ORACLE (free tk2f routing prologue). python ab_orc.py <filter> <rounds> <seeds> base.py orc.py"""
import sys, statistics, importlib.util, torch
sys.path.insert(0, ".")
import bench, reference
flt = sys.argv[1]; rounds = int(sys.argv[2]); seeds = [int(s) for s in sys.argv[3].split(",")]; files = sys.argv[4:6]
w = reference.generate_weights(reference.WEIGHTS_SEED)
subs = []
for i, f in enumerate(files):
    spec = importlib.util.spec_from_file_location(f"sub{i}", f)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    subs.append((m, m.prepare(w)))
olib = subs[1][0]._moe__lib()
e0 = bench.empty_cost()
for l, c in bench.SCORED:
    n = f"{l} {c['layer']}"
    if flt not in n:
        continue
    per = [[], []]; ok = True
    for sd in seeds:
        d = bench.make_input(c, w, sd)
        ref = subs[0][0].kernel_fn(d, subs[0][1]); ref = [x.clone() for x in ref] if isinstance(ref, (tuple, list)) else ref.clone()
        olib.moe_orset(1); subs[1][0].kernel_fn(d, subs[1][1]); torch.cuda.synchronize(); olib.moe_orset(2)
        o2 = subs[1][0].kernel_fn(d, subs[1][1]); torch.cuda.synchronize()
        if isinstance(ref, list):
            ok &= all(torch.equal(a, b) for a, b in zip(ref, o2))
        else:
            ok &= torch.equal(ref, o2)
        res = [[], []]
        for r in range(rounds):
            for i, (m, st) in enumerate(subs):
                res[i].append(bench.time_graph(lambda: m.kernel_fn(d, st)) - e0)
        olib.moe_orset(0)
        for i in range(2):
            per[i].append(statistics.median(res[i]))
        del d
        torch.cuda.empty_cache()
    mean = [sum(p) / len(p) for p in per]
    print(f"{n:18s} base {mean[0]:7.2f} orc {mean[1]:7.2f} | per seed", " ".join(f"{b - a:+6.2f}" for a, b in zip(per[0], per[1])),
          f" mean {mean[1] - mean[0]:+6.2f}  bitwise {ok}", flush=True)
