"""Interleaved A/B of layer variants in one process: python ab.py <filter> <rounds> name=ATTR:VAL;ATTR:VAL ...
VAL is a python literal. Each variant captures its own graph; rounds interleave the replays."""
import sys, ast, statistics, torch
sys.path.insert(0, ".")
import bench
sub = bench.load_sub("submission.py")
flt, rounds = sys.argv[1], int(sys.argv[2])
variants = []
for spec in sys.argv[3:]:
    name, _, body = spec.partition("=")
    variants.append((name, [(k, ast.literal_eval(v)) for k, v in (kv.split(":", 1) for kv in body.split(";") if kv)]))
# all PR1 template configs compiled up front
cfgs = {v for _, kv in variants for k, v in kv if k == "PR1_CFG1"} | {sub.PR1_CFG[1]}
sub.PR1_CFG = {i + 100: c for i, c in enumerate(sorted(cfgs))}
sub.PR1_CFG[1] = sorted(cfgs)[0]
w, st = bench.weights_state(sub)
e = bench.empty_cost()
for load, c in bench.SCORED:
    nm = f"{load} {c['layer']}"
    if flt not in nm:
        continue
    data = bench.make_input(c, w, 99)
    res = {n: [] for n, _ in variants}
    saved = {}
    for r in range(rounds):
        for n, kv in variants:
            for k, v in kv:
                if k == "PR1_CFG1":
                    sub.PR1_CFG[1] = v
                else:
                    saved.setdefault(k, getattr(sub, k)); setattr(sub, k, v)
            res[n].append(bench.time_graph(lambda: sub.kernel_fn(data, st)) - e)
            for k, v in saved.items():
                setattr(sub, k, v)
    print(nm, "  ".join(f"{n} {statistics.median(v):.2f}" for n, v in res.items()), flush=True)
