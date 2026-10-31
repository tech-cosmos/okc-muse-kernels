import sys, importlib.util, torch
sys.path.insert(0, ".")
import bench, reference
w = reference.generate_weights(reference.WEIGHTS_SEED)
mods = []
for i, f in enumerate(["/tmp/o7/base_p.py", "/tmp/o7/orc_p.py"]):
    spec = importlib.util.spec_from_file_location(f"c{i}", f); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    mods.append((m, m.prepare(w)))
olib = mods[1][0]._moe__lib()
def run(i, d):
    r = mods[i][0].kernel_fn(d, mods[i][1]); torch.cuda.synchronize()
    return [x.clone() for x in r] if isinstance(r, (list, tuple)) else [r.clone()]
def eq(a, b): return [torch.equal(x, y) for x, y in zip(a, b)], [float((x.float() - y.float()).abs().max()) for x, y in zip(a, b)]
for l, c in bench.SCORED:
    if sys.argv[1] not in f"{l} {c['layer']}": continue
    d = bench.make_input(c, w, 801)
    b1 = run(0, d); b2 = run(0, d)
    o0 = run(1, d); olib.moe_orset(1); o1 = run(1, d); olib.moe_orset(2); o2 = run(1, d); o2b = run(1, d); olib.moe_orset(0)
    print(l, c['layer'], "base-base", eq(b1, b2), "| orc0-base", eq(o0, b1), "| m1-base", eq(o1, b1), "| m2-m1", eq(o2, o1), "| m2-m2", eq(o2, o2b))
