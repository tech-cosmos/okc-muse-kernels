"""n5s26: outputs of free_oracle variants vs base0 on every scored shape (2nd call = cached path). python tools/free_check.py <dir>"""
import sys, importlib.util, torch
sys.path.insert(0, ".")
import bench, reference
d = sys.argv[1]
w = reference.generate_weights(reference.WEIGHTS_SEED)
mods = {}
for n in ("base0", "free_a", "free_b1", "free_r", "free_b2"):
    spec = importlib.util.spec_from_file_location(n, f"{d}/{n}.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    mods[n] = (m, m.prepare(w))
for l, c in bench.SCORED:
    data = bench.make_input(c, w, 99)
    ref = None; row = []
    for n, (m, st) in mods.items():
        m.kernel_fn(data, st); r = m.kernel_fn(data, st); torch.cuda.synchronize()
        r = [t.float() for t in r]
        if ref is None: ref = r; continue
        row.append(f"{n}: " + " ".join(f"{(a - b).abs().max().item():.1e}" for a, b in zip(r, ref)))
    print(f"{l} {c['layer']:8s}", " | ".join(row), flush=True)
