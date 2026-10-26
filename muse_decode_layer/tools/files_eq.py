"""Bitwise output comparison of two submission files on the scored shapes (eager kernel_fn, same inputs):
python tools/files_eq.py <file A> <file B> [filter] [seeds]. Prints max |diff| per output tensor."""
import sys, importlib.util, torch
sys.path.insert(0, ".")
import bench, reference
fa, fb = sys.argv[1], sys.argv[2]; flt = sys.argv[3] if len(sys.argv) > 3 else ""
seeds = int(sys.argv[4]) if len(sys.argv) > 4 else 2
w = reference.generate_weights(reference.WEIGHTS_SEED)
subs = []
for i, f in enumerate((fa, fb)):
    spec = importlib.util.spec_from_file_location(f"sub{i}", f)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    subs.append((m, m.prepare(w)))
bad = 0
for load, c in bench.SCORED:
    n = f"{load} {c['layer']}"
    if flt not in n:
        continue
    for sd in range(seeds):
        outs = []
        for m, st in subs:
            data = bench.make_input(c, w, 1000 + sd)
            o = m.kernel_fn(data, st); torch.cuda.synchronize()
            outs.append([t.clone() for t in o])
        d = [(a.float() - b.float()).abs().max().item() if a.dtype.is_floating_point else int((a != b).sum()) for a, b in zip(*outs)]
        bad += any(x != 0 for x in d)
        print(n, sd, d)
print("not identical:", bad)
