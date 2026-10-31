"""x23: is each load's output bitwise deterministic run-to-run (same state, fresh identical inputs)? Which output differs?"""
import sys, torch
sys.path.insert(0, ".")
import bench, reference
sub = bench.load_sub(sys.argv[1]); flts = sys.argv[2].split(",")
w = reference.generate_weights(reference.WEIGHTS_SEED); s = sub.prepare(w)
for load, c in bench.SCORED:
    name = f"{load} {c['layer']}"
    if name not in flts:
        continue
    outs = []
    for r in range(4):
        d = bench.make_input(c, w, 99)
        o = sub.kernel_fn(d, s); torch.cuda.synchronize()
        o = o if isinstance(o, (list, tuple)) else [o]
        outs.append([x.clone() for x in o])
        del d
    for r in range(1, 4):
        diffs = [(i, (a.float() - b.float()).abs().max().item(), (a != b).sum().item(), tuple(a.shape))
                 for i, (a, b) in enumerate(zip(outs[0], outs[r])) if not torch.equal(a, b)]
        print(name, "run", r, "vs 0:", diffs if diffs else "bitwise", flush=True)
