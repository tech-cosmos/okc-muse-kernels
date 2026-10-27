import sys, torch
sys.path.insert(0, "/tmp/pj71/L")
import os; os.chdir("/tmp/pj71/L")
import bench
sub = bench.load_sub("submission.py")
w, st = bench.weights_state(sub)
names = [(f"{l} {c['layer']}", c) for l, c in bench.SCORED if "decode 1 " in f"{l} {c['layer']} "]
print([n for n, _ in names])
for n, c in names:
    for sd in (99, 7):
        data = bench.make_input(c, w, sd)
        outs = {}
        for v in (0, 0x100, 0x200, 0x300, 0x1200, 0x1300):
            sub.PR1_CFG[1] = sub.PR1_CFG[1][:6] + (v,)
            o = sub.kernel_fn(data, st); torch.cuda.synchronize()
            outs[v] = [t.clone() for t in (o if isinstance(o, (tuple, list)) else (o,))]
        for v in outs:
            same = all(torch.equal(a, b) for a, b in zip(outs[0], outs[v]))
            print(n, sd, hex(v), "bitwise same" if same else "DIFF", flush=True)
