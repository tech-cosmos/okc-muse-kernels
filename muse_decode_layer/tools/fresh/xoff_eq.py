"""fresh f1s9: HOT_XN on (default) vs HOT_XOFF (hot grid + 1e8) at verify 32, bitwise, several seeds."""
import sys, torch
sys.path.insert(0, ".")
import bench
sub = bench.load_sub("submission.py")
w, st = bench.weights_state(sub)
bad = 0
for load, c in bench.SCORED:
    if load != "verify 32":
        continue
    key = (256, c["layer"] == "sliding")
    for sd in range(int(sys.argv[1]) if len(sys.argv) > 1 else 4):
        data = bench.make_input(c, w, 3000 + sd)
        outs = []
        for opts in ({}, {"hot": sub.HOT256 + sub.HOT_XOFF}):
            st["tune"] = {key: opts}
            o = sub.kernel_fn(data, st); torch.cuda.synchronize()
            outs.append([t.clone() for t in (o if isinstance(o, (tuple, list)) else (o,))])
        eq = all(torch.equal(a, b) for a, b in zip(*outs))
        bad += not eq
        print(load, c["layer"], sd, "bitwise" if eq else "DIFF", flush=True)
print("xoff_eq bad", bad)
