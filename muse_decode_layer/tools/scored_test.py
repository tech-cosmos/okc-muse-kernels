"""Correctness (reference.check_implementation) on the scored shapes matching a filter, several seeds:
python tools/scored_test.py <file> <filter> <seeds>"""
import sys, torch
sys.path.insert(0, ".")
import bench, reference
sub = bench.load_sub(sys.argv[1]); flt = sys.argv[2]; seeds = int(sys.argv[3])
w, st = bench.weights_state(sub)
bad = 0
for load, c in bench.SCORED:
    n = f"{load} {c['layer']}"
    if flt not in n:
        continue
    for sd in range(seeds):
        data = bench.make_input(c, w, 1000 + sd)
        out = sub.kernel_fn(data, st)
        torch.cuda.synchronize()
        ok, msg = reference.check_implementation(data, reference.collect(data, out), w)
        bad += not ok
        print("PASS" if ok else "FAIL", n, sd, msg[:120], flush=True)
        del data, out
        getattr(reference.generate_input, "cache_clear", lambda: None)()
print("bad", bad)
