import sys, os
sys.path.insert(0, "tools"); sys.path.insert(0, ".")
import bench, submission
sc = dict(bench.SCORED)
loads = [l for l, c in bench.SCORED if c["layer"] == "sliding" and l.startswith(sys.argv[1] if len(sys.argv) > 1 else "verify")]
for rnd in range(3):
    for load in loads:
        c = [c for l, c in bench.SCORED if l == load and c["layer"] == "sliding"][0]
        res = []
        for clb in (32, 16):
            submission.NVS_CL_BATCH = clb
            res.append(round(bench.run_shape(load, c), 2))
        print(rnd, load, "cl32", res[0], "cl16", res[1], flush=True)
