"""Correctness of every TUNE option set ON TOP OF TUNE_BASE (s77): per (T, sliding) and option set, force
state["tune"][key] = opts, run kernel_fn eagerly on the scored shape with that T, reference.check_implementation.
python tools/opt_test.py [submission.py] [seeds=1] [T list]"""
import sys, torch
sys.path.insert(0, ".")
import bench, reference
sub = bench.load_sub(sys.argv[1] if len(sys.argv) > 1 else "submission.py")
seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 1
Ts = {int(t) for t in sys.argv[3].split(",")} if len(sys.argv) > 3 else set(sub.TUNE)
w, st = bench.weights_state(sub)
TL = {1: "decode 1", 16: "decode 16", 32: "decode 32", 8: "verify 1", 128: "verify 16", 256: "verify 32"}
bad = n = 0
for T, cands in sub.TUNE.items():
    if T not in Ts:
        continue
    for load, c in bench.SCORED:
        if load != TL[T]:
            continue
        key = (T, c["layer"] == "sliding")
        for opts in cands:
            st["tune"] = {key: opts}
            for sd in range(seeds):
                data = bench.make_input(c, w, 2000 + sd)
                out = sub.kernel_fn(data, st)
                torch.cuda.synchronize()
                ok, msg = reference.check_implementation(data, reference.collect(data, out), w)
                bad += not ok; n += 1
                if not ok:
                    print("FAIL", load, c["layer"], opts, sd, msg[:150], flush=True)
                del data, out
                getattr(reference.generate_input, "cache_clear", lambda: None)()
        print("done", load, c["layer"], len(cands), "sets", flush=True)
print("bad", bad, "of", n)
