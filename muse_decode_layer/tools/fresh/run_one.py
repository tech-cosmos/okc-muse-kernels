"""Run one scored layer shape a few times (for ncu; run from muse_decode_layer/): python tools/fresh/run_one.py <submission.py> '<load> <layer>' [calls]"""
import sys, torch
sys.path.insert(0, ".")
import bench
sub = bench.load_sub(sys.argv[1])
w, st = bench.weights_state(sub)
for load, c in bench.SCORED:
    if f"{load} {c['layer']}" == sys.argv[2]:
        data = bench.make_input(c, w, 99)
        for _ in range(int(sys.argv[3]) if len(sys.argv) > 3 else 3):
            sub.kernel_fn(data, st)
        torch.cuda.synchronize()
