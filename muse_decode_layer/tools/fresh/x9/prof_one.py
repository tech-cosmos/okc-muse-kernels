"""ncu driver: python /tmp/fx9/prof_one.py <sub.py> '<load> <layer>' (profiles only the last call)"""
import sys, torch
sys.path.insert(0, ".")
import bench
sub = bench.load_sub(sys.argv[1])
w, st = bench.weights_state(sub)
for load, c in bench.SCORED:
    if f"{load} {c['layer']}" == sys.argv[2]:
        data = bench.make_input(c, w, 99)
        for _ in range(3):
            sub.kernel_fn(data, st)
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        sub.kernel_fn(data, st)
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
