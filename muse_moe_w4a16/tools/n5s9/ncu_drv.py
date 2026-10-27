import sys, torch
sys.path.insert(0, ".")
import bench
sub = bench.load_sub("submission.py"); flt = sys.argv[1]
w, st = bench.weights_state(sub)
for load, c in bench.SCORED:
    if flt != f"{load} {c['layer']}": continue
    data = bench.make_input(c, w, 99)
    sub.kernel_fn(data, st); torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    sub.kernel_fn(data, st); torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
