"""n5s29: one layer call (d16 sliding, seed) with PR W under ncu profiler range. python dev/n5s29/pr_ncu.py <sub> <W> <seed>"""
import sys, torch
sys.path.insert(0, ".")
import bench
sub = bench.load_sub(sys.argv[1]); W = int(sys.argv[2]); sd = int(sys.argv[3])
w, st = bench.weights_state(sub)
c = [c for l, c in bench.SCORED if l == "decode 16" and c["layer"] == "sliding"][0]
data = bench.make_input(c, w, sd)
sub._ab(16, tk2pr=W) if W else sub._ab(16, bal=False)
for _ in range(2): sub.kernel_fn(data, st)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart(); sub.kernel_fn(data, st); torch.cuda.synchronize(); torch.cuda.cudart().cudaProfilerStop()
