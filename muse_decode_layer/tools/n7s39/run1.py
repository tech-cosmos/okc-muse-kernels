import sys, importlib.util, torch, fcntl
sys.path.insert(0, ".")
import bench, reference
f, flt = sys.argv[1], sys.argv[2]
w = reference.generate_weights(reference.WEIGHTS_SEED)
spec = importlib.util.spec_from_file_location("sub", f); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
st = m.prepare(w)
c = [c for l, c in bench.SCORED if flt == f"{l} {c['layer']}"][0]
d = bench.make_input(c, w, 99)
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    for _ in range(3): m.kernel_fn(d, st)
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g): m.kernel_fn(d, st)
torch.cuda.synchronize()
with open("/tmp/okc-gpu.lock", "w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(30): g.replay()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
