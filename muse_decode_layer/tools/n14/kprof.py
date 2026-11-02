"""layer n14: per-kernel CUDA time (torch.profiler, eager, after warmup) for submission files on one scored shape.
python tools/n14/kprof.py <shape filter> <file> ..."""
import sys, importlib.util, torch
from torch.profiler import profile, ProfilerActivity
sys.path.insert(0, ".")
import bench, reference as R
flt, files = sys.argv[1], sys.argv[2:]
w = R.generate_weights(R.WEIGHTS_SEED)
(load, c), = [(l, c) for l, c in bench.SCORED if flt in f"{l} {c['layer']}"]
d = bench.make_input(c, w, 99)
for i, f in enumerate(files):
    spec = importlib.util.spec_from_file_location(f"s{i}", f); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    st = m.prepare(w)
    for _ in range(5): m.kernel_fn(d, st)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(20): m.kernel_fn(d, st)
        torch.cuda.synchronize()
    print("==", f)
    for e in sorted(p.key_averages(), key=lambda e: -e.device_time_total)[:14]:
        print(f"  {e.device_time_total / e.count:8.2f} us x{e.count // 20:2d}  {e.key[:90]}")
    del st; torch.cuda.empty_cache()
