import sys, importlib.util, fcntl, torch
sys.path.insert(0, ".")
import reference
from torch.profiler import profile, ProfilerActivity
def load(p, n):
    spec = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(spec); sys.modules[n] = m; spec.loader.exec_module(m); return m
W = reference.generate_weights(reference.WEIGHTS_SEED)
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
for p in sys.argv[1:]:
    m = load(p, "m" + str(abs(hash(p)))); st = m.prepare(W)
    for bs in (1, 16):
        d = reference.generate_input(bs=bs, sharding="ep8", seed=1)
        for _ in range(3): m.kernel_fn(d, st)
        torch.cuda.synchronize()
        with open("/tmp/okc-gpu.lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            with profile(activities=[ProfilerActivity.CUDA]) as pr:
                for _ in range(20): flush.zero_(); m.kernel_fn(d, st)
                torch.cuda.synchronize()
        ev = {}
        for e in pr.events():
            if e.device_type.name == "CUDA" and "elementwise" not in e.name and "fill" not in e.name.lower():
                ev.setdefault(e.name[:40], []).append(e.device_time_total if hasattr(e,'device_time_total') else e.cuda_time_total)
        print(p, bs, {k: round(sorted(v)[len(v)//2], 1) for k, v in ev.items()})
