# s27: per-kernel durations (torch.profiler) of the big-M path on 1 L2-warm expert, ntok tokens (T = 40).
import sys, torch, fcntl
sys.path.insert(0, '.'); sys.path.insert(0, 'tools'); sys.path.insert(0, 'tools/s27m')
import layer_ab as L
m = L.submission; st = L.st
from twm_warm import mkd
from torch.profiler import profile, ProfilerActivity
for ntok in (8, 16, 32):
    d = mkd(40, ntok)
    for _ in range(3): m.kernel_fn(d, st)
    torch.cuda.synchronize()
    with open('/tmp/okc-gpu.lock', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            for _ in range(10): m.kernel_fn(d, st)
            torch.cuda.synchronize()
    ev = {}
    for e in p.events():
        if e.device_type.name == "CUDA": ev.setdefault(e.name[:40], []).append(e.device_time)
    print(ntok, {k: round(sorted(v)[len(v) // 2], 1) for k, v in ev.items()}, flush=True)
