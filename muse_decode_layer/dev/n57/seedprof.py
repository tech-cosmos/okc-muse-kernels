"""n57: per-kernel CUDA time (profiler, 5 graph replays, warm) of one file's full shapes per input seed, VJX on/off.
python dev/n57/seedprof.py <file> <filter> <seeds> ; prints top kernels per (shape, seed, vjx)."""
import sys, fcntl, torch, collections
sys.path.insert(0, ".")
import bench
from torch.profiler import profile, ProfilerActivity
sub = bench.load_sub(sys.argv[1]); flt = sys.argv[2]; seeds = [int(x) for x in sys.argv[3].split(",")]
w, st = bench.weights_state(sub)
for load, c in bench.SCORED:
    name = f"{load} {c['layer']}"
    if flt not in name:
        continue
    for sd in seeds:
        data = bench.make_input(c, w, sd)
        for vt in ((), (1, 8, 16, 32, 128, 256)):
            sub.VJX_TOKENS = vt
            g = bench.capture(lambda: sub.kernel_fn(data, st))
            with open("/tmp/okc-gpu.lock", "w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                with profile(activities=[ProfilerActivity.CUDA]) as p:
                    for _ in range(5):
                        g.replay()
                    torch.cuda.synchronize()
            ev = sorted(p.key_averages(), key=lambda e: -e.self_device_time_total)[:6]
            print(f"{name} s{sd} vjx={bool(vt)}: " + " | ".join(f"{e.key[:28]} {e.self_device_time_total / 5:.1f}" for e in ev), flush=True)
        del data
