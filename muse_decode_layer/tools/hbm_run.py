"""n7s4: HBM-busy probe (lead 06:00Z big-bet step 1). Run under nsys GPU metrics (tools/hbm_fold.py reads the result):
python tools/hbm_run.py <submission.py> "<load> <layer>" [reps]
Replays the shape's graph `reps` times, each after a 512 MB L2 flush and a ~40 us idle spin, so the DRAM-bandwidth samples
(5 us apart at 200 kHz) land at random phases of the replay and can be folded per replay start (super-resolution)."""
import sys, os, fcntl, torch
sys.path.insert(0, ".")
import bench
sub = bench.load_sub(sys.argv[1]); flt = sys.argv[2]; reps = int(sys.argv[3]) if len(sys.argv) > 3 else 400
w, st = bench.weights_state(sub)
for load, c in bench.SCORED:
    if flt != f"{load} {c['layer']}":
        continue
    data = bench.make_input(c, w, 99)
    g = bench.capture(lambda: sub.kernel_fn(data, st))
    flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda"); sink = torch.zeros((), dtype=torch.int64, device="cuda")
    with open("/tmp/okc-gpu.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for i in range(reps):
            torch.sum(flush.view(torch.int64), 0, out=sink)
            torch.cuda._sleep(40000 + (i * 7919) % 20000)  # idle gap, random-ish phase vs the sampler
            g.replay()
        torch.cuda.synchronize()
    print("done", flt, reps)
