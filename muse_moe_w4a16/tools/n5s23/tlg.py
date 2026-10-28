import sys, torch, fcntl
sys.path.insert(0, ".")
import reference, submission, bench
from torch.profiler import profile, ProfilerActivity
w, st = bench.state()
bs = int(sys.argv[1]); seeds = [int(s) for s in sys.argv[2].split(",")]
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
for seed in seeds:
    d = reference.generate_input(bs=bs, sharding="ep8", seed=seed)
    for _ in range(3): submission.kernel_fn(d, st)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): submission.kernel_fn(d, st)
    torch.cuda.synchronize()
    with open("/tmp/okc-gpu.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            for _ in range(3):
                flush.zero_(); torch.cuda.synchronize()
                g.replay(); torch.cuda.synchronize()
    ev = [e for e in p.events() if e.device_type.name == "CUDA" and "fill" not in e.name and "zero" not in e.name.lower() and "elementwise" not in e.name]
    ev.sort(key=lambda e: e.time_range.start)
    ev = ev[-len(ev)//3:]
    t0 = ev[0].time_range.start
    ids = d.topk_ids if hasattr(d, "topk_ids") else d["topk_ids"]
    loc = ids[(ids >= 0) & (ids < 32)]
    cnt = torch.bincount(loc.flatten(), minlength=32).tolist()
    print(f"bs {bs} seed {seed} experts {sum(c>0 for c in cnt)} max {max(cnt)} routes {sum(cnt)}")
    for e in ev:
        print(f"   {e.name[:40]:40s} start {e.time_range.start - t0:8.1f} end {e.time_range.end - t0:8.1f} dur {e.time_range.end - e.time_range.start:7.1f}")
