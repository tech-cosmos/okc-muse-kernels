import sys, fcntl, torch
sys.path.insert(0, ".")
import bench
sub = bench.load_sub(sys.argv[1])
w, st = bench.weights_state(sub)
for load, c in bench.SCORED:
    if f"{load} {c['layer']}" != sys.argv[2]:
        continue
    data = bench.make_input(c, w, 99)
    g = bench.capture(lambda: sub.kernel_fn(data, st))
    flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda"); sink = torch.zeros((), dtype=torch.int64, device="cuda")
    for r in range(3):
        with open("/tmp/okc-gpu.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            torch.sum(flush.view(torch.int64), 0, out=sink); torch.cuda.synchronize()
            sub.DBG13.zero_(); torch.cuda.synchronize()
            g.replay(); torch.cuda.synchronize()
        d = sub.DBG13[:4095].view(-1, 3).cpu()
        d = d[d[:, 0] > 0].double()
        t0 = d[:, 0].min()
        d = (d - t0) / 1000
        q = lambda v: " ".join(f"{float(v.quantile(p)):6.2f}" for p in (0, .1, .5, .9, 1))
        print(f"blocks {d.shape[0]}  wait {q(d[:,0])} | x ready {q(d[:,1])} | end {q(d[:,2])} | end-x med {float((d[:,2]-d[:,1]).median()):.2f}")
