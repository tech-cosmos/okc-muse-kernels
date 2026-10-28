import sys, torch
sys.path.insert(0, ".")
import bench, fcntl
sub = bench.load_sub(sys.argv[1]); w, st = bench.weights_state(sub)
for load, c in bench.SCORED:
    if f"{load} {c['layer']}" != "verify 1 sliding": continue
    data = bench.make_input(c, w, 99)
    g = bench.capture(lambda: sub.kernel_fn(data, st))
    flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
    lock = open('/tmp/okc-gpu.lock','w'); fcntl.flock(lock, fcntl.LOCK_EX)
    for r in range(4):
        st["ob8_part"][128 * 32:].zero_()
        flush.add_(1); torch.cuda.synchronize(); g.replay(); torch.cuda.synchronize()
        G = 128
        d = st["ob8_part"][G * 32:].view(torch.int64)[: G * 9].cpu()
        st0 = d[G * 8:G * 9]; t0 = int(st0.min()); rows = d[:G * 8].view(G, 8)
        f = lambda x: (x - t0) / 1000.
        print("start max %.2f  bstore max %.2f  atomic max %.2f" % (f(st0.max()), f(rows[:, 0].max()), f(rows[:, 4].max())))
        tl = rows[rows[:, 2] > 0]
        for k, nm in ((5, "spin"), (1, "spin+sync"), (6, "loads+red"), (2, "end")):
            print("   %-10s min %.2f max %.2f" % (nm, f(tl[:, k].min()), f(tl[:, k].max())))
