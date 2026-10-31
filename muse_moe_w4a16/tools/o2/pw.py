# o2: power/clock under twm load (board bs256 + layer-like), H100. Samples nvidia-smi while replaying the graph.
import sys, fcntl, subprocess, time, torch
sys.path.insert(0, '.'); sys.path.insert(0, 'tools')
import reference, submission as m
W = reference.generate_weights(reference.WEIGHTS_SEED); st = m.prepare(W)
for bs in [int(a) for a in sys.argv[1:]] or [256]:
    d = reference.generate_input(bs=bs, sharding="ep8", seed=3000 + bs)
    m.kernel_fn(d, st); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(20): o = m.kernel_fn(d, st)
    g.replay(); torch.cuda.synchronize()
    with open('/tmp/okc-gpu.lock', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        p = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,power.draw,clocks.sm,clocks.mem,clocks_event_reasons.active,temperature.gpu',
                              '--format=csv,noheader', '-lms', '100'], stdout=subprocess.PIPE, text=True)
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        t0 = time.time(); a.record(); n = 0
        while time.time() - t0 < 4.0:
            g.replay(); n += 1
            if n % 20 == 0: torch.cuda.synchronize()
        b.record(); torch.cuda.synchronize()
        p.terminate(); out = p.communicate()[0]
    us = a.elapsed_time(b) * 1e3 / (n * 20)
    rows = [r.split(', ') for r in out.strip().splitlines()][5:-2]
    print(f"bs{bs}: {us:.1f} us/call (warm, back-to-back), {len(rows)} samples")
    for r in rows[::6]: print('  ', ', '.join(r[1:]))
