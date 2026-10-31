"""proj n13s1: pr8 per-block TR stamps (PR8_TRACE build with a state-held part buffer, /tmp copy only) at verify 1.
python stamps8.py submission.py R  (env VARJ, LAY); us from the earliest block start."""
import sys, json, os, fcntl, torch
sys.path.insert(0, ".")
import bench
sub = bench.load_sub(sys.argv[1]); R = int(sys.argv[2]) if len(sys.argv) > 2 else 5
VAR = [tuple(v) for v in json.loads(os.environ.get("VARJ", '[["base", {}]]'))]
LAY = os.environ.get("LAY", "sliding")
w, st = bench.weights_state(sub)
flush = torch.empty(1 << 30, dtype=torch.uint8, device="cuda"); sink = torch.zeros((), dtype=torch.int64, device="cuda")
NAMES = ["start", "x staged", "router stored", "router released", "pooler loop end", "epilogue end", "-", "finisher end"]
for load, c in bench.SCORED:
    if load != "verify 1" or c["layer"] != LAY: continue
    key = (8, c["layer"] == "sliding"); data = bench.make_input(c, w, 99)
    orig = dict(st.get("tune", {}).get(key, {}))
    for v, o in VAR:
        st.setdefault("tune", {})[key] = dict(orig, **o)
        g = bench.capture(lambda: sub.kernel_fn(data, st))
        buf = st["pr8_part"]; acc = []
        lock = open("/tmp/okc-gpu.lock", "w"); fcntl.flock(lock, fcntl.LOCK_EX)
        for r in range(R + 1):
            buf[8 * 8 * 256:].zero_(); torch.sum(flush.view(torch.int64), 0, out=sink); torch.cuda.synchronize()
            g.replay(); torch.cuda.synchronize()
            if r: acc.append(buf[8 * 8 * 256:].view(torch.int64)[:129 * 8].view(129, 8).cpu().double())
        fcntl.flock(lock, fcntl.LOCK_UN); lock.close()
        rows = []
        for d in acc:
            t0 = d[:129, 0].min()
            rows.append(torch.where(d > 0, (d - t0) / 1000, torch.full_like(d, float("nan"))))
        rel = torch.stack(rows).nanmedian(0).values
        print(f"== {v} {o}")
        fin = rel[128]
        print("  finisher: entry %.1f  past wait %.1f  counter done %.1f  sums done %.1f  routed/end %.1f" % tuple(fin[[0, 1, 2, 3, 7]].tolist()))
        for i, nm in enumerate(NAMES):
            x = rel[:128, i]
            x = x[~torch.isnan(x)]
            if len(x): print(f"  {nm:16s} min {x.min():5.1f} med {x.median():5.1f} max {x.max():5.1f}")
    st["tune"][key] = orig
