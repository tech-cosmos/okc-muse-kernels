"""proj n13s1: pr1 per-block globaltimer stamps (pr1_src.cu dbg build) per variant at decode 1 sliding, cold L2.
python stamps.py submission.py R  (env VARJ = json [[name, opts], ...]); times in us from the earliest block start."""
import sys, json, os, statistics, fcntl, torch
sys.path.insert(0, ".")
import bench
sub = bench.load_sub(sys.argv[1]); R = int(sys.argv[2]) if len(sys.argv) > 2 else 5
VAR = [tuple(v) for v in json.loads(os.environ.get("VARJ", '[["base", {}]]'))]
LAY = os.environ.get("LAY", "sliding")
w, st = bench.weights_state(sub)
flush = torch.empty(1 << 30, dtype=torch.uint8, device="cuda"); sink = torch.zeros((), dtype=torch.int64, device="cuda")
for load, c in bench.SCORED:
    if load != "decode 1" or c["layer"] != LAY: continue
    key = (1, c["layer"] == "sliding"); data = bench.make_input(c, w, 99)
    orig = dict(st.get("tune", {}).get(key, {}))
    for v, o in VAR:
        st.setdefault("tune", {})[key] = dict(orig, **o)
        st["pr1_dbg"] = torch.zeros(132 * 16, dtype=torch.int64, device="cuda")
        g = bench.capture(lambda: sub.kernel_fn(data, st))
        acc = []
        lock = open("/tmp/okc-gpu.lock", "w"); fcntl.flock(lock, fcntl.LOCK_EX)
        for r in range(R + 1):
            st["pr1_dbg"].zero_(); torch.sum(flush.view(torch.int64), 0, out=sink); torch.cuda.synchronize()
            g.replay(); torch.cuda.synchronize()
            if r: acc.append(st["pr1_dbg"].view(132, 16).cpu().double())
        fcntl.flock(lock, fcntl.LOCK_UN); lock.close()
        del st["pr1_dbg"]
        rows = []
        for d in acc:
            t0 = d[:, 0][d[:, 0] > 0].min()
            rel = torch.where(d > 0, (d - t0) / 1000, torch.full_like(d, float("nan")))
            rows.append(rel)
        rel = torch.stack(rows).nanmedian(0).values  # [132, 16] median over replays
        def q(x):
            x = x[~torch.isnan(x)]
            return "n/a" if not len(x) else f"min {x.min():5.1f} med {x.median():5.1f} max {x.max():5.1f}"
        print(f"== {v} {o}")
        print("  block start      ", q(rel[:, 0]))
        print("  router wait done ", q(rel[:, 1]))
        print("  router loop done ", q(rel[:, 2]))
        print("  router fenced    ", q(rel[:, 12]))
        print("  last: ticket won ", q(rel[:, 13]))
        print("  last: logits in  ", q(rel[:, 14]))
        print("  last: top8 done  ", q(rel[:, 15]))
        print("  last: ls in      ", q(rel[:, 11]))
        print("  routing done     ", q(rel[:, 3]))
        print("  pooler wait done ", q(rel[:, 4:8].flatten()))
        print("  pooler vg end    ", q(rel[:, 8:11].flatten()))
        e = rel[:, 8:12]
        print("  slowest 6 vgroups (block, vg, end):", sorted([(round(float(e[b, k]), 1), b, k) for b in range(132) for k in range(4) if not torch.isnan(e[b, k])])[-6:])
    st["tune"][key] = orig
