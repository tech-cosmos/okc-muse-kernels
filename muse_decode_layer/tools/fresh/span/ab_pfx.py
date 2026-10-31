"""fresh x8 (b) oracle: tk2f PFX (first w13 rows of CTAs 32.. L2-prefetched by CTAs 0..31 at T=1), modes = rows per CTA.
python tools/fresh/span/ab.py [sub_st.py] [filter] [seeds] [modes] [rounds]"""
import sys, fcntl, statistics, torch
sys.path.insert(0, ".")
import bench
path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fx8/sub_st.py"
flt = sys.argv[2] if len(sys.argv) > 2 else "decode 1 "
seeds = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "13,11,17,7,99").split(",")]
modes = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "0,1,3,5").split(",")]
rounds = int(sys.argv[5]) if len(sys.argv) > 5 else 6
sub = bench.load_sub(path)
W, st = bench.weights_state(sub)
mlib = sub._moe__lib()

setm = lambda md: mlib.moe_set_pfx(md)
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
for load, c in bench.SCORED:
    if flt not in f"{load} {c['layer']} ":
        continue
    res = {md: [] for md in modes}
    for seed in seeds:
        data = bench.make_input(c, W, seed)
        outs = {}
        for md in modes:
            setm(md); o = sub.kernel_fn(data, st); torch.cuda.synchronize()
            outs[md] = [x.clone() for x in o]
        bad = [md for md in modes if any(not torch.equal(a, b) for a, b in zip(outs[md], outs[modes[0]]))]
        setm(0)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            sub.kernel_fn(data, st)
        with open("/tmp/okc-gpu.lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            for r in range(rounds):
                for md in (modes if r % 2 == 0 else modes[::-1]):
                    setm(md); g.replay(); torch.cuda.synchronize()
                    ts = []
                    for _ in range(10):
                        flush.add_(1)
                        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                        a.record(); g.replay(); b.record(); torch.cuda.synchronize()
                        ts.append(a.elapsed_time(b) * 1e3)
                    res[md].append(statistics.median(ts))
            fcntl.flock(lk, fcntl.LOCK_UN)
        print(f"{load} {c['layer']} s{seed}: " + "  ".join(f"m{md} {statistics.median(res[md][-rounds:]):.2f}" for md in modes)
              + (f"  NOT BITWISE {bad}" if bad else "  bitwise"), flush=True)
        setm(0)
    b0 = statistics.median(res[modes[0]])
    print(f"== {load} {c['layer']} median over seeds x rounds: " + "  ".join(f"m{md} {statistics.median(res[md]):.2f} ({statistics.median(res[md]) - b0:+.2f})" for md in modes), flush=True)
