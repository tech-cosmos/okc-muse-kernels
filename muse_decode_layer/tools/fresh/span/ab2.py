"""fresh x8 (b) oracle: configs name=mode:idf:n2unoidf (mode: pr1 oracle bits, see mkstamp.py). One graph per config,
interleaved cold-L2 replays. python tools/fresh/span/ab2.py [sub] [filter] [seeds] [configs] [rounds]"""
import sys, fcntl, statistics, torch
sys.path.insert(0, ".")
import bench
path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fx8/sub_st.py"
flt = sys.argv[2] if len(sys.argv) > 2 else "decode 1 "
seeds = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "13,11,17,7,99").split(",")]
cfgs = [(c.split("=")[0], *[int(v) for v in c.split("=")[1].split(":")]) for c in
        (sys.argv[4] if len(sys.argv) > 4 else "base=0:0:0,hoist=1:0:0,idf=1:1:0,idfn=1:1:1,idf0=0:1:0").split(",")]
rounds = int(sys.argv[5]) if len(sys.argv) > 5 else 6
sub = bench.load_sub(path)
W, st = bench.weights_state(sub)
glib = sub._g10_lib()
m = st["moe"]
setm = lambda md: glib.pr1_set_pf(md, m["w13q"].data_ptr(), m["w13s2"].data_ptr())
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
print("tune T=1:", {k: v for k, v in st.get("tune", {}).items() if k[0] == 1})
for load, c in bench.SCORED:
    if flt not in f"{load} {c['layer']} ":
        continue
    res = {n: [] for n, *_ in cfgs}
    for seed in seeds:
        data = bench.make_input(c, W, seed)
        graphs, outs = {}, {}
        for n, md, idf, nn in cfgs:
            sub.IDF_DEFAULT, sub.N2U_NOIDF = bool(idf), bool(nn)
            setm(md)
            for _ in range(2):
                o = sub.kernel_fn(data, st); torch.cuda.synchronize()
            outs[n] = [x.clone() for x in o]
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                sub.kernel_fn(data, st)
            graphs[n] = g
        sub.IDF_DEFAULT, sub.N2U_NOIDF = False, False
        n0 = cfgs[0][0]
        diff = {n: max(float((a.float() - b.float()).abs().max()) for a, b in zip(outs[n], outs[n0])) for n in outs}
        with open("/tmp/okc-gpu.lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            for r in range(rounds):
                for n, md, idf, nn in (cfgs if r % 2 == 0 else cfgs[::-1]):
                    setm(md); graphs[n].replay(); torch.cuda.synchronize()
                    ts = []
                    for _ in range(10):
                        flush.add_(1)
                        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                        a.record(); graphs[n].replay(); b.record(); torch.cuda.synchronize()
                        ts.append(a.elapsed_time(b) * 1e3)
                    res[n].append(statistics.median(ts))
            fcntl.flock(lk, fcntl.LOCK_UN)
        setm(0)
        print(f"{load} {c['layer']} s{seed}: " + "  ".join(f"{n} {statistics.median(res[n][-rounds:]):.2f}" for n, *_ in cfgs)
              + "  maxdiff " + " ".join(f"{n}:{d:.1e}" for n, d in diff.items() if n != n0), flush=True)
        del graphs
    b0 = statistics.median(res[cfgs[0][0]])
    print(f"== {load} {c['layer']}: " + "  ".join(f"{n} {statistics.median(res[n]):.2f} ({statistics.median(res[n]) - b0:+.2f})" for n, *_ in cfgs), flush=True)
