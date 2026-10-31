"""fresh x11: run stamped standalone moe (tools/fresh/bs1/stamp.py output). Run from muse_moe_w4a16/:
python ../muse_decode_layer/tools/fresh/bs1/strun.py <stamped submission.py> [bs] [seeds]
Per seed, medians over 21 cold-L2 graph replays of per-stamp min/median/max over tk2f CTAs (us from the first CTA entry),
the norm block's stamps, and the event time of the graph minus the empty graph."""
import sys, fcntl, importlib.util, torch
sys.path.insert(0, ".")
spec = importlib.util.spec_from_file_location("submission", sys.argv[1]); sub = importlib.util.module_from_spec(spec)
sys.modules["submission"] = sub; spec.loader.exec_module(sub)
import bench, reference
bs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
seeds = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "100,101,102,103,104,105").split(",")]
_, st = bench.state()
emp = bench.empty_cost()
names = ["entry", "routed", "prodTMA", "-", "ph1data", "ph1end", "cntok", "ph2data", "end"]
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
o = torch.zeros(1024 * 16, dtype=torch.int64)
for sd in seeds:
    data = reference.generate_input(bs=bs, sharding="ep8", seed=sd)
    sub.kernel_fn(data, st); torch.cuda.synchronize()
    lib = sub._LIB
    ids = [v for k, v in data.items() if torch.is_tensor(v) and v.dtype in (torch.int32, torch.int64) and v.dim() == 2][0]
    E = int(((ids.flatten().unique() >= 0) & (ids.flatten().unique() < 32)).sum())
    g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        sub.kernel_fn(data, st)
        with torch.cuda.graph(g, stream=s):
            sub.kernel_fn(data, st)
    torch.cuda.current_stream().wait_stream(s); g.replay(); torch.cuda.synchronize()
    acc, nacc, tot = [], [], []
    with open("/tmp/okc-gpu.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for _ in range(21):
            flush.add_(1)
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record(); g.replay(); b.record(); torch.cuda.synchronize()
            tot.append(a.elapsed_time(b) * 1e3 - emp)
            lib.moe_tdbg(o)
            t = o.view(1024, 16).double()
            live = t[:1000, 0] > 0
            tt = t[:1000][live][:, :9]
            t0 = tt[:, 0].min()
            acc.append((tt - t0) / 1000)
            nacc.append((t[1000, :4] - t0) / 1000)
            o.zero_()
        fcntl.flock(lk, fcntl.LOCK_UN)
    n = min(x.shape[0] for x in acc)
    Tm = torch.stack([x[:n] for x in acc]).median(0).values
    N = torch.stack(nacc).median(0).values
    print(f"bs{bs} s{sd} E{E} CTAs {n} graph-empty {sorted(tot)[10]:.2f}: " + "  ".join(
        f"{nm} {Tm[:, k].min():.2f}/{Tm[:, k].median():.2f}/{Tm[:, k].max():.2f}" for k, nm in enumerate(names) if nm != "-"), flush=True)
    print(f"   norm: entry {N[0]:.2f} pastwait {N[1]:.2f} Yld {N[2]:.2f} end {N[3]:.2f}", flush=True)
