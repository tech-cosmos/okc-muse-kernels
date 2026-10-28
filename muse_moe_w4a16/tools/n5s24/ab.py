# n5s24: paired H100 A/B of two submission.py files: graph replay (whole kernel_fn), cold L2, alternating, median us.
# usage (from muse_moe_w4a16/): python tools/n5s24/ab.py A.py B.py [bs,...] [seeds]
import sys, importlib.util, fcntl, statistics, torch
sys.path.insert(0, ".")
import reference
def load(p, n):
    spec = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(spec); sys.modules[n] = m; spec.loader.exec_module(m); return m
W = reference.generate_weights(reference.WEIGHTS_SEED)
mods = [load(sys.argv[1], "subA"), load(sys.argv[2], "subB")]
sts = [m.prepare(W) for m in mods]
bss = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "1,8,16,32,128,256").split(",")]
seeds = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "1,2,3").split(",")]
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
for bs in bss:
    res = [[], []]
    for seed in seeds:
        d = reference.generate_input(bs=bs, sharding="ep8", seed=seed)
        gs = []
        for m, st in zip(mods, sts):
            for _ in range(2): m.kernel_fn(d, st)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g): m.kernel_fn(d, st)
            gs.append(g)
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with open("/tmp/okc-gpu.lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            for it in range(60):
                for i in ((0, 1) if it % 2 == 0 else (1, 0)):
                    flush.zero_()
                    e0.record(); gs[i].replay(); e1.record(); e1.synchronize()
                    if it >= 4: res[i].append(e0.elapsed_time(e1) * 1000)
    a, b = statistics.median(res[0]), statistics.median(res[1])
    print(f"bs {bs:4d}  A {a:7.1f}  B {b:7.1f}  B-A {b - a:+6.2f} us", flush=True)
