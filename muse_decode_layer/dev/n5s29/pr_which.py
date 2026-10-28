import sys, torch
sys.path.insert(0, ".")
import bench
sub = bench.load_sub(sys.argv[1]); w, st = bench.weights_state(sub)
for l, c in bench.SCORED:
    if c["batch"] * c["q_len"] not in (16, 32): continue
    data = bench.make_input(c, w, 7)
    for on in (False, True):
        sub._ab(16)
        for T in (16, 32):
            for s in (True, False):
                d = sub.TUNE_BASE.setdefault((T, s), {})
                if on: d["tk2pr"] = 3
        sub.kernel_fn(data, st); torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
            sub.kernel_fn(data, st); torch.cuda.synchronize()
        ks = [e.name for e in p.events() if "tk2f" in e.name]
        print(l, c["layer"], on, [k[:20] + "..." + k[k.find("<"):k.find(">") + 1] for k in ks])
