import os, sys, time, fcntl, torch
sys.path.insert(0, "/tmp/pj"); sys.path.insert(0, os.path.expanduser("~/r-proj/muse_bf16_proj"))
from lt import mod
import bench, reference
W = reference.generate_weights(reference.WEIGHTS_SEED)
wsb = 64 << 20
ws = torch.empty(wsb, dtype=torch.uint8, device="cuda")
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
for spec in sys.argv[1:]:
    m, layer = spec.split(":"); m = int(m)
    data = reference.generate_input(m, layer, seed=7)
    x = data["x"]; n = data["config"]["n"]; k = x.shape[1]
    wt = W[layer][:n].contiguous()
    ref = torch.nn.functional.linear(x, wt)
    al = mod.lt_enum(m, n, k, wsb, 10**7)
    out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    N = al.shape[0]
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(N)]
    best = {}
    t0 = time.time()
    for rep in range(2):
        with open("/tmp/okc-gpu.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            sts = []
            for i in range(N):
                flush.zero_()
                ev[i][0].record()
                sts.append(mod.lt_mm(wt, x, out, al[i], ws))
                ev[i][1].record()
            torch.cuda.synchronize()
        for i in range(N):
            if sts[i] == 0:
                t = ev[i][0].elapsed_time(ev[i][1]) * 1000
                best[i] = min(best.get(i, 1e9), t)
    order = sorted(best, key=best.get)
    tc = bench.time_graph(lambda: torch.nn.functional.linear(x, wt)) - bench.empty_cost()
    top = []
    for i in order[:40]:
        a = al[i].clone()
        mod.lt_mm(wt, x, out, a, ws); torch.cuda.synchronize()
        err = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
        if not err < 1e-3: continue
        t2 = bench.time_graph(lambda: mod.lt_mm(wt, x, out, a, ws), reps=50) - bench.empty_cost()
        top.append((t2, i))
    top.sort()
    print(f"m={m} {layer} cublas {tc:.2f} valid {N} ok {len(best)} {time.time()-t0:.0f}s best", " ".join(f"{t:.2f}{mod.lt_desc(al[i]).tolist()[1:]}" for t, i in top[:4]), flush=True)
    torch.save({"top": [(t, al[i].clone()) for t, i in top]}, f"/tmp/pj/lte_{m}_{layer}.pt")
