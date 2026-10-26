"""Router m>32 via the submission's g7_gemv (hi/lo mma, token groups) vs the current path. python dev/r7.py"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench, reference
w, st = bench.state()
mod = bench.submission._mod
W = w["router"]
e = bench.empty_cost()
for m in (128, 256):
    data = reference.generate_input(m, "router", seed=3 + m)
    x = data["x"]
    ref = x.double() @ W.double().t()
    t0 = bench.time_graph(lambda: bench.submission.kernel_fn(data, st)) - e
    print(f"m={m} current {t0:.2f}", flush=True)
    for mt, nwr, nwk in ((4, 2, 2), (2, 2, 4)):
        for cs in (4, 8, 16):
            tgs = m // (8 * mt)
            out = torch.empty(m, 256, dtype=torch.float32, device="cuda")
            try:
                mod.g7_gemv(st["router_hl_p"], x, out, 256, mt, nwr, nwk, 8, 2, cs, tgs)
                torch.cuda.synchronize()
            except Exception as ex:
                print("cfg fail", mt, cs, str(ex)[:80]); continue
            err = ((out.double() - ref).norm() / ref.norm()).item()
            t = bench.time_graph(lambda: mod.g7_gemv(st["router_hl_p"], x, out, 256, mt, nwr, nwk, 8, 2, cs, tgs)) - e
            print(f"m={m} g7 mt{mt} nwr{nwr} nwk{nwk} cs{cs} tgs{tgs}: {t:.2f} us err {err:.2e}", flush=True)
