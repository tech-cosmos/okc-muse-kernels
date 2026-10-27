"""s35 local driver: decode 1 full timeline from the mktl_d1.py copy. python run_d1.py <dir with submission.py> [reps]"""
import sys, fcntl, statistics as st, torch
sys.path.insert(0, sys.argv[1])
import reference, submission, json, os
_ov = json.loads(os.environ.get('OV', '{}'))
_p0 = submission._plan
submission._plan = lambda *a: {**_p0(*a), **_ov}
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 20
d = reference.generate_input(seed=7, batch=1, ctx=80000, heads="tp8", layer="full", page=1, q_len=1, dist="coherent")
dev = d["qkv"].device
ts = torch.zeros(65536 + 64, dtype=torch.int64, device=dev)
if hasattr(submission, "_TS35"):
    submission._TS35[0] = ts
ref = None
out = submission.kernel_fn(d); torch.cuda.synchronize()
if ref is not None:
    print("maxerr", (out.float() - ref.float()).abs().max().item())
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    for _ in range(3): submission.kernel_fn(d)
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    submission.kernel_fn(d)
flush = torch.empty(512 << 20, dtype=torch.uint8, device=dev); sink = torch.empty(1, device=dev, dtype=torch.int64)
e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
times, rows = [], []
lk = open("/tmp/okc-gpu.lock", "w")
for r in range(reps):
    torch.sum(flush.view(torch.int64), 0, out=sink[0])
    fcntl.flock(lk, fcntl.LOCK_EX)
    e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
    fcntl.flock(lk, fcntl.LOCK_UN)
    times.append(e0.elapsed_time(e1) * 1000)
    rows.append(ts.clone().cpu())
print(f"replay us median {st.median(times):.2f} min {min(times):.2f}")
if not hasattr(submission, "_TS35"):
    sys.exit()
P = 511
ph = {k: [] for k in ["q", "loop", "store", "red", "prog", "t0", "tN", "start_spread", "end_after_first", "rf_wait", "rf_rel", "rf_load", "rf_end", "total"]}
for t in rows:
    a = t[:P * 32].view(P, 32).double()
    ck = a[:, [1, 2, 3, 4, 5]]
    # calibrate clock (cycles) -> us with globaltimer per program
    gt0, gt1 = a[:, 0], a[:, 6]
    cyc = ck[:, 4] - ck[:, 0]
    f = (cyc / (gt1 - gt0).clamp(min=1)).median().item()  # cycles per ns
    us = lambda x: x / f / 1000
    ph["q"].append(us(ck[:, 1] - ck[:, 0]).median().item())
    ph["loop"].append(us(ck[:, 2] - ck[:, 1]).median().item())
    ph["store"].append(us(ck[:, 3] - ck[:, 2]).median().item())
    ph["red"].append(us(ck[:, 4] - ck[:, 3]).median().item())
    ph["prog"].append(us(cyc).median().item())
    nt = a[:, 7]
    t0 = a[:, 10] - ck[:, 1]
    ph["t0"].append(us(t0).median().item())
    tn = (a[:, 11:14] - a[:, 10:13]).flatten()
    ph["tN"].append(us(tn[tn > 0]).median().item())
    first = gt0.min()
    ph["start_spread"].append(((gt0.max() - first) / 1000).item())
    ph["end_after_first"].append(((gt1.max() - first) / 1000).item())
    rf = t[65536:65540].double()
    ph["rf_wait"].append(((rf[1] - first) / 1000).item())
    ph["rf_rel"].append(((rf[1] - gt1.max()) / 1000).item())
    ph["rf_load"].append(((rf[2] - rf[1]) / 1000).item())
    ph["rf_end"].append(((rf[3] - rf[1]) / 1000).item())
    ph["total"].append(((rf[3] - first) / 1000).item())
print("ntiles per program", torch.bincount(rows[-1][:P * 32].view(P, 32)[:, 7]).tolist(), f"clk {f:.3f} GHz")
print("phase medians over reps (us):", {k: round(st.median(v), 2) for k, v in ph.items()})
