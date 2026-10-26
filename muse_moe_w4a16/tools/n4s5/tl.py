import sys, os, torch, fcntl
sys.path.insert(0, '/tmp/moe16'); os.environ["SUB"] = "sub_t"
import layer_ab as L
sub = L.submission
load, seed = sys.argv[1], int(sys.argv[2])
d = L.mk(L.R[(load, seed)], seed)
G = int(os.environ.get("MOE_GRID", "132"))
for name, glb in [("h8", {"TK2H": 8}), ("h16", {"TK2H": 16}), ("h16w2", {"TK2H": 16, "TK2H_W": 2}), ("h16w4", {"TK2H": 16, "TK2H_W": 4})]:
    for k, v in glb.items(): setattr(sub, k, v)
    acc = []
    for rep in range(5):
        sub.kernel_fn(d, L.st); torch.cuda.synchronize()
        torch.sum(L.flush.view(torch.int64), 0, out=L.sink); torch.cuda.synchronize()
        with open('/tmp/okc-gpu.lock', 'w') as lk:
            fcntl.flock(lk, fcntl.LOCK_EX); sub.kernel_fn(d, L.st); torch.cuda.synchronize()
        o = torch.zeros(1024 * 8, dtype=torch.int64); sub._lib().moe_tdbg(o)
        t = o.view(1024, 8)[:G].double(); t0 = t[:, 0].min()
        t = (t - t0) / 1000.0
        acc.append(t)
    t = torch.stack(acc).median(0).values
    ph1 = t[:, 2]; sp = t[:, 3] - t[:, 2]; end = t[:, 5]
    q = lambda x: f"{x.min():.1f}/{x.median():.1f}/{x.max():.1f}"
    a0 = t[:, 6]
    print(f"{name:6s} start {q(t[:, 0])} ph1end {q(ph1)}  spin {q(sp)}  end {q(end)}  ph2 {q(end - t[:, 3])}")
    for k in glb: setattr(sub, k, 0)
    if name == "h16": torch.save(t, f"/tmp/moe16/tl_{load.replace(' ','')}_{seed}_h16.pt")
    if name == "h8": torch.save(t, f"/tmp/moe16/tl_{load.replace(' ','')}_{seed}_h8.pt")
