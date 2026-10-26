import sys, os, torch, fcntl
sys.path.insert(0, '/tmp/moe16'); os.environ["SUB"] = "sub_t"
import layer_ab as L
sub = L.submission
G = int(os.environ.get("MOE_GRID", "132"))
for load in sys.argv[1:]:
  for (l, seed), loc in sorted(L.R.items()):
    if l != load: continue
    d = L.mk(loc, seed)
    sub.TK2H = 8
    acc = []
    for rep in range(5):
        sub.kernel_fn(d, L.st); torch.cuda.synchronize()
        torch.sum(L.flush.view(torch.int64), 0, out=L.sink); torch.cuda.synchronize()
        with open('/tmp/okc-gpu.lock', 'w') as lk:
            fcntl.flock(lk, fcntl.LOCK_EX); sub.kernel_fn(d, L.st); torch.cuda.synchronize()
        o = torch.zeros(1024 * 8, dtype=torch.int64); sub._lib().moe_tdbg(o)
        t = o.view(1024, 8)[:G].double(); t0 = t[:, 0].min()
        acc.append((t - t0) / 1000.0)
    t = torch.stack(acc).median(0).values
    m = lambda k: t[:, k].median().item()
    q = lambda x: f"{x.min():.1f}/{x.median():.1f}/{x.max():.1f}"
    print(f"{load} s{seed}: p1 start {m(0):.1f} row1 {m(1):.1f} row2 {m(7):.1f} | ph1end {q(t[:,2])} cnt {m(3):.1f} row1 {m(6):.1f} row2 {m(4):.1f} end {q(t[:,5])}")
