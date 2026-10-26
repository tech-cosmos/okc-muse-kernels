import sys, os, torch, fcntl
sys.path.insert(0, '/tmp/moe7'); os.environ["SUB"] = "sub_i"
import layer_ab as L
sub = L.submission
G = int(os.environ.get("MOE_GRID", "132"))
for load in sys.argv[1:]:
  for (l, seed), loc in sorted(L.R.items()):
    if l != load or seed not in (7, 99): continue
    d = L.mk(loc, seed)
    for cfg in (84204, 98204):
      sub.TK2F_DYN = cfg; sub.TK2F = 204 if cfg == 84204 else 99204
      acc = []
      for rep in range(5):
        sub.kernel_fn(d, L.st); torch.cuda.synchronize()
        torch.sum(L.flush.view(torch.int64), 0, out=L.sink); torch.cuda.synchronize()
        with open('/tmp/okc-gpu.lock', 'w') as lk:
            fcntl.flock(lk, fcntl.LOCK_EX); sub.kernel_fn(d, L.st); torch.cuda.synchronize()
        o = torch.zeros(1024 * 8, dtype=torch.int64); sub._lib().moe_tdbg(o)
        t = o.view(1024, 8)[:G].double(); t0 = t[:, 0].min()
        t[:, :5] = t[:, :5] - t0
        acc.append(t / 1000.0)
      t = torch.stack(acc).median(0).values
      q = lambda x: f"{x.min():.1f}/{x.median():.1f}/{x.max():.1f}"
      print(f"{load} s{seed} {cfg}: start {q(t[:,0])} ph1end {q(t[:,1])} ph2start {q(t[:,2])} firstspin_end {q(t[:,4])} end {q(t[:,3])} spin2 {q(t[:,6])}")
