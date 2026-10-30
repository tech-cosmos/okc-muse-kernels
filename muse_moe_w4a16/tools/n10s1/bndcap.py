"""Capture the args of the second _boundary call per scored sliding T (eager kernel_fn), pickle to /tmp/moeend/bnd_args.pt."""
import sys, os, torch
sys.path.insert(0, ".")
import bench
sub = bench.load_sub(sys.argv[1])
w, st = bench.weights_state(sub)
real = sub._boundary
calls = []
class W:
    def __getitem__(self, g):
        def f(*a, **k):
            calls.append((g, [x.clone() if torch.is_tensor(x) else x for x in a],
                          {kk: (v.clone() if torch.is_tensor(v) else v) for kk, v in k.items()}))
            return real[g](*a, **k)
        return f
sub._boundary = W()
out = {}
for load, c in bench.SCORED:
    nm = f"{load} {c['layer']}"
    if nm not in ("decode 1 sliding", "decode 16 sliding", "decode 32 sliding", "verify 1 sliding"):
        continue
    data = bench.make_input(c, w, 99)
    calls.clear()
    r = sub.kernel_fn(data, st); torch.cuda.synchronize()
    out[nm] = calls[-1]
    g, a, k = calls[-1]
    print(nm, g, len(calls), {kk: v for kk, v in k.items() if not torch.is_tensor(v)},
          [tuple(x.shape) if torch.is_tensor(x) else x for x in a])
torch.save(out, "/tmp/moeend/bnd_args.pt")
