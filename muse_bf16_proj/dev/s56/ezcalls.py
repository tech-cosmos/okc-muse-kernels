import sys, torch
sys.path.insert(0, ".")
import bench
sub = bench.load_sub("submission.py")
w, st = bench.weights_state(sub)
real = sub._G24["lib"]
calls = []
class P:
    def __getattr__(self, a): return getattr(real, a)
    def g24_gemm(self, *a):
        calls.append((tuple(a[4].shape), a[3].shape[0], (a[12] >> 16) & 3, a[14].numel(), a[4].data_ptr(), a[14].data_ptr() if a[14].numel() else 0))
        return real.g24_gemm(*a)
sub._G24["lib"] = P()
for load, c in bench.SCORED:
    if load not in ("decode 16", "decode 32", "verify 16", "verify 32"): continue
    data = bench.make_input(c, w, 99)
    calls.clear(); sub.kernel_fn(data, st); torch.cuda.synchronize()
    zs = {z for *_, z in calls if z}
    print(load, c["layer"], [(o, m, "trig" if b & 1 else "", "EZ" if b & 2 else "", zn, "zeroed-by-prev" if p in zs else "") for o, m, b, zn, p, z in calls])
