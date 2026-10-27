"""Cold prepare timing (s68): run with fresh TORCH_EXTENSIONS_DIR and TRITON_CACHE_DIR, e.g.
rm -rf /tmp/pt; TORCH_EXTENSIONS_DIR=/tmp/pt/te TRITON_CACHE_DIR=/tmp/pt/tr python tools/prep_time.py [submission.py]
The judge fails a run whose prepare() exceeds 300 s: keep this <= 200 s."""
import os, sys, time, functools
t0 = time.time()
sys.path.insert(0, ".")
import torch, reference, bench
sub = bench.load_sub(sys.argv[1] if len(sys.argv) > 1 else "submission.py")
print(f"import {time.time()-t0:.1f}s", flush=True)
T = {}
for n in ("_moe_prepare", "_marlin_bank", "_g10_lib", "_g12_prepare", "_g19_prepare", "_lt_prepare", "_g22_prepare", "_tune", "_g24_prepare", "_moe__lib", "_join_builds"):
    f = getattr(sub, n, None)
    if f is None: continue
    def w(*a, _f=f, _n=n, **k):
        s = time.time(); r = _f(*a, **k); torch.cuda.synchronize(); T[_n] = T.get(_n, 0) + time.time() - s
        print(f"  {_n} {time.time()-s:.1f}s", flush=True); return r
    setattr(sub, n, w)
w = reference.generate_weights(reference.WEIGHTS_SEED)
s = time.time(); st = sub.prepare(w); torch.cuda.synchronize()
print(f"prepare {time.time()-s:.1f}s total; {T}", flush=True)
