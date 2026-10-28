# proj n5s11: H100 MEAN (cold L2, graph replay, layer routings via submission._diag) for KQ variants; GPU lock held
import sys, fcntl; sys.path.insert(0, '.')
import torch, bench, submission as S
w, st = bench.state()
S.DIAG_TL = ()
V = [("twm", {})]
for a in sys.argv[1:]:  # name:ns:kq
    n, ns, kq = a.split(":"); V.append((n, {"TWF_SPLIT_M": {128: int(ns), 256: int(ns)}, "TWM_KQ_M": {128: int(kq), 256: int(kq)} if int(kq) else {}}))
V.append(("twmb", {}))
S.DIAG_VARIANTS = {"dec": [("sq", {})], "big": V}
with open("/tmp/okc-gpu.lock", "w") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    S._diag(st)
