# n5s17 TWS H100 MEAN (cold L2, graph replay, layer routings via submission._diag); GPU lock held only in _diag
import sys, fcntl; sys.path.insert(0, '.')
import torch, bench, submission as S
w, st = bench.state()
S.DIAG_TL = ()
V = [("twm", {})]
for a in sys.argv[1:]:  # name:stw
    n, v = a.split(":"); v = int(v, 0); V.append((n, {"TWM_STEAL_M": {128: v, 256: v}}))
V.append(("twmb", {}))
S.DIAG_VARIANTS = {"dec": [("sq", {})], "big": V}
S._DIAG_HDR = [h for h in S._DIAG_HDR if h[2] >= 128] if False else S._DIAG_HDR
with open("/tmp/okc-gpu.lock", "w") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    S._diag(st)
