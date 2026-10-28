import sys, fcntl; sys.path.insert(0, '.')
import torch, bench, submission as S
w, st = bench.state()
S.DIAG_TL = ()
S.DIAG_VARIANTS = {"dec": [("sq", {})], "big": [("twm", {}), ("partx", {"TWS_PARTX": 1}), ("noscan32", {"TWM_STEAL_M": {128: 32 | 1 << 22 | 1 << 24}}), ("twmb", {})]}
with open("/tmp/okc-gpu.lock", "w") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    S._diag(st)
