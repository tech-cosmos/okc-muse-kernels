# n18: local A/B of TK2F_PF2 on the layer routings (DIAG _diag with dec variants), + bitwise check vs off
import fcntl, sys, torch
sys.path.insert(0, "/tmp/moe18")
import reference, submission as S
w = reference.generate_weights(reference.WEIGHTS_SEED)
st = S.prepare(w)
vs = [("sq", {})]
if len(sys.argv) > 2:  # tk2h W3 at 32 too
    vs += [("h3", {"TK2H_M": {16: 3, 32: 3}})] + [("h3b", {"TK2H_M": {16: 3, 32: 3}})]
S.DIAG_VARIANTS = {"dec": vs + [("sqb", {})], "big": [("s66", {})]}
hdr = S._DIAG_HDR
with open("/tmp/okc-gpu.lock", "w") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    S._diag(st)
