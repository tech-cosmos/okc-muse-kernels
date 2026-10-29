"""proj n7s1 R2 ceiling oracle (timing only, never submit): python mk_n3one.py <layer submission.py> <outdir>
base0.py = TUNE = {}; n3one.py = same, but moe_norm3/norm2 after moe_tk2f run on out[:1] (1 block: still zeroes tk2f's
counters, so the PDL secondary's launch/wait/exit stays, its y32 reload + norm work go). base - n3one bounds what R2
(column-stationary gemm2 + in-register norm) can save at T=16/32 (T=1/T=8 already fuse the norm: n2u / nu8)."""
import os, sys
L = open(sys.argv[1]).read().split("\n"); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
i = next(k for k, l in enumerate(L) if l.startswith("TUNE_BASE = {"))
L.insert(i, "TUNE = {}  # proj n7 oracle: defaults only")
s = "\n".join(L); open(f"{out}/base0.py", "w").write(s)
old = 'lib.moe_norm3_launch(y32, ids, tw, st["w1"], st["start"], st["count"], _moe_POST_NORM_EPS, out, st["cnt"])\n                return out'
old2 = ('            lib.moe_norm2_launch(y32, inv, ids, tw, st["w1"], st["gs2"], st["start"], st["count"], _moe_POST_NORM_EPS, out,\n'
        '                                 st["cnt"])\n            return out')
assert s.count(old) == 1 and s.count(old2) == 1
s = s.replace(old, old.replace(", out, st[", ", out[:1], st[")).replace(old2, old2.replace(", out,\n", ", out[:1],\n"))
open(f"{out}/n3one.py", "w").write(s); print("ok")
