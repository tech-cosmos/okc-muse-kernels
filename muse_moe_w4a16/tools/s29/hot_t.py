# correctness of HOT_M on the layer's T=128/256 routings vs reference + vs HOT off (bitwise)
import sys, os, torch
sys.path.insert(0, '/home/daytona/r-moe/muse_moe_w4a16/tools'); sys.path.insert(0, '.')
import layer_ab as L, reference
sub = L.submission; w = L.w
G = int(os.environ.get("G", "32")); REPS = int(os.environ.get("REPS", "3"))
bad = 0; nd = 0
for (l, sd), loc in sorted(L.R.items()):
    if l not in ("verify 32", "verify 16"): continue
    M = loc.shape[0]
    data = L.mk(loc, sd)
    nt = torch.bincount(loc[loc >= 0].flatten().long(), minlength=32)
    sub.HOT_M = {}; o0 = sub.kernel_fn(data, L.st).clone()
    for r in range(REPS):
        sub.HOT_M = {M: G}; o1 = sub.kernel_fn(data, L.st).clone()
        torch.cuda.synchronize()
        ok, msg = reference.check_implementation(data, o1, w)
        bad += not ok
        d = (o1.float() - o0.float()).abs().max().item(); nd += d > 0
        if r == 0 or not ok: print(l, sd, "hot", [int(x) for x in nt if 64 < x <= 128], "ok" if ok else "FAIL " + msg, "maxdiff vs off", d, flush=True)
print("bad", bad, "nonbitwise", nd)
