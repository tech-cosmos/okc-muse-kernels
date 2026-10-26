import sys, os
sys.path.insert(0, '/tmp/moe16')
import layer_ab as L, torch
sub = L.submission; sub.DIAG_TL = (0,)
NG = int(os.environ.get("MOE_GRID", "132"))
for (l, sd), loc in sorted(L.R.items()):
    if l not in sys.argv[1].split(","): continue
    data = L.mk(loc, sd)
    for v in [int(x) for x in sys.argv[2].split(",")]:
        sub.TWM_ORDER = 2; sub.TWF_SPLIT_M = {128: v, 256: v} if v else {}
        for _ in range(3): sub.kernel_fn(data, L.st)
        torch.cuda.synchronize()
        part, meta = L.st.pop("_tl")
        tl = part[528 * 16384:].view(torch.int64)[:NG * 64].view(NG, 64).cpu()
        t0 = tl[:, 0].min().item(); us = lambda x: (x - t0) / 1000
        n = tl[:, 1]
        work = sum(us(tl[c, 3 + 3 * k].item()) - us(tl[c, 2 + 3 * k].item()) for c in range(NG) for k in range(min(n[c].item(), 20)))
        ex = sorted(us(tl[c, 63].item()) for c in range(NG))
        print(f"{l} s{sd} split {v}: items {n.sum().item()} work/G {work / NG:.1f} exit min/med/max {ex[0]:.1f}/{ex[NG // 2]:.1f}/{ex[-1]:.1f} loss {ex[-1] - work / NG:.1f}", flush=True)
sub.TWM_ORDER = 0
