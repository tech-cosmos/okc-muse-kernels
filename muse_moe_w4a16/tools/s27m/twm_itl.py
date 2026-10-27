# s27: per-item durations of moe_twm (TWM_TL build, SUB=sub_tl) on 1 L2-warm expert with ntok tokens (T = 40):
# item start (after its first stage arrived) -> end (after epilogue + release), us. Usage (muse_moe_w4a16/):
#   sed 's/^#define TWM_TL 0$/#define TWM_TL 1/' submission.py > /tmp/s27/sub_tl.py; PYTHONPATH=/tmp/s27 SUB=sub_tl python tools/s27m/twm_itl.py
import sys, torch, fcntl, statistics as S
sys.path.insert(0, '.'); sys.path.insert(0, 'tools'); sys.path.insert(0, 'tools/s27m')
import layer_ab as L
m = L.submission; st = L.st
from twm_warm import mkd
m.DIAG_TL = ("x",)
for ntok in (8, 16, 24, 32):
    d = mkd(40, ntok)
    m.TWM_ORDER_M = {40: 2}
    with open('/tmp/okc-gpu.lock', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for _ in range(3): m.kernel_fn(d, st)
        torch.cuda.synchronize()
    part, meta = st.pop("_tl")
    tl = part[66 * 32768:].view(torch.int64)[:132 * 64].view(132, 64).cpu()
    act = [c for c in range(132) if tl[c, 63].item() > 0]
    t0 = min(tl[c, 0].item() for c in act)
    rows = []
    for c in act:
        for k in range(min(tl[c, 1].item(), 20)):
            a, b, desc = tl[c, 2 + 3 * k].item(), tl[c, 3 + 3 * k].item(), tl[c, 4 + 3 * k].item()
            rows.append((desc & 1, (a - t0) / 1e3, (b - a) / 1e3))
    for g2 in (0, 1):
        r = [x for x in rows if x[0] == g2]
        print(f"ntok {ntok} g2 {g2}: n {len(r)} start {min(x[1] for x in r):.1f}-{max(x[1] for x in r):.1f} dur med {S.median(x[2] for x in r):.1f} max {max(x[2] for x in r):.1f}", flush=True)
    ex = [(tl[c, 63].item() - t0) / 1e3 for c in act]
    print(f"  exit max {max(ex):.1f} us; CTA start spread {max((tl[c,0].item()-t0)/1e3 for c in act):.1f}")
