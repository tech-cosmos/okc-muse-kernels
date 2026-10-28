# n6s3: per-item split of moe_twm on the H100 (build: python tools/n6s3/mk_tl4.py; PYTHONPATH=/tmp/n6s3 SUB=sub_tl python tools/n6s3/itl4.py [board])
# loop = first stage arrived -> tw_item loop end; epi = loop end -> item end (epilogue + release); gap = prev item end -> next start.
# default: 1 L2-warm expert, T=40 (s27 proxy). 'board': reference bs128/bs256 inputs, cold L2 (flush before the call).
import sys, torch, fcntl, statistics as S
sys.path.insert(0, '.'); sys.path.insert(0, 'tools'); sys.path.insert(0, 'tools/s27m')
import layer_ab as L
m = L.submission; st = L.st
m.DIAG_TL = ("x",)
def run(d, M, warm):
    m.TWM_ORDER_M = dict(m.TWM_ORDER_M); m.TWM_ORDER_M[M] = m.TWM_ORDER_M.get(M, m.TWM_ORDER) | 2
    with open('/tmp/okc-gpu.lock', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for _ in range(3):
            if not warm: torch.sum(L.flush.view(torch.int64), 0, out=L.sink)
            m.kernel_fn(d, st)
        torch.cuda.synchronize()
    part, meta = st.pop("_tl")
    tl = part[66 * 32768:].view(torch.int64)[:132 * 64].view(132, 64).cpu()
    act = [c for c in range(132) if tl[c, 63].item() > 0]
    t0 = min(tl[c, 0].item() for c in act)
    rows, gaps, first = [], [], []
    for c in act:
        n = min(tl[c, 1].item(), 15); pe = None
        for k in range(n):
            a, b, desc, le = (tl[c, 2 + 4 * k + i].item() for i in range(4))
            rows.append((desc & 1, (desc >> 8) & 255, (desc >> 1) & 3, (le - a) / 1e3, (b - le) / 1e3, (desc >> 32) / 1e3))
            if pe is not None: gaps.append((a - pe) / 1e3)
            else: first.append((a - t0) / 1e3)
            pe = b
    for g2 in (0, 1):
        for cl in sorted({r[2] for r in rows}):
            for nb in sorted({(r[1] + 7) // 8 for r in rows if r[0] == g2 and r[2] == cl}):
                r = [x for x in rows if x[0] == g2 and x[2] == cl and (x[1] + 7) // 8 == nb]
                if not r: continue
                print(f"  g2 {g2} cl {cl} N{nb*8}: n {len(r)} loop med {S.median(x[3] for x in r):.1f} epi med {S.median(x[4] for x in r):.2f} max {max(x[4] for x in r):.2f} (stores med {S.median(x[5] for x in r):.2f}) us", flush=True)
    if gaps: print(f"  gap between items: n {len(gaps)} med {S.median(gaps):.2f} p90 {sorted(gaps)[int(.9*len(gaps))]:.2f} max {max(gaps):.2f}; first item start med {S.median(first):.1f} max {max(first):.1f}")
    ex = [(tl[c, 63].item() - t0) / 1e3 for c in act]
    print(f"  CTAs {len(act)} exit min/med/max {min(ex):.1f}/{S.median(ex):.1f}/{max(ex):.1f} us")
if len(sys.argv) > 1 and sys.argv[1] == "board":
    import reference
    for bs in (128, 256):
        for seed in (1, 2):
            d = reference.generate_input(bs=bs, sharding='ep8', seed=seed)
            print(f"board bs{bs} seed {seed}:"); run(d, bs, False)
else:
    from twm_warm import mkd
    for ntok in (8, 16, 32):
        print(f"warm 1 expert ntok {ntok}:"); run(mkd(40, ntok), 40, True)
