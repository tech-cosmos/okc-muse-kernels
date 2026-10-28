# n6s3: H200 DIAG copy (NEVER submit): 4-stamp twm timeline (tools/n6s3/mk_tl4.py) on the layer's embedded v16/v32
# routings (the judge rejects any 'import reference'). Per routing one 'n6tl' line:
# per item kind (g2 cl N): n, loop med, epi med, stores med (us); gap between items; exit med/max.
# usage (muse_moe_w4a16/): python tools/n6s3/mk_tl4.py && python tools/n6s3/mkdiag.py -> /tmp/n6s3/diag/submission.py
import os
s = open('/tmp/n6s3/sub_tl.py').read()
s = s.replace('\nDIAG = 0\n', '\nDIAG = 1\n', 1)
s = s.replace('        _diag(st)\n    return st\n', '        _diag_n6s3(st)\n    return st\n', 1)
s += r'''

def _n6tl(st, data, name):
    import statistics as S
    M = data["config"]["bs"]
    g = globals()
    old = g["TWM_ORDER_M"]
    g["TWM_ORDER_M"] = dict(old); g["TWM_ORDER_M"][M] = old.get(M, TWM_ORDER) | 2
    old_tl = g["DIAG_TL"]; g["DIAG_TL"] = ("x",)
    try:
        flush = torch.zeros(256 << 20, dtype=torch.uint8, device="cuda").view(torch.int64)
        sink = torch.empty((), dtype=torch.int64, device="cuda")
        for _ in range(3):
            torch.sum(flush, 0, out=sink)
            kernel_fn(data, st)
        torch.cuda.synchronize()
        del flush
    finally:
        g["TWM_ORDER_M"] = old; g["DIAG_TL"] = old_tl
    if "_tl" not in st:
        print(f"n6tl {name}: no timeline", flush=True); return
    part, meta = st.pop("_tl")
    tl = part[66 * 32768:].view(torch.int64)[:132 * 64].view(132, 64).cpu()
    act = [c for c in range(132) if tl[c, 63].item() > 0]
    t0 = min(tl[c, 0].item() for c in act)
    rows, gaps = [], []
    for c in act:
        n = min(tl[c, 1].item(), 15); pe = None
        for k in range(n):
            a, b, d, le = (tl[c, 2 + 4 * k + i].item() for i in range(4))
            rows.append(((d & 1, (d >> 1) & 3, (((d >> 8) & 255) + 7) // 8 * 8), (le - a) / 1e3, (b - le) / 1e3, (d >> 32) / 1e3))
            if pe is not None: gaps.append((a - pe) / 1e3)
            pe = b
    ks = sorted({r[0] for r in rows})
    out = []
    for k in ks:
        r = [x for x in rows if x[0] == k]
        out.append(f"{k[0]}{k[1]} N{k[2]}: {len(r)} {S.median(x[1] for x in r):.1f}/{S.median(x[2] for x in r):.2f}/{S.median(x[3] for x in r):.2f}")
    ex = sorted((tl[c, 63].item() - t0) / 1e3 for c in act)
    tot_epi = sum(x[2] for x in rows) / len(act); tot_loop = sum(x[1] for x in rows) / len(act)
    print(f"n6tl {name}: exit med/max {ex[len(ex)//2]:.1f}/{ex[-1]:.1f} per-CTA loop {tot_loop:.1f} epi {tot_epi:.1f} gap med {S.median(gaps) if gaps else 0:.2f} sum/CTA {sum(gaps)/len(act):.1f} | "
          + "; ".join(out), flush=True)


def _diag_n6s3(st):
    import zlib, base64
    raw = zlib.decompress(base64.b64decode(_DIAG_R))
    p = 0
    for load, seed, T in _DIAG_HDR:
        loc = torch.frombuffer(bytearray(raw[p:p + T * 8]), dtype=torch.int8).view(T, 8).int().cuda()
        p += T * 8
        if T <= TK2_MAX: continue
        gen = torch.Generator(device="cuda").manual_seed(seed)
        ids = torch.where(loc >= 0, loc + st["start"], (st["start"] + st["count"] + 5) % 256).int()
        tw = torch.rand(T, 8, device="cuda", generator=gen)
        data = {"hidden_states": torch.randn(T, H, device="cuda", generator=gen).bfloat16(), "topk_ids": ids,
                "topk_weights": tw / tw.sum(-1, keepdim=True),
                "config": {"bs": T, "d_hidden": H, "local_start": st["start"], "local_count": st["count"]}}
        _n6tl(st, data, f"{load} s{seed}")
    torch.cuda.synchronize()
'''
os.makedirs('/tmp/n6s3/diag', exist_ok=True)
open('/tmp/n6s3/diag/submission.py', 'w').write(s)
print('ok')
