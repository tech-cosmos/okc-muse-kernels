"""proj n5s7: G30_GX escape records for g30k (stream-K g30, -DG30K_ON -DG30_GX=1): one record per CTA (G of them).
int32 [rlen]: [0] ncols, [1] nseg, [2 + 5 s .. 2 + 5 s + 5) list starts per 16 rows + end of segment s (s < GXS = 4),
[24 .. 24 + gxc) the CTA's escape columns ascending (slot = index), [24 + gxc ..] uint2 (slot | row << 16 | last-of-row << 22 |
(col & 7) << 28, v bits), segment-major, row order within a segment. Segment s = tile gA + s, k stages [i0, i1) as in g30k."""
import torch

GXH = 24


def g30k_gx_tables(hdr, K, G, gxc=64, gxr=512, gxs=4):
    h = hdr.detach().cpu()
    N = h.shape[0]
    cols = h[:, :, 0].contiguous().view(torch.int32)
    vb = h[:, :, 1].contiguous().view(torch.int32)
    valid = h[:, :, 1] != 2.0 ** -15
    KP = K // 128
    P = N // 64 * KP
    recs = []
    for b in range(G):
        u0, u1 = b * P // G, (b + 1) * P // G
        gA, gB = u0 // KP, (u1 - 1) // KP
        if gB - gA + 1 > gxs:
            return None
        segs, allc = [], []
        for g in range(gA, gB + 1):
            ta = g * KP
            i0, i1 = 2 * (max(u0, ta) - ta), 2 * (min(u1, ta + KP) - ta)
            c = cols[g * 64:(g + 1) * 64]
            ok = valid[g * 64:(g + 1) * 64] & (c >= i0 * 64) & (c < i1 * 64)
            rows, es = ok.nonzero(as_tuple=True)
            segs.append((rows, c[rows, es], vb[g * 64:(g + 1) * 64][rows, es]))
            allc.append(c[rows, es])
        uc = torch.unique(torch.cat(allc))
        if len(uc) > gxc:
            return None
        hdrw, ents, n = [len(uc), len(segs)] + [0] * (GXH - 2), [], 0
        for s, (rows, cc, vv) in enumerate(segs):
            hdrw[2 + 5 * s:2 + 5 * s + 5] = [n + int((rows < 16 * w).sum()) for w in range(4)] + [n + len(rows)]
            slot = torch.searchsorted(uc, cc)
            last = torch.ones_like(rows, dtype=torch.bool)
            last[:-1] = rows[1:] != rows[:-1]
            ents.append(torch.stack([slot | (rows << 16) | (last.int() << 22) | ((cc & 7) << 28), vv], 1).reshape(-1))
            n += len(rows)
        recs.append((uc, hdrw, torch.cat(ents)))
    rlen = (GXH + gxc + max(len(r[2]) for r in recs) + 3) // 4 * 4
    if rlen > gxr:
        return None
    out = torch.zeros(G, rlen, dtype=torch.int32)
    for i, (uc, hw, ent) in enumerate(recs):
        out[i, :GXH] = torch.tensor(hw, dtype=torch.int32)
        out[i, GXH:GXH + len(uc)] = uc
        out[i, GXH + gxc:GXH + gxc + len(ent)] = ent
    return out.to(hdr.device)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/tmp/pjgx/muse_decode_layer")
    import reference, bench
    layer = bench.load_sub("/tmp/pjgx_nt.py")
    w = reference.generate_weights(reference.WEIGHTS_SEED)
    wp = w["pooler"].cuda() if not w["pooler"].is_cuda else w["pooler"]
    code, sm, hdr, scale = layer._compress(wp)
    print("pooler", tuple(wp.shape))
    for G in (104, 120):
        r = g30k_gx_tables(hdr, wp.shape[1], G)
        print(G, None if r is None else (tuple(r.shape), int(r[:, 0].max()), int(r[:, 1].max()), int(r[:, 1].min())))
