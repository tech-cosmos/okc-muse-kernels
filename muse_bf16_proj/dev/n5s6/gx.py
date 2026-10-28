"""proj n5s6: G30_GX escape records for dev/g30_src.cu (-DG30_GX=1), built in prepare() from the 12-bit header.
Record per (64-row tile g, K split y) = int32 [rlen]: [0] ncols, [1..4] list start per 16 rows, [5] list end,
[8 .. 8 + GXC) escape columns ascending (slot = index), [8 + GXC ..] (slot | row << 16 | last-of-row << 22 | (col & 7) << 28, v bits) in row order."""
import torch


def g30_gx_tables(hdr, K, S, gxc=64, gxr=512):
    """hdr [N][ESC] float32 pairs (col bits, v) as built by _compress; returns int32 [N / 64 * S][rlen] on hdr's
    device or None if a record does not fit (more than gxc distinct escape columns in one CTA: use the non-GX g30)."""
    h = hdr.detach().cpu()
    N, E = h.shape[0], h.shape[1]
    cols = h[:, :, 0].contiguous().view(torch.int32)
    vb = h[:, :, 1].contiguous().view(torch.int32)
    valid = h[:, :, 1] != 2.0 ** -15
    KS = K // 64
    recs = []
    for g in range(N // 64):
        for y in range(S):
            i0, i1 = 2 * (y * (KS // 2) // S), 2 * ((y + 1) * (KS // 2) // S)
            c = cols[g * 64:(g + 1) * 64]
            ok = valid[g * 64:(g + 1) * 64] & (c >= i0 * 64) & (c < i1 * 64)
            rows, es = ok.nonzero(as_tuple=True)  # row-major = header order
            cc = c[rows, es]
            uc = torch.unique(cc)  # sorted
            if len(uc) > gxc:
                return None
            slot = torch.searchsorted(uc, cc)
            starts = [int((rows < 16 * w).sum()) for w in range(4)] + [len(rows)]
            last = torch.ones_like(rows, dtype=torch.bool)
            last[:-1] = rows[1:] != rows[:-1]
            ent = torch.stack([slot | (rows << 16) | (last.int() << 22) | ((cc & 7) << 28), vb[g * 64:(g + 1) * 64][rows, es]], 1).reshape(-1)
            recs.append((uc, starts, ent))
    emax = max(len(r[2]) for r in recs)
    rlen = (8 + gxc + emax + 3) // 4 * 4
    if rlen > gxr:
        return None
    out = torch.zeros(len(recs), rlen, dtype=torch.int32)
    for i, (uc, starts, ent) in enumerate(recs):
        out[i, 0] = len(uc)
        out[i, 1:6] = torch.tensor(starts, dtype=torch.int32)
        out[i, 8:8 + len(uc)] = uc
        out[i, 8 + gxc:8 + gxc + len(ent)] = ent
    return out.to(hdr.device)
