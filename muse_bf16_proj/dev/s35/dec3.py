# s35: DEC3 = 12-bit layout with 3 instructions per BF16 pair (was 4, DEC4): per 8-weight group (8 sm bytes + 1 code
# word, the dec8 unit of every 12-bit kernel), move the sign into the code word and the code's low bit into the byte:
#   byte' = c0 << 7 | mantissa;  word' bits: weight 2P+h -> S, c1, c2, c3 at (15, 8, 9, 10) + 16h + d_P (mod 32),
#   d = (0, -4, -8, +4) for P = 0..3.
# Kernel word P: prmt(bytes, 0x38383838, sel) puts the exponent constant 0x38 into byte 1/3 (sel 0x4140 / 0x4342),
# then ONE lop3: prmt | (sh_P(nw) & 0x87008700) with sh = (nw, nw << 4, nw << 8, rotr(nw, 4)): P0 needs no shift.
# 11 instructions per dec8 instead of 16. Decoded bits identical (checked by check() below on random + edge data).
import torch
D = (0, -4, -8, 4)

def tr3(sm8, nw):
    """sm8 uint8 [G, 8], nw int32 [G] (old DEC4 layout) -> (uint8 [G, 8], int32 [G]) DEC3 layout."""
    b = sm8.long()
    w = nw.long() & 0xFFFFFFFF
    nb = torch.zeros_like(b)
    nwn = torch.zeros_like(w)
    for i in range(8):
        P, h = i // 2, i % 2
        c = (w >> (4 * P + 16 * h)) & 15
        s = (b[:, i] >> 7) & 1
        nb[:, i] = ((c & 1) << 7) | (b[:, i] & 0x7F)
        for bit, pos in ((s, 15), ((c >> 1) & 1, 8), ((c >> 2) & 1, 9), ((c >> 3) & 1, 10)):
            nwn |= bit << ((pos + 16 * h + D[P]) % 32)
    nwn = torch.where(nwn >= 2 ** 31, nwn - 2 ** 32, nwn)
    return nb.to(torch.uint8), nwn.int()

def tr3_packed(P, sbytes):
    """A packed stage tensor [stages, sbytes + sbytes / 2] (gemv19 / gemv14 style: S plane then code plane, both in
    the same lane order) -> DEC3 copy."""
    S = P[:, :sbytes].reshape(-1, 8)
    N = P[:, sbytes:].contiguous().view(torch.int32).reshape(-1)
    assert S.shape[0] == N.shape[0]
    S2, N2 = tr3(S, N)
    return torch.cat([S2.view(P.shape[0], sbytes), N2.view(torch.uint8).view(P.shape[0], -1)], 1).contiguous()

def _dec_old(sm8, nw):
    b = sm8.long(); w = nw.long() & 0xFFFFFFFF; out = []
    for i in range(8):
        P, h = i // 2, i % 2
        c = (w >> (4 * P + 16 * h)) & 15
        out.append(((b[:, i] & 0x80) << 8) | ((c | 0x70) << 7) | (b[:, i] & 0x7F))
    return torch.stack(out, 1)

def _dec_new(sm8, nw):  # emulates the kernel ops
    b = sm8.long(); w = nw.long() & 0xFFFFFFFF; M = 0xFFFFFFFF; out = []
    sh = (w, (w << 4) & M, (w << 8) & M, ((w >> 4) | (w << 28)) & M)
    for P in range(4):
        p = b[:, 2 * P] | (0x38 << 8) | (b[:, 2 * P + 1] << 16) | (0x38 << 24)
        r = p | (sh[P] & 0x87008700)
        out += [r & 0xFFFF, r >> 16]
    return torch.stack(out, 1)

def check(G=1 << 20):
    sm8 = torch.randint(0, 256, (G, 8), dtype=torch.uint8)
    nw = torch.randint(-2 ** 31, 2 ** 31 - 1, (G,), dtype=torch.int64).int()
    sm8[:4] = 0; nw[:4] = 0; sm8[4:8] = 255; nw[4:8] = -1
    a = _dec_old(sm8, nw)
    s2, n2 = tr3(sm8, nw)
    assert torch.equal(a, _dec_new(s2, n2)), "DEC3 mismatch"
    print("dec3 check ok", G)

if __name__ == "__main__":
    check()
