import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../muse_decode_layer'))
import bench, torch
layer = bench.load_sub(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../muse_decode_layer/submission.py'))

def pack30(code, sm):
    """[N, K] codes / sign|mantissa bytes -> [N/64][K/64][6144 B] stages: S bytes [2][128 thr][16], words [128][4]."""
    n, k = code.shape
    G, KS = n // 64, k // 64
    # row = 16 w + 8 rr + gq ; k = 16 j + 8 kh + 2 q + h ; element e = (kh, rr, h) ; thread = (w, gq, q)
    perm = lambda t: t.view(G, 4, 2, 8, KS, 4, 2, 4, 2).permute(0, 4, 1, 3, 7, 5, 6, 2, 8).reshape(G, KS, 128, 4, 8)  # noqa: E731
    b3, v = layer._dec3(perm(sm), perm(code))  # [G, KS, 128, 4, 8] uint8, [G, KS, 128, 4] int32
    Sb = b3.view(G, KS, 128, 2, 16).permute(0, 1, 3, 2, 4).reshape(G, KS, 4096)
    Nb = v.contiguous().view(torch.uint8).view(G, KS, 2048)
    return torch.cat([Sb, Nb], 2).contiguous()


