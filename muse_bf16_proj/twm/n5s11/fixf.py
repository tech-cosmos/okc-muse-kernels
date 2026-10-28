# proj n5s11: per-item fixed cost of moe_twm: whole vs K-half items at low grid (TWM_G env), layer v16/v32 routings
import sys, os, torch, zlib, base64
sys.path.insert(0, '.')
import bench, submission as S
w, st = bench.state()
raw = zlib.decompress(base64.b64decode(S._DIAG_R)); p = 0
for load, seed, T in S._DIAG_HDR:
    loc = torch.frombuffer(bytearray(raw[p:p + T * 8]), dtype=torch.int8).view(T, 8).int().cuda(); p += T * 8
    if T < 128 or seed not in (7, 13): continue
    gen = torch.Generator(device="cuda").manual_seed(seed)
    ids = torch.where(loc >= 0, loc + st["start"], (st["start"] + st["count"] + 5) % 256).int()
    tw = torch.rand(T, 8, device="cuda", generator=gen)
    data = {"hidden_states": torch.randn(T, S.H, device="cuda", generator=gen).bfloat16(), "topk_ids": ids,
            "topk_weights": tw / tw.sum(-1, keepdim=True),
            "config": {"bs": T, "d_hidden": S.H, "local_start": st["start"], "local_count": st["count"]}}
    for v in S.DIAG_TLV:
        S._diag_tl(st, data, f"G{os.environ.get('TWM_G', 'all')} {load} s{seed}", v)
