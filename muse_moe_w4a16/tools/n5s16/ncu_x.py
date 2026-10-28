"""n5s16: one tk2f call on a layer routing (DIAG table) under ncu. usage: VAR=sq|x3|xn|y3 python ncu_x.py "decode 16" 99"""
import os, sys, zlib, base64, torch
sys.path.insert(0, os.path.expanduser("~/r-moe/muse_moe_w4a16"))
import reference, submission as S
VARS = {"sq": {}, "x3": {"TK2X_M": {16: 3, 32: 3}}, "xn": {"TK2X_M": {16: 10, 32: 10}}, "y3": {"TK2X2_M": {16: 3, 32: 3}},
        "h3": {"TK2H_M": {16: 3, 32: 3}}}
load, seed = sys.argv[1], int(sys.argv[2])
w = reference.generate_weights(reference.WEIGHTS_SEED)
S.DIAG = 1  # DIAG build (all variants compiled); prepare() then runs _diag: skip it
S._diag = lambda st: None
st = S.prepare(w)
raw = zlib.decompress(base64.b64decode(S._DIAG_R)); p = 0
for l, sd, T in S._DIAG_HDR:
    if l == load and sd == seed:
        loc = torch.frombuffer(bytearray(raw[p:p + T * 8]), dtype=torch.int8).view(T, 8).int().cuda(); break
    p += T * 8
gen = torch.Generator(device="cuda").manual_seed(seed)
ids = torch.where(loc >= 0, loc + st["start"], (st["start"] + st["count"] + 5) % 256).int()
tw = torch.rand(T, 8, device="cuda", generator=gen)
data = {"hidden_states": torch.randn(T, S.H, device="cuda", generator=gen).bfloat16(), "topk_ids": ids,
        "topk_weights": tw / tw.sum(-1, keepdim=True),
        "config": {"bs": T, "d_hidden": S.H, "local_start": st["start"], "local_count": st["count"]}}
for k, v in VARS[os.environ.get("VAR", "sq")].items(): setattr(S, k, v)
S.kernel_fn(data, st); torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart(); S.kernel_fn(data, st); torch.cuda.synchronize(); torch.cuda.cudart().cudaProfilerStop()
