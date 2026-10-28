"""n5s16: one tk2f call on a layer routing (DIAG table) under ncu. usage: VAR=sq|x3|xn|y3 python ncu_x.py "decode 16" 99"""
import os, sys, zlib, base64, torch
sys.path.insert(0, "/tmp/n5s26")
import reference, submission as S
VARS = {"sq": {}, "r1": {"TK2R_M": {16: 1, 32: 1}}, "r2": {"TK2R_M": {16: 2, 32: 2}}, **{f"p{k}": {"TK2R_M": {16: k, 32: k}} for k in range(3, 14)}, "q16": {"TK2R_M": {16: -6, 32: -6}}, "q31": {"TK2R_M": {16: -5, 32: -5}}, "q10": {"TK2R_M": {16: -4, 32: -4}}, "z3": {"TK2XP_M": {16: (3, 0), 32: (3, 0)}}, "w3": {"TK2XP_M": {16: (3, 1), 32: (3, 1)}}, "s4": {"TK2H_PF4": 0}, "x3": {"TK2X_M": {16: 3, 32: 3}}, "xn": {"TK2X_M": {16: 10, 32: 10}}, "y3": {"TK2X2_M": {16: 3, 32: 3}},
        "h3": {"TK2H_M": {16: 3, 32: 3}}, "dt": {"TK2DT_M": {16: 1, 32: 1, 8: 1}}}
load, seed = sys.argv[1], int(sys.argv[2])
w = reference.generate_weights(reference.WEIGHTS_SEED)
S.TK2F_MAXR = int(os.environ.get("MAXR", "168"))
S.DIAG = 1  # DIAG build (TK2DT compiled in); prepare() then runs _diag: skip it
S._diag = lambda st: None
S._diag = lambda st: None
st = S.prepare(w)
raw = zlib.decompress(base64.b64decode(S._DIAG_R)); p = 0
for l, sd, T in S._DIAG_HDR:
    if l == load and sd == seed: print("found", l, sd, T)
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
print("cfg", S._tk2f_cfg(T, st)); S.kernel_fn(data, st); torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart(); S.kernel_fn(data, st); torch.cuda.synchronize(); torch.cuda.cudart().cudaProfilerStop()
