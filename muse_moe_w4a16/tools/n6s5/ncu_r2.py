"""n6s5: n5s17 ncu_r.py + twm variants e0 (pre-n6s3 twm: EPI 0, HOTC 1, KSC 1, REL 0) / e1 (main). usage: VAR=e0 python ncu_r2.py "verify 32" 7"""
import os, sys, zlib, base64, torch
sys.path.insert(0, os.path.expanduser("~/r-moe/muse_moe_w4a16"))
import reference, submission as S
VARS = {"sq": {}, "r1": {"TK2R_M": {16: 1, 32: 1}}, "r2": {"TK2R_M": {16: 2, 32: 2}}, **{f"p{k}": {"TK2R_M": {16: k, 32: k}} for k in range(3, 14)}, "q16": {"TK2R_M": {16: -6, 32: -6}}, "q31": {"TK2R_M": {16: -5, 32: -5}}, "q10": {"TK2R_M": {16: -4, 32: -4}}, "z3": {"TK2XP_M": {16: (3, 0), 32: (3, 0)}}, "w3": {"TK2XP_M": {16: (3, 1), 32: (3, 1)}}, "s4": {"TK2H_PF4": 0}, "x3": {"TK2X_M": {16: 3, 32: 3}}, "xn": {"TK2X_M": {16: 10, 32: 10}}, "y3": {"TK2X2_M": {16: 3, 32: 3}},
        "h3": {"TK2H_M": {16: 3, 32: 3}}, "e0": {"TWM_EPI": 0, "TWM_HOTC_FORCE": 1, "TWM_KSC_FORCE": 1, "TWM_REL": 0}, "e1": {}, "k1": {"TWF_SPLIT_M": {}, "TWM_KSC_FORCE": 1}, "k0": {"TWF_SPLIT_M": {}, "TWM_KSC_FORCE": 0}}
load, seed = sys.argv[1], int(sys.argv[2])
w = reference.generate_weights(reference.WEIGHTS_SEED)
S.DIAG = 1  # DIAG build (all variants compiled); prepare() then runs _diag: skip it
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
