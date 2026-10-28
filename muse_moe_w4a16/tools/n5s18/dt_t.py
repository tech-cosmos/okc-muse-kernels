"""n5s18: deferred-TWO tk2f (TK2DT_M) vs the default sq: bitwise on the DIAG layer routings + synthetic hot experts,
and vs reference.py. usage: python dt_t.py [reps]"""
import os, sys, zlib, base64, torch
sys.path.insert(0, os.path.expanduser("~/r-moe/muse_moe_w4a16"))
import reference, submission as S
reps = int(sys.argv[1]) if len(sys.argv) > 1 else 3
w = reference.generate_weights(reference.WEIGHTS_SEED)
S.DIAG = 1; S._diag = lambda st: None  # TK2DT is compiled only in DIAG builds
st = S.prepare(w)
raw = zlib.decompress(base64.b64decode(S._DIAG_R)); p = 0
cases = []
for l, sd, T in S._DIAG_HDR:
    loc = torch.frombuffer(bytearray(raw[p:p + T * 8]), dtype=torch.int8).view(T, 8).int().cuda(); p += T * 8
    if T <= 32: cases.append((f"{l} s{sd}", sd, T, loc))
g = torch.Generator().manual_seed(5)
for T, hot in ((16, [16]), (32, [32, 17, 9]), (32, [24, 24, 16]), (16, [12, 9, 9]), (8, [8, 8]), (32, [32, 32, 32, 32, 25])):
    loc = torch.full((T, 8), -1, dtype=torch.int32)
    for j, c in enumerate(hot): loc[:c, j] = j  # expert j on tokens 0..c-1, slot j
    for t in range(T):
        for k in range(len(hot), 8):
            loc[t, k] = int(torch.randint(len(hot), st["count"], (1,), generator=g)) if torch.rand(1, generator=g) < 0.6 else -1
            while (loc[t, :k] == loc[t, k]).any() and loc[t, k] >= 0: loc[t, k] = int(torch.randint(len(hot), st["count"], (1,), generator=g))
    cases.append((f"syn T{T} {hot}", 1, T, loc.cuda()))
bad = 0
for name, sd, T, loc in cases:
    gen = torch.Generator(device="cuda").manual_seed(sd)
    ids = torch.where(loc >= 0, loc + st["start"], (st["start"] + st["count"] + 5) % 256).int()
    tw = torch.rand(T, 8, device="cuda", generator=gen)
    data = {"hidden_states": torch.randn(T, S.H, device="cuda", generator=gen).bfloat16(), "topk_ids": ids,
            "topk_weights": tw / tw.sum(-1, keepdim=True),
            "config": {"bs": T, "d_hidden": S.H, "local_start": st["start"], "local_count": st["count"]}}
    S.TK2DT_M = {}
    o0 = S.kernel_fn(data, st).clone()
    S.TK2DT_M = {T: 1}
    c = S._tk2f_cfg(T, st)
    md = 0.0
    for _ in range(reps):
        o1 = S.kernel_fn(data, st); torch.cuda.synchronize()
        md = max(md, (o1.float() - o0.float()).abs().max().item())
    S.TK2DT_M = {}
    chk = reference.check_implementation(data, o1, w)
    cnt = torch.bincount(loc[loc >= 0].flatten().long()).max().item()
    print(f"{name:28s} cfg {c} maxtok {cnt:2d} maxdiff_vs_sq {md:.3g} ref {chk}", flush=True)
    bad += md != 0
print("BAD", bad)
