"""moe n5s8: T=8 pr32 (pooler g30n 12-bit S2 + FP32 router (FP32 h32 x FP32 rows, FMA) + routing in one launch) as a
TUNE[8] option {"pr32": True} replacing pr8 on the layer's pr1 path. Patches a COPY of r-layer muse_decode_layer/layer.py
(main v144, 4b30a76) in place; copy tools/n5s8/pr32_src.cu over that copy's pr32_src.cu first (the T=16/32 launches get
an extra xr argument = h, unused there). python patch_layer_pr8.py <dir>/layer.py"""
import sys

p = sys.argv[1]
s = open(p).read()


def rep(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (old[:80], s.count(old))
    s = s.replace(old, new)


rep('PR32_CFG = {16: (8, 16), 32: (8, 16)}  # T: (NW, NS)',
    'PR8_32 = False  # moe n5s8: T=8 default of TUNE "pr32" (pr32 PR_F32 instead of pr8)\nPR32_CFG = {8: (16, 8), 16: (8, 16), 32: (8, 16)}  # T: (NW, NS); 8 = moe n5s8 PR_F32 (FP32 router, pr1 path, TUNE pr32)')
rep('dfs = [f"-DG30_MTOK={c[0]}", "-DG30_S=2", f"-DG30_NW={c[1]}", f"-DG30_NS={c[2]}"]',
    'dfs = [f"-DG30_MTOK={c[0]}", "-DG30_S=2", f"-DG30_NW={c[1]}", f"-DG30_NS={c[2]}"] + (["-DPR_F32=1"] if c[0] == 8 else [])')
rep('cpp_sources="void pr32_launch(torch::Tensor x, torch::Tensor wp,',
    'cpp_sources="void pr32_launch(torch::Tensor x, torch::Tensor xr, torch::Tensor wp,')
rep('    return [(lambda c=(m, *PR32_CFG[m]): _pr32_build(c)) for m in PR32_TOKENS]',
    '    return [(lambda c=(m, *PR32_CFG[m]): _pr32_build(c)) for m in (8, *PR32_TOKENS)]')
rep('''        state["pr32"][m] = (lib, e[1], rw)
''', '''        state["pr32"][m] = (lib, e[1], rw)
    # moe n5s8: T=8 = FP32 router rows [4 G][256 KS][64 rows][8 chunks of 4 k], chunk c of row j at c ^ (j & 7)
    e = state.get("g30n", {}).get(("pooler", 16)) or state.get("g30n", {}).get(("pooler", 32))
    lib = _pr32_build((8, *PR32_CFG[8])) if e is not None else None
    if lib is not None:
        t = state["router"].float().contiguous().view(4, 64, 256, 8, 4).permute(0, 2, 1, 3, 4).contiguous()
        j = torch.arange(64, device=t.device).view(64, 1)
        c = torch.arange(8, device=t.device).view(1, 8)
        rf = torch.empty_like(t)
        rf[:, :, j, c ^ (j & 7)] = t[:, :, j, c]
        state["pr32"][8] = (lib, e[1], rf.contiguous().view(torch.uint8))
''')
rep('        lib.pr32_launch(h, wp, hdr, pooled, scale, rw, part, logits,',
    '        lib.pr32_launch(h, h, wp, hdr, pooled, scale, rw, part, logits,')
rep('''        else:
            pk = state["g14"]["pooler"]
            part = torch.empty(8 * tokens * N_EXPERTS, dtype=torch.float32, device=dev)
''', '''        elif state.get("pr32", {}).get(8) is not None and _opt(state, tokens, "pr32", PR8_32):
            # moe n5s8: pooler (g30n 12-bit) + FP32 router (h32) + routing in one launch (pr32 PR_F32), instead of pr8
            lib, (wp, hdr, scale), rw = state["pr32"][8]
            part = torch.empty(64 * tokens * 64, dtype=torch.float32, device=dev)
            logits = torch.empty((tokens, 2 * N_EXPERTS), dtype=torch.float32, device=dev)
            lib.pr32_launch(h, h32, wp, hdr, pooled, scale, rw, part, logits, state["pr32_cnt"], state["router_bias"],
                            data["local_start"], topk_ids, local_ids, topk_weights, ROUTER_MULTIPLIER, ROUTE_EPS,
                            LOCAL_EXPERTS, G30_PDL, part)
        else:
            pk = state["g14"]["pooler"]
            part = torch.empty(8 * tokens * N_EXPERTS, dtype=torch.float32, device=dev)
''')
rep('''            {"g30n:o": True}),  # n5s5''', '''            {"g30n:o": True}, {"pr32": True}, {"pr32": True, "cslpw": True}),  # moe n5s8: pr32 PR_F32 for pr8  # n5s5''')
open(p, "w").write(s)
print("patched", p)
