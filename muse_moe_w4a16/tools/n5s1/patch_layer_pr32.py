"""moe n5s1: add the pr32 option (T=16/32 pooler + router + routing in one launch, pr32_src.cu) to a copy of the
layer's submission.py or layer.py: python patch_layer_pr32.py <file> [src path embedded as a string: default = read at
prepare time from this dir]. Option "pr32" (TUNE / PR32_DEFAULT), tokens PR32_TOKENS; needs the g30n pooler packs."""
import sys, os
p = sys.argv[1]
s = open(p).read()
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pr32_src.cu")
glob = f'''
PR32_TOKENS = (16, 32)
PR32_DEFAULT = False  # moe n5s1: one launch for the T=16/32 pooler (g30n S2) + router (hi/lo) + routing
PR32_CFG = {{16: (8, 16), 32: (8, 16)}}  # T: (NW, NS)
_PR32_SRC = open({src_path!r}).read()
_PR32 = {{}}


def _pr32_prepare(state):
    state["pr32"] = {{}}
    if _PR32_SRC is None:
        return
    import hashlib
    from torch.utils.cpp_extension import load_inline
    r = torch.cat([state["router_hi"], state["router_lo"]]).contiguous().view(torch.int16)
    G, KS = r.shape[0] // 64, r.shape[1] // 64
    rw = r.view(G, 4, 2, 8, KS, 4, 2, 4, 2).permute(0, 4, 1, 3, 7, 5, 6, 2, 8).reshape(G, KS, 128, 4, 8)
    rw = rw.permute(0, 1, 3, 2, 4).contiguous().view(torch.uint8)
    state["pr32_cnt"] = torch.zeros(16, dtype=torch.int32, device=r.device)
    for m in PR32_TOKENS:
        e = state.get("g30n", {{}}).get(("pooler", m))
        if e is None:
            continue
        nw, ns = PR32_CFG[m]
        dfs = [f"-DG30_MTOK={{m}}", "-DG30_S=2", f"-DG30_NW={{nw}}", f"-DG30_NS={{ns}}"]
        if (m, nw, ns) not in _PR32:
            _PR32[(m, nw, ns)] = load_inline(
                name="layer_pr32_" + hashlib.sha256((_PR32_SRC + " ".join(dfs)).encode()).hexdigest()[:10],
                cpp_sources="void pr32_launch(torch::Tensor x, torch::Tensor wp, torch::Tensor hdr, torch::Tensor out, double scale, torch::Tensor rw, torch::Tensor part, torch::Tensor logits, torch::Tensor cnt, torch::Tensor bias, torch::Tensor ls, torch::Tensor ids, torch::Tensor lids, torch::Tensor wts, double mult, double reps, int64_t local, int64_t pdl, torch::Tensor tl);",
                cuda_sources=_PR32_SRC, functions=["pr32_launch"],
                extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", *dfs], extra_ldflags=["-lcuda"])
        state["pr32"][m] = (_PR32[(m, nw, ns)], e[1], rw)

'''
anchor = "G30_PDL = 1"
i = s.index(anchor); i = s.index("\n", i) + 1
s = s[:i] + glob + s[i:]
s = s.replace("    _g30n_prepare(state, w, comp, g)\n", "    _g30n_prepare(state, w, comp, g)\n    _pr32_prepare(state)\n", 1)
old = "    else:\n        side_stream.wait_stream(main_stream)\n        with torch.cuda.stream(side_stream):\n            bkt = "
assert s.count(old) == 1
new = '''    elif (pr32 := (tokens in PR32_TOKENS and state.get("pr32", {}).get(tokens) is not None
                   and _opt(state, tokens, "pr32", PR32_DEFAULT))):
        lib, (wp, hdr, scale), rw = state["pr32"][tokens]
        pooled = torch.empty((tokens, D_EXPERT), dtype=torch.bfloat16, device=dev)
        topk_ids = torch.empty((tokens, TOP_K), dtype=torch.int32, device=dev)
        local_ids = torch.empty((tokens, TOP_K), dtype=torch.int32, device=dev)
        topk_weights = torch.empty((tokens, TOP_K), dtype=torch.float32, device=dev)
        part = torch.empty(64 * tokens * 64, dtype=torch.float32, device=dev)
        logits = torch.empty((tokens, 2 * N_EXPERTS), dtype=torch.float32, device=dev)
        lib.pr32_launch(h, wp, hdr, pooled, scale, rw, part, logits, state["pr32_cnt"], state["router_bias"],
                        data["local_start"], topk_ids, local_ids, topk_weights, ROUTER_MULTIPLIER, ROUTE_EPS,
                        LOCAL_EXPERTS, G30_PDL, part)
        bucket = None
''' + old
s = s.replace(old, new, 1)
old2 = "    if pr1:\n        pass\n    elif tokens in POOLER_HI_PRIORITY:"
assert s.count(old2) == 1
s = s.replace(old2, "    if pr1 or pr32:\n        pass\n    elif tokens in POOLER_HI_PRIORITY:", 1)
# pr32 must exist on the pr1 path too
old3 = "    if pr1:  # pooler + router + routing in one launch (proj pr1)"
assert s.count(old3) == 1
s = s.replace(old3, "    pr32 = False\n" + old3, 1)
open(p, "w").write(s)
print("patched", p)
