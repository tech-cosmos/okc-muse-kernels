"""n5s23 prefix oracle (WRONG OUTPUT, timing only; never submit): python tools/prefix_oracle.py <submission.py> <outdir>
Writes base0.py (TUNE = {}) and stop_{q,a,o,p}.py: kernel_fn returns right after qkv (q), attention (a), o GEMM +
first boundary (o), the pre-expert rmsnorm / routing join (p), the experts
(e: _tail returns before the unpool + boundary). Segment cost in-layer = difference of adjacent prefixes."""
import os, sys
L0 = open(sys.argv[1]).read().split("\n")
out = sys.argv[2]; os.makedirs(out, exist_ok=True)
RET = ("return (torch.empty((tokens, HIDDEN), dtype=torch.bfloat16, device=dev), "
       "torch.empty((tokens, HIDDEN), dtype=torch.float32, device=dev), torch.empty((tokens, TOP_K), dtype=torch.int32, device=dev))")


def find(L, s, start=0):
    i = next(i for i in range(start, len(L)) if L[i].startswith(s)); return i


def tune_off(L):
    L.insert(find(L, "TUNE_BASE = {"), "TUNE = {}  # n5s23 oracle: defaults only")


def stop(L, tag):
    k = find(L, "def _kernel_fn(data, state):")
    if tag == "q":
        i = find(L, "    o = attn_kernel_fn({", k); L.insert(i, "    " + RET)
    elif tag == "a":
        i = find(L, '        "seq_lens": data["seq_lens"], "config": cfg, "qkv_ss"', k); L.insert(i + 1, "    " + RET)
    elif tag == "o":
        i = find(L, '    main_stream, side_stream = torch.cuda.current_stream(), state["side_stream"]', k); L.insert(i, "    " + RET)
    elif tag == "e":  # before the unpool + second boundary (T=1/8: n2u/nu8 carry the unpool: only the boundary goes)
        i = find(L, "def _tail(moe, stream1, h, tokens, state, pdl, f=None, sc=None):"); L.insert(i + 2, "    return torch.empty_like(h), torch.empty_like(stream1)")
    elif tag == "p":
        i = find(L, "        moe = _experts_moe(p, wts, lids, state, bucket, gnorm) if _use_moe", k); L.insert(i, "        " + RET)
        j = find(L, "    if not pr1:", i)
        assert "main_stream.wait_stream(side_stream)" in L[j + 1], L[j + 1]
        L.insert(j + 2, "    " + RET)


for name, tag in (("base0", None), ("stop_q", "q"), ("stop_a", "a"), ("stop_o", "o"), ("stop_p", "p"), ("stop_e", "e")):
    L = list(L0); tune_off(L)
    if tag: stop(L, tag)
    open(f"{out}/{name}.py", "w").write("\n".join(L)); print(name, "ok")
