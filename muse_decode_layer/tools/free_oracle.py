"""n5s26 'free piece' oracle (lead 08:15Z; timing only, never submit): python tools/free_oracle.py <submission.py> <outdir>
Writes base0.py (TUNE = {}) and free_{a,b1,r,b2}.py. In a free_X variant, piece X runs once eagerly (bench.capture's
warm-up call) and its outputs are cloned and kept per (tokens, sliding); every later call (incl. the captured graph)
launches nothing for X and hands the kept outputs on. So the rest of the layer sees the same values (same routing) and
in-layer cost of X = base - free_X, with overlap effects included (unlike prefix_oracle's differences).
  a  = attention (attn_kernel_fn: qkv rope/cache store, split-K, merge)       outputs identical to base
  b1 = first boundary (o -> residual/norm -> h, h32, stream1), where launched  outputs identical to base
  r  = router GEMM + routing (_route side stream at T=16/32/128, _routing at the FR T=256 path)  identical
  b2 = second boundary in _tail (T=8: incl. the nu8 scatter-combine; T=1 n2u has it fused: n/a)  output differs
Pieces fused into bigger kernels (T=1 pr1 / T=8 pr8 router+routing, T=1 bnd in pr1, T=8 ob8) are not removable."""
import os, sys
L0 = open(sys.argv[1]).read().split("\n")
out = sys.argv[2]; os.makedirs(out, exist_ok=True)
HELPER = '''_ORC, _ORC_T = {}, [0]


def _orc(tag, launcher, outs=None):
    def run(*a, **k):
        key = (tag, _ORC_T[0], _SLIDING[0])
        if key not in _ORC:
            r = launcher(*a, **k)
            if outs is not None:
                _ORC[key] = tuple(a[i].clone() for i in outs)
            else:
                _ORC[key] = tuple(t.clone() for t in r) if isinstance(r, tuple) else r.clone()
        return _ORC[key]
    return run
'''


def find(L, s, start=0):
    return next(i for i in range(start, len(L)) if L[i].startswith(s))


def sub(L, s, old, new):
    i = find(L, s); assert old in L[i], (s, L[i]); L[i] = L[i].replace(old, new, 1)


def free(L, tag):
    k = find(L, "def _kernel_fn(data, state):")
    L[k:k] = HELPER.split("\n")
    sub(L, "    _SLIDING[0] = cfg[\"window\"] > 0", "_SLIDING[0] =", "_ORC_T[0] = cfg[\"tokens\"]; _SLIDING[0] =")
    if tag == "a":
        sub(L, "    o = attn_kernel_fn({", "attn_kernel_fn(", '_orc("a", attn_kernel_fn)(')
    elif tag == "b1":
        sub(L, "      _boundary[(tokens * spl + npf,)](a,", "_boundary[(tokens * spl + npf,)](",
            'stream1, h, h32 = _orc("b1", _boundary[(tokens * spl + npf,)], (6, 7, 8))(')
    elif tag == "r":
        sub(L, "            topk_ids, local_ids, topk_weights = _route(state, h, h32,", "_route(", '_orc("r", _route)(')
        sub(L, "        _routing[(tokens,)](pr[:, D_EXPERT:]", "_routing[(tokens,)](",
            'ids, lids, wts = _orc("r", _routing[(tokens,)], (3, 4, 5))(')
    elif tag == "b2":
        sub(L, "    _boundary[(tokens * spl,)](f, stream1,", "_boundary[(tokens * spl,)](", '_orc("b2", _boundary[(tokens * spl,)], ())(')


for name, tag in (("base0", None), ("free_a", "a"), ("free_b1", "b1"), ("free_r", "r"), ("free_b2", "b2")):
    L = list(L0)
    L.insert(find(L, "TUNE_BASE = {"), "TUNE = {}  # n5s26 oracle: defaults only")
    if tag: free(L, tag)
    open(f"{out}/{name}.py", "w").write("\n".join(L)); print(name, "ok")
