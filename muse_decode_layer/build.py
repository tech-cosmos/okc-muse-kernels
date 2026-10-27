"""Build submission.py = attention kernels (attn_src.py, from ~/attn submission.py) + the layer (layer.py).
The judge takes one file, so the attention module is inlined with its kernel_fn renamed attn_kernel_fn."""
import re, sys, pathlib
import ast, io, tokenize


def moe_namespaced(src):
    """The moe agent's submission with every top-level Python name prefixed _moe_ (tokens only: strings,
    including the CUDA source, and attributes stay as they are)."""
    src = src.split("\n", 1)[1]
    tree = ast.parse(src)
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    out, prev = [], None
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.NAME and tok.string in names and not (prev and prev.string == "."):
            tok = tok._replace(string="_moe_" + tok.string)
        out.append(tok)
        if tok.type not in (tokenize.NL, tokenize.COMMENT):
            prev = tok
    return tokenize.untokenize(out)


here = pathlib.Path(__file__).parent
attn = (here / "attn_src.py").read_text()
layer = (here / "layer.py").read_text()
attn = attn.split("\n", 1)[1]  # drop its "# okc: problem=" line
assert attn.count("\ndef kernel_fn(") == 1
attn = attn.replace("\ndef kernel_fn(", "\ndef attn_kernel_fn(")
_a = "from torch.utils.cpp_extension import load_inline as _load_inline\n"
assert attn.count(_a) == 1
attn = attn.replace(_a, "def _load_inline(**kw):  # s68: built on layer's _BUILD_POOL, resolved in prepare (_join_builds)\n"
                        "    from torch.utils.cpp_extension import load_inline\n"
                        "    return _LazyBuild(_BUILD_POOL.submit(load_inline, **kw), kw['name'])\n")
# In the judge image SGLang's import leaves triton.next_power_of_2 a plain function, which a @triton.jit body
# can't call (check 7e3e63a1): use our own constexpr function inside kernels.
jit_np2 = "PF_N: tl.constexpr = triton.next_power_of_2("
assert attn.count(jit_np2) == 1
attn = attn.replace(jit_np2, "PF_N: tl.constexpr = _np2_jit(")
attn = attn.replace("\nclass Config(TypedDict):", """
from triton.runtime.jit import constexpr_function as _constexpr_function


@_constexpr_function
def _np2_jit(n):
    return 1 << max(0, int(n) - 1).bit_length()


class Config(TypedDict):""", 1)
# csl::decode (decode 1 sliding) triggers its dependents at start: the o GEMV (a PDL secondary) issues its first
# weight loads during attention.
csl_anchor = "    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3;\n    const int h = (blockIdx.x >> 4) & 1, wk = warp;\n"
assert attn.count(csl_anchor) == 2  # csl and its verify twin (no PDL dependent there)
attn = attn.replace(csl_anchor, csl_anchor + '    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n')
moe = moe_namespaced((here / "moe_src.py").read_text())
# s37: GNORM reads the pooler rows strided (fused pooler+router GEMM output): no contiguous() copy when gather_norm is set
_a = '    x = data["hidden_states"].contiguous()\n'
assert moe.count(_a) == 1
moe = moe.replace(_a, '    x = data["hidden_states"] if data.get("gather_norm") is not None else data["hidden_states"].contiguous()\n')
layer = layer.replace("# @@MOE@@\n", "# ---- experts (moe_src.py, names prefixed _moe_) ----\n" + moe + "\n# ---- layer ----\n")
g10 = (here / "g10_src.cu").read_text()  # proj's gemv10 CUDA source, as a raw string literal in the layer
Q3 = "'" * 3
assert Q3 not in g10 and layer.count("_G10_SRC = None  # @@G10@@\n") == 1
layer = layer.replace("_G10_SRC = None  # @@G10@@\n", "_G10_SRC = r" + Q3 + g10 + Q3 + "\n")
g12 = (here / "g12_src.cu").read_text()  # proj's gemv12 (m = 8)
assert Q3 not in g12 and layer.count("_G12_SRC = None  # @@G12@@\n") == 1
layer = layer.replace("_G12_SRC = None  # @@G12@@\n", "_G12_SRC = r" + Q3 + g12 + Q3 + "\n")
for _n in ("13", "14", "19", "22", "24", "27", "PR3", "PR1", "PR8"):  # proj's gemv13 / gemv14 (compressed BF16 weights), router3 (hi/lo, T 9..128)
    _s = (here / (f"g{_n}_src.cu" if _n.isdigit() else f"{_n.lower()}_src.cu")).read_text()
    assert Q3 not in _s and layer.count(f"_G{_n}_SRC = None  # @@G{_n}@@\n") == 1
    layer = layer.replace(f"_G{_n}_SRC = None  # @@G{_n}@@\n", f"_G{_n}_SRC = r" + Q3 + _s + Q3 + "\n")
lt = (here / "lt_src.cu").read_text()  # proj's cuBLASLt GEMM with an explicit algo config
assert Q3 not in lt and layer.count("_LT_SRC = None  # @@LT@@\n") == 1
layer = layer.replace("_LT_SRC = None  # @@LT@@\n", "_LT_SRC = r" + Q3 + lt + Q3 + "\n")
out = "# okc: problem=muse_decode_layer\n" + layer.replace("# @@ATTENTION@@\n", "# ---- attention (attn_src.py) ----\n" + attn + "\n# ---- layer ----\n")
assert "@@ATTENTION@@" not in out and "@@MOE@@" not in out and "attn_kernel_fn" in out
(here / (sys.argv[1] if len(sys.argv) > 1 else "submission.py")).write_text(out)
import ast as _ast
_ast.parse(out)  # the built file must at least parse
