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
DEAD_SRCS = ("22", "OB8")  # n5s6: g22 (G22_CFG = {}, s22 dead) and ob8 (OB8 False, s86 dead) not embedded: judge 1 MB limit
for _n in ("13", "14", "19", "22", "24", "27", "30", "PR3", "PR1", "PR8", "OB8", "PR32", "BND2"):  # proj's gemv13 / gemv14 (compressed BF16 weights), router3 (hi/lo, T 9..128)
    assert layer.count(f"_G{_n}_SRC = None  # @@G{_n}@@\n") == 1
    if _n in DEAD_SRCS:  # n5s6: left None (size limit); their paths are off and never built
        continue
    _s = (here / (f"g{_n}_src.cu" if _n.isdigit() else f"{_n.lower()}_src.cu")).read_text()
    assert Q3 not in _s
    layer = layer.replace(f"_G{_n}_SRC = None  # @@G{_n}@@\n", f"_G{_n}_SRC = r" + Q3 + _s + Q3 + "\n")
lt = (here / "lt_src.cu").read_text()  # proj's cuBLASLt GEMM with an explicit algo config
assert Q3 not in lt and layer.count("_LT_SRC = None  # @@LT@@\n") == 1
layer = layer.replace("_LT_SRC = None  # @@LT@@\n", "_LT_SRC = r" + Q3 + lt + Q3 + "\n")
out = "# okc: problem=muse_decode_layer\n" + layer.replace("# @@ATTENTION@@\n", "# ---- attention (attn_src.py) ----\n" + attn + "\n# ---- layer ----\n")
assert "@@ATTENTION@@" not in out and "@@MOE@@" not in out and "attn_kernel_fn" in out


def strip_comments(src):
    """n5s6: the judge rejects files > 1,000,000 bytes: drop Python comments (COMMENT tokens only; strings, incl. the
    CUDA sources, stay as they are) and the blank/whitespace lines left behind; the AST must not change."""
    all_toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    inner = set()  # rows inside a multi-line string token (after its first row): left exactly as they are
    for t in all_toks:
        if t.start[0] != t.end[0] and t.type not in (tokenize.NEWLINE, tokenize.NL):
            inner.update(range(t.start[0] + 1, t.end[0] + 1))
    res = tokenize.untokenize([t for t in all_toks if t.type != tokenize.COMMENT])
    lines, keep = res.split("\n"), []
    for i, ln in enumerate(lines):
        keep.append(ln if i + 1 in inner else ln.rstrip() if ln.strip() else None)
    res = "\n".join(l for l in keep if l is not None) + "\n"
    assert ast.dump(ast.parse(res)) == ast.dump(ast.parse(src)), "strip_comments changed the AST"
    return res


HEADER = "# okc: problem=muse_decode_layer\n"
assert out.startswith(HEADER)
out = HEADER + strip_comments(out[len(HEADER):])


_CUDA_MARK = re.compile(r"__global__|__device__|#include")


def strip_cuda_comments(src):
    """n27 (session 21): the CUDA sources are triple-quoted Python strings; their // comments were ~90 KB of the 1 MB
    cap. Drop whole-line // comments and trailing // comments inside triple-quoted (non-f) string tokens that hold CUDA
    (__global__/__device__/#include). The sources the compiler sees are only comment-stripped: tools/n27/srccmp.py checks
    this against the previous build (every load_inline source, after the runtime text patches)."""
    lines = src.split("\n")
    offs = [0]
    for ln in lines:
        offs.append(offs[-1] + len(ln) + 1)
    toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    frags = set()  # comment fragments named by other string literals (runtime text-patch anchors): those comments stay
    for t in toks:
        if t.type == tokenize.STRING and "//" in t.string and not _CUDA_MARK.search(t.string):
            try:
                v = ast.literal_eval(t.string)
            except Exception:
                v = t.string
            if isinstance(v, str):
                frags.update(f.split("\n")[0].strip() for f in v.split("//")[1:])
    frags.discard("")
    edits = []
    for t in toks:
        if t.type != tokenize.STRING:
            continue
        body = t.string
        q = body.lstrip("rRbBuU")[:3]
        if q not in ('"""', "'''") or not _CUDA_MARK.search(body):
            continue
        out_lines = []
        for k, ln in enumerate(body.split("\n")):
            i, inq = 0, False  # first // outside a "..." C string literal
            while i < len(ln) and (inq or ln[i:i + 2] != "//"):
                inq ^= ln[i] == '"' and (i == 0 or ln[i - 1] != "\\")
                i += 1
            i = -1 if i >= len(ln) else i
            if k == 0 or i < 0 or any(ln[i + 2:].strip().startswith(f) or f.startswith(ln[i + 2:].strip()) for f in frags):
                out_lines.append(ln)
            elif ln[:i].strip() == "":
                continue
            else:
                assert not ln.rstrip().endswith("\\"), ln
                out_lines.append(ln[:i].rstrip())
        a = offs[t.start[0] - 1] + t.start[1]
        b = offs[t.end[0] - 1] + t.end[1]
        assert src[a:b] == body
        edits.append((a, b, "\n".join(out_lines)))
    for a, b, rep in reversed(edits):
        src = src[:a] + rep + src[b:]
    return src


out = strip_cuda_comments(out)
assert len(out.encode()) < 1_000_000, f"submission.py is {len(out.encode())} bytes (judge limit 1,000,000)"
(here / (sys.argv[1] if len(sys.argv) > 1 else "submission.py")).write_text(out)
import ast as _ast
_ast.parse(out)  # the built file must at least parse
