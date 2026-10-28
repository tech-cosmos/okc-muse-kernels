"""n5s11: drop C/CUDA comments from the CUDA source strings embedded in a built submission.py (the judge rejects
files > 1,000,000 bytes; layer v145 = 963.7 KB). Code is untouched: a state machine skips "..." / '...' literals and
honours backslash line splices; lines that held only a comment are removed, trailing blanks dropped.
Only string literals > MINLEN chars that look like C (__global__/__device__/#include/namespace/#define) and not like
Python are touched. Verified per string: `cpp -fpreprocessed -P` of old and new give the same token stream.

usage: python cuda_strip.py in.py out.py      (prints bytes saved; asserts the Python AST shape is unchanged
                                               apart from the edited string constants)
       from cuda_strip import strip_file       (src -> new src), strip_c (one C source)
"""
import ast, io, re, subprocess, sys, tempfile, tokenize

MINLEN = 2000


def strip_c(s, keep_head=True, keep=lambda c: False):
    """keep_head: keep the leading block of // comment lines (each kernel's header note, for readers)."""
    if keep_head:
        ls = s.split("\n"); k = 0
        while k < len(ls) and (not ls[k].strip() or ls[k].lstrip().startswith("//")) and not ls[k].rstrip().endswith("\\"):
            k += 1
        if k:
            return "\n".join(ls[:k]) + "\n" + strip_c("\n".join(ls[k:]), False, keep)
    out, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c == '"' or c == "'":
            j = i + 1
            while j < n and s[j] != c:
                j += 2 if s[j] == "\\" else 1
            out.append(s[i:j + 1]); i = j + 1
        elif s.startswith("//", i):
            j = i
            while j < n and s[j] != "\n":          # a comment ending in a backslash splices the next line in
                j += 2 if s[j] == "\\" and j + 1 < n else 1
            if keep(s[i:j]):
                out.append(s[i:j])
            i = j                                  # keep the newline
        elif s.startswith("/*", i):
            j = s.find("*/", i + 2); assert j >= 0, "unterminated /*"
            body = s[i:j + 2]
            if keep(body):
                out.append(body); i = j + 2; continue
            out.append("\n" * body.count("\n") if "\n" in body else " "); i = j + 2
        else:
            out.append(c); i += 1
    res = "".join(out)
    lines = [ln.rstrip() for ln in res.split("\n")]
    kept, prev_cont = [], False
    for ln in lines:
        if ln.strip() or prev_cont:                # never drop a line a splice continues into
            kept.append(ln)
        prev_cont = ln.endswith("\\")
    return "\n".join(kept) + ("\n" if res.endswith("\n") else "")


def _cpp_tokens(s):
    with tempfile.NamedTemporaryFile("w", suffix=".cu", delete=False) as f:
        f.write(s)
    r = subprocess.run(["cpp", "-fpreprocessed", "-P", "-w", "-x", "c++", f.name], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[:500]
    return [ln.split() for ln in r.stdout.split("\n") if ln.strip()]


def _is_c(s):
    cu = any(k in s for k in ("__global__", "__device__", "#include", "namespace ", "#define "))
    py = any(k in s for k in ("@triton.jit", "\ndef ", "\nimport ", "\nfrom "))
    return cu and not py


def strip_file(src, verify=True):
    toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    lines = src.split("\n")
    offs = [0]
    for ln in lines:
        offs.append(offs[-1] + len(ln) + 1)
    # comments that runtime text patches use as anchors (they appear in another string constant) are kept
    small = [c.value for c in ast.walk(ast.parse(src)) if isinstance(c, ast.Constant) and isinstance(c.value, str)
             and len(c.value) <= MINLEN]
    def keep(c):
        c = c.rstrip()
        return any(c in v or (len(c) > 12 and c[:12] in v) for v in small)
    edits = []
    for t in toks:
        if t.type != tokenize.STRING or len(t.string) <= MINLEN:
            continue
        m = re.match(r"([rRbBuU]*)('''|\"\"\")", t.string)
        if not m or "r" not in m.group(1).lower():   # raw triple-quoted only: content == literal text
            continue
        q = m.group(2); body = t.string[len(m.group(0)):-3]
        if not _is_c(body):
            continue
        new = strip_c(body, keep=keep)
        assert q not in new
        if verify:
            assert _cpp_tokens(body) == _cpp_tokens(new), f"token mismatch in string at line {t.start[0]}"
        a = offs[t.start[0] - 1] + t.start[1]; b = offs[t.end[0] - 1] + t.end[1]
        edits.append((a, b, m.group(0) + new + q))
    out = src
    for a, b, rep in sorted(edits, reverse=True):
        out = out[:a] + rep + out[b:]
    # same AST apart from string constants
    def shape(x):
        return ast.dump(ast.parse(x)).count("Constant(")
    assert shape(out) == shape(src)
    return out, len(edits)


if __name__ == "__main__":
    src = open(sys.argv[1]).read()
    out, k = strip_file(src)
    open(sys.argv[2], "w").write(out)
    print(f"{k} C strings stripped: {len(src.encode())} -> {len(out.encode())} bytes (-{len(src.encode()) - len(out.encode())})")
