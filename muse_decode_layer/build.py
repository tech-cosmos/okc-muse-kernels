"""Build submission.py = attention kernels (attn_src.py, from ~/attn submission.py) + the layer (layer.py).
The judge takes one file, so the attention module is inlined with its kernel_fn renamed attn_kernel_fn."""
import re, sys, pathlib
here = pathlib.Path(__file__).parent
attn = (here / "attn_src.py").read_text()
layer = (here / "layer.py").read_text()
attn = attn.split("\n", 1)[1]  # drop its "# okc: problem=" line
assert attn.count("\ndef kernel_fn(") == 1
attn = attn.replace("\ndef kernel_fn(", "\ndef attn_kernel_fn(")
out = "# okc: problem=muse_decode_layer\n" + layer.replace("# @@ATTENTION@@\n", "# ---- attention (attn_src.py) ----\n" + attn + "\n# ---- layer ----\n")
assert "@@ATTENTION@@" not in out and "attn_kernel_fn" in out
(here / (sys.argv[1] if len(sys.argv) > 1 else "submission.py")).write_text(out)
