"""s33: H200 A/B DIAG copy (NEVER submit): tools/s32/mkdiag.py with its variant table replaced by spec on/off at
verify 1 full (each twice, round-robin, cold). Usage: python tools/s33/mkab.py submission.py /tmp/s33ab/submission.py"""
import os, re, subprocess, sys

here = os.path.dirname(os.path.abspath(__file__))
subprocess.run([sys.executable, os.path.join(here, "..", "s32", "mkdiag.py"), sys.argv[1], sys.argv[2]], check=True)
src = open(sys.argv[2]).read()
i = src.index("_DIAG = {\n")
j = src.index("\n}\n", i) + 3
src = src[:i] + '''_DIAG = {
    ("verify", 1, "full"): [("v1spec", {}), ("v1nospec", {"spec": False}), ("v1spec2", {}),
                            ("v1nospec2", {"spec": False})],
    ("decode", 1, "full"): [("d1a", {}), ("d1nospec", {"spec": False}), ("d1a2", {}), ("d1nospec2", {"spec": False}),
                            ("d1a3", {}), ("d1nospec3", {"spec": False})],
}
''' + src[j:]
open(sys.argv[2], "w").write(src)
print("ab written")
