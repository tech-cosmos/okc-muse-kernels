"""s35: H200 A/B DIAG copy (NEVER submit): tools/s32/mkdiag.py with its variant table replaced by decode 1 full
qf (query rows loaded before the first copies) x stages, and splits 640. (v2: 3 round-robin repeats of main / qf / splits 640 / both.) Usage: python tools/s35/mkab.py submission.py /tmp/s35ab/submission.py"""
import os, subprocess, sys

here = os.path.dirname(os.path.abspath(__file__))
subprocess.run([sys.executable, os.path.join(here, "..", "s32", "mkdiag.py"), sys.argv[1], sys.argv[2]], check=True)
src = open(sys.argv[2]).read()
i = src.index("_DIAG = {\n")
j = src.index("\n}\n", i) + 3
src = src[:i] + '''_DIAG = {
    ("decode", 1, "full"): [(f"{n}{k}", ov) for k in range(3) for n, ov in
                            (("d1a", {}), ("d1qf", {"qf": True}), ("d1p640", {"splits": 640}),
                             ("d1p640qf", {"splits": 640, "qf": True}))],
}
''' + src[j:]
open(sys.argv[2], "w").write(src)
print("ab written")
