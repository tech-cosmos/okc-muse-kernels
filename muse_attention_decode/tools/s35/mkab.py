"""s35: H200 A/B DIAG copy (NEVER submit): tools/s32/mkdiag.py with its variant table replaced by decode 1 full
qf (query rows loaded before the first copies) x stages, and splits 640. Usage: python tools/s35/mkab.py submission.py /tmp/s35ab/submission.py"""
import os, subprocess, sys

here = os.path.dirname(os.path.abspath(__file__))
subprocess.run([sys.executable, os.path.join(here, "..", "s32", "mkdiag.py"), sys.argv[1], sys.argv[2]], check=True)
src = open(sys.argv[2]).read()
i = src.index("_DIAG = {\n")
j = src.index("\n}\n", i) + 3
src = src[:i] + '''_DIAG = {
    ("decode", 1, "full"): [("d1a", {}), ("d1qf", {"qf": True}), ("d1qf3", {"qf": True, "stages": 3}),
                            ("d1qf4", {"qf": True, "stages": 4}), ("d1s3", {"stages": 3}), ("d1p640", {"splits": 640}),
                            ("d1p640qf", {"splits": 640, "qf": True}), ("d1a2", {}), ("d1qf2", {"qf": True}),
                            ("d1qf3b", {"qf": True, "stages": 3})],
}
''' + src[j:]
open(sys.argv[2], "w").write(src)
print("ab written")
