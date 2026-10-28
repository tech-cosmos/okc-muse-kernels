"""n5s11: compare the SASS of every load_inline build of two submission files: build each with its own
TORCH_EXTENSIONS_DIR (A, B), then: python sass_eq.py A_dir B_dir. Pairs .so files by their set of kernel names
(hashed dir names differ when only comments differ) and prints SAME/DIFF per pair (addresses/encodings stripped)."""
import glob, hashlib, os, re, subprocess, sys
CUOBJ = "/usr/local/cuda-13.0/bin/cuobjdump"
def sass(so):
    t = subprocess.run([CUOBJ, "-sass", so], capture_output=True, text=True).stdout
    t = re.sub(r"/\*[0-9a-f]{4}\*/", "", t); t = re.sub(r" +/\* 0x[0-9a-f]+ \*/", "", t)
    names = tuple(sorted(re.findall(r"Function : (\S+)", t)))
    return names, hashlib.md5(t.encode()).hexdigest()
def load(d):
    out = {}
    for so in glob.glob(os.path.join(d, "*", "*.so")):
        n, h = sass(so); out.setdefault(n, []).append((h, so))
    return out
A, B = load(sys.argv[1]), load(sys.argv[2])
bad = 0
for n, lb in sorted(B.items()):
    la = A.get(n, [])
    ha = sorted(h for h, _ in la); hb = sorted(h for h, _ in lb)
    ok = ha == hb and len(n) > 0
    bad += not ok
    print("SAME" if ok else "DIFF", len(lb), os.path.basename(lb[0][1]), f"{len(n)} kernels")
print("A-only:", [os.path.basename(v[0][1]) for k, v in A.items() if k not in B])
print("pairs differing:", bad)
