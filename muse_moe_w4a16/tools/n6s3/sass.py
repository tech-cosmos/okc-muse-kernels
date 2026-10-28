# n6s3: ptxas spills + SASS instruction count of the big-M kernels for a submission file (same source substitutions as _lib()).
# usage: python tools/n6s3/sass.py <submission.py>
import sys, hashlib, subprocess, importlib.util, os, re, sysconfig
sp = importlib.util.spec_from_file_location("s", sys.argv[1]); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
src = m.CUDA_SRC.replace("#define TK2F_L2H 0 ", f"#define TK2F_L2H {int(m.TK2F_L2H)} ").replace("#define DQW 0 ", f"#define DQW {int(m.DQW)} ")
src = src.replace("#define NORM_EARLY 0 ", f"#define NORM_EARLY {int(m.NORM_EARLY)} ").replace("#define TWM_L2H 0 ", f"#define TWM_L2H {int(m.TWM_L2H)} ")
h = hashlib.sha256(src.encode()).hexdigest()[:12]
d = f"/tmp/n6s3/sass_{h}"; os.makedirs(d, exist_ok=True)
open(f"{d}/k.cu", "w").write("#include <torch/extension.h>\n" + src)
tp = os.path.dirname(__import__("torch").__file__) + "/include"
r = subprocess.run(["/usr/local/cuda-13.0/bin/nvcc", "-O3", "-gencode", "arch=compute_90a,code=sm_90a", "-std=c++20", "--expt-relaxed-constexpr",
                    "-isystem", tp, "-isystem", tp + "/torch/csrc/api/include", "-isystem", sysconfig.get_paths()["include"],
                    "-c", f"{d}/k.cu", "-o", f"{d}/k.o", "-Xptxas", "-v"], capture_output=True, text=True)
lines = r.stderr.splitlines()
for i, l in enumerate(lines):
    if "Compiling entry" in l and re.search(r"moe_(twm|hot|twf)", l):
        print(re.search(r"_Z\d+(moe_\w+?)I", l).group(1), re.search(r"'(\S+)'", l).group(1)[:40], lines[i + 1].strip())
o = subprocess.run(["/usr/local/cuda-13.0/bin/cuobjdump", "-sass", f"{d}/k.o"], capture_output=True, text=True).stdout
n, name = 0, None
for l in o.splitlines():
    if "Function : " in l:
        if name and re.search(r"moe_(twm|hot)", name): print(n, name[:50])
        name = l.split()[-1]; n = 0
    elif re.match(r"\s+/\*[0-9a-f]+\*/", l): n += 1
if name and re.search(r"moe_(twm|hot)", name): print(n, name[:50])
