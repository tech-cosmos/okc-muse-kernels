"""proj n5s3: g30n consumer oracle. Run from an r-layer muse_decode_layer/ dir (bench.py, submission.py there):
python oracle.py <g30_oracle.cu>  [env M=16|32, KEYS=pooler,unpooler,qkv_gate, CFG=S,NW,NS per key via defaults, PRED ns]
Times g30n as built by the layer vs the same source with -DG30_ORACLE (no DEC3 decode / escape fix) and a bare 12-bit
byte stream estimate, cold L2, after an optional PDL predecessor."""
import os, sys
sys.path.insert(0, ".")
import torch, bench, hashlib
from torch.utils.cpp_extension import load_inline
M = int(os.environ.get("M", 32)); PRED = int(os.environ.get("PRED", 3000))
L = bench.load_sub("submission.py")
src = open(sys.argv[1]).read()
CFG = {"pooler": (2, 16, 16), "unpooler": (1, 16, 16), "qkv_gate": (3, 16, 16)}
def build(c, extra):
    dfs = [f"-DG30_MTOK={c[0]}", f"-DG30_S={c[1]}", f"-DG30_NW={c[2]}", f"-DG30_NS={c[3]}"] + extra
    return load_inline(name="pj_or_" + hashlib.sha256((src + " ".join(dfs)).encode()).hexdigest()[:10],
                       cpp_sources="void g30_gemm(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, int64_t);",
                       cuda_sources=src, functions=["g30_gemm"],
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", *dfs], extra_ldflags=["-lcuda"])
pm = load_inline(name="pjn5_pred", cpp_sources="void pred(torch::Tensor, int64_t);", cuda_sources=r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
__global__ void pk(int ns_) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t0));
    do { asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t1)); } while (t1 - t0 < (unsigned long long)ns_);
}
void pred(torch::Tensor x, int64_t ns_) { pk<<<1, 32, 0, c10::cuda::getCurrentCUDAStream()>>>(ns_); }""",
    functions=["pred"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])
import reference
w_all = reference.generate_weights(reference.WEIGHTS_SEED)
for key in os.environ.get("KEYS", "pooler,unpooler,qkv_gate").split(","):
    w = w_all[key]
    c = (M, *CFG[key])
    code, sm, hdr, scale = L._compress(w)
    wp = L._pack30(code, sm)
    x = torch.randn(M, w.shape[1], device="cuda").bfloat16()
    out = torch.empty(M, w.shape[0], device="cuda", dtype=torch.bfloat16)
    pre = lambda: pm.pred(x, PRED)
    e0 = bench.time_graph(pre)
    res = {}
    for nm, ex in (("g30", []), ("oracle", ["-DG30_ORACLE"])):
        lib = build(c, ex)
        res[nm] = [bench.time_graph(lambda: (pre(), lib.g30_gemm(x, wp, hdr, out, scale, 1))) - e0 for _ in range(3)]
    nb = wp.numel() * wp.element_size()
    print(f"M{M} {key} {tuple(w.shape)} cfg {c} W {nb/1e6:.1f} MB: " + " | ".join(
        f"{k} {min(v):.2f} us ({nb/min(v)/1e6:.2f} TB/s)" for k, v in res.items()), flush=True)
