"""x21: oracles on the s100 T=256 g30 form (tools/fresh/x21/tt2o.cu): which resource limits it? H100 cold L2 graph timing.
Variants by env V='name:DEF=1,DEF2=3;...'. Prints us per variant (minus empty cost) and cuBLAS."""
import os, sys, hashlib, fcntl
D = os.path.dirname(os.path.abspath(__file__)); R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(D))))
sys.path.insert(0, os.path.join(R, "muse_decode_layer"))
import torch, torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import bench, reference
L = bench.load_sub(os.path.join(R, "muse_decode_layer", "submission.py"))
src = open(os.path.join(D, "tt2o.cu")).read()
M = int(os.environ.get("M", 256))
W = reference.generate_weights(0)
key = os.environ.get("KEY", "unpool")
w = W["unpooler"] if key == "unpool" else W[key]
code, sm, hdr, scale = L._compress(w)
wp = L._pack30(code, sm)
x = torch.randn(M, w.shape[1], device="cuda").bfloat16()
out = torch.empty(M, w.shape[0], device="cuda", dtype=torch.bfloat16)
ref32 = x.float() @ w.float().t()
e0 = bench.empty_cost()
def tm(fn):
    return bench.time_graph(fn) - e0  # bench.time_graph takes /tmp/okc-gpu.lock itself
print(f"M{M} {key} {tuple(w.shape)} cuBLAS {tm(lambda: F.linear(x, w)):.2f} us", flush=True)
for v in os.environ.get("V", "base:").split(";"):
    name, dfs = v.split(":")
    flags = ["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-std=c++20"] + [f"-D{d}" for d in dfs.split(",") if d]
    lib = load_inline(name="x21tt2_" + hashlib.sha256((src + " ".join(flags)).encode()).hexdigest()[:10], cpp_sources="void g30_gemm(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, int64_t);",
                      cuda_sources=src, functions=["g30_gemm"], extra_cuda_cflags=flags, extra_ldflags=["-lcuda"])
    lib.g30_gemm(x, wp, hdr, out, scale, 0); torch.cuda.synchronize()
    e = ((out.float() - ref32).norm() / ref32.norm()).item()
    print(f"  {name:12s} {dfs:30s} {tm(lambda: lib.g30_gemm(x, wp, hdr, out, scale, 0)):7.2f} us  err {e:.1e}", flush=True)
