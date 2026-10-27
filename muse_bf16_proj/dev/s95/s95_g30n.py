"""proj s95: g30 at T=32/16 (G30_MTOK) on the reference weights: exactness vs FP32 ref + cold-L2 timing vs cuBLAS (x2)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F
import bench, reference
M = int(os.environ.get("M", 32))
L = bench.load_sub("submission.py")
L._G30_SRC = "".join(f"#define {d}\n" for d in (f"G30_MTOK {M}," + os.environ.get("DEFS", "")).split(",") if d) + \
    open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "g30_src.cu")).read()
L._g30_build()
lib = L._G30["lib"]
from torch.utils.cpp_extension import load_inline
PRED = int(os.environ.get("PRED", -1))
pm = load_inline(name="pj95_pred", cpp_sources="void pred(torch::Tensor, torch::Tensor, int64_t);", cuda_sources=r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
__global__ void pk(const uint4* __restrict__ src, uint4* __restrict__ dst, int n8, int ns_) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n8; i += gridDim.x * blockDim.x) dst[i] = src[i];
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t0));
    do { asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t1)); } while (t1 - t0 < (unsigned long long)ns_);
}
void pred(torch::Tensor src, torch::Tensor dst, int64_t ns_) {
    pk<<<32, 256, 0, c10::cuda::getCurrentCUDAStream()>>>((const uint4*)src.data_ptr(), (uint4*)dst.data_ptr(), src.numel() * 2 / 16, ns_);
}""", functions=["pred"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])
W = reference.generate_weights(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
for key in os.environ.get("KEYS", "unpool,pooler").split(","):
    w = W["unpooler"] if key == "unpool" else W[key]
    code, sm, hdr, scale = L._compress(w)
    nesc = int((hdr[:, :, 1] != 2.0 ** -15).sum())
    wp = L._pack30(code, sm)
    x = torch.randn(M, w.shape[1], device="cuda").bfloat16()
    out = torch.empty(M, w.shape[0], device="cuda", dtype=torch.bfloat16)
    lib.g30_gemm(x, wp, hdr, out, scale, 0)
    ref32 = x.float() @ w.float().t()
    ref = F.linear(x, w)
    e = ((out.float() - ref32).norm() / ref32.norm()).item()
    ec = ((ref.float() - ref32).norm() / ref32.norm()).item()
    mx = (out.float() - ref32).abs().max().item(); mc = (ref.float() - ref32).abs().max().item()
    xsrc = x.clone()
    pre = (lambda: pm.pred(xsrc, x, PRED)) if PRED >= 0 else (lambda: None)
    e0 = bench.time_graph(pre) if PRED >= 0 else bench.empty_cost()
    pdl = int(os.environ.get("PDL", 1 if PRED >= 0 else 0))
    ts = []
    for _ in range(2):
        ts.append((bench.time_graph(lambda: (pre(), lib.g30_gemm(x, wp, hdr, out, scale, pdl))) - e0,
                   bench.time_graph(lambda: (pre(), F.linear(x, w))) - e0))
    print(f"M{M} {key} {tuple(w.shape)} esc {nesc}: g30 err {e:.2e} max {mx:.2e} | cuBLAS err {ec:.2e} max {mc:.2e} | "
          + " ; ".join(f"g30 {a:.2f} cuBLAS {b:.2f}" for a, b in ts) + " us", flush=True)
