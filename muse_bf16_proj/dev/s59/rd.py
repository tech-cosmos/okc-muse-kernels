# s59: bare first-op reader (ld.global.v4 grid-stride) timed like bench.time_graph: python dev/s59/rd.py
import sys, torch
sys.path.insert(0, "/home/daytona/r-proj/muse_bf16_proj")
import bench
from torch.utils.cpp_extension import load_inline
src = r'''
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
__global__ void __launch_bounds__(1024,1) rd(const uint4* __restrict__ p, long n, unsigned* o) {
  unsigned a = 0;
  for (long i = blockIdx.x * 1024L + threadIdx.x; i < n; i += gridDim.x * 1024L) { uint4 v = __ldcs(p + i); a ^= v.x ^ v.y ^ v.z ^ v.w; }
  if (a == 0x12345678u) o[0] = a;
}
void rdk(torch::Tensor w, torch::Tensor o, int64_t g) { rd<<<g, 1024, 0, c10::cuda::getCurrentCUDAStream()>>>((const uint4*)w.data_ptr(), w.numel() / 16, (unsigned*)o.data_ptr()); }
'''
m = load_inline(name="pj59rd", cpp_sources="void rdk(torch::Tensor, torch::Tensor, int64_t);", cuda_sources=src, functions=["rdk"], extra_cuda_cflags=["-O3", "-arch=sm_90a"])
o = torch.zeros(4, dtype=torch.int32, device="cuda")
e = bench.time_graph(lambda: None)
for mb in [28.3, 37.7, 48]:
    w = torch.empty(int(mb * 1e6) // 16 * 16, dtype=torch.uint8, device="cuda")
    print(mb, "MB", " ".join(f"g{g}:{bench.time_graph(lambda: m.rdk(w, o, g)) - e:.2f}" for g in [132, 264]))
