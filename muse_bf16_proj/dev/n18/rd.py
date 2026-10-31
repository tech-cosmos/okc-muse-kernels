"""n18: bare reader yardstick: read nbytes (grid-stride uint4, 132*k blocks) inside a graph, minus empty graph."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from torch.utils.cpp_extension import load_inline
import bench
src = r'''
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
__global__ void rd(const uint4* __restrict__ p, size_t n, unsigned* o) {
  unsigned a = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) { uint4 v = __ldcs(p + i); a ^= v.x ^ v.y ^ v.z ^ v.w; }
  if (a == 0x12345678) o[0] = a;
}
void rdk(torch::Tensor t, torch::Tensor o, int64_t nb, int64_t blocks) {
  rd<<<blocks, 1024, 0, c10::cuda::getCurrentCUDAStream()>>>((const uint4*)t.data_ptr(), nb / 16, (unsigned*)o.data_ptr());
}
'''
mod = load_inline(name="n18rd", cpp_sources="void rdk(torch::Tensor, torch::Tensor, int64_t, int64_t);", cuda_sources=src, functions=["rdk"], extra_cuda_cflags=["-O3"])
buf = torch.empty(64 << 20, dtype=torch.uint8, device="cuda"); o = torch.zeros(4, dtype=torch.int32, device="cuda")
e = bench.empty_cost()
for mb in [float(a) for a in sys.argv[1:]]:
    nb = int(mb * 1e6) // 16 * 16
    print(f"read {mb} MB: " + " ".join(f"b{b} {bench.time_graph(lambda: mod.rdk(buf, o, nb, b), reps=200) - e:.2f}" for b in (132, 264)), flush=True)
