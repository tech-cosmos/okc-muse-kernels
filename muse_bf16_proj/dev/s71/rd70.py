import torch, fcntl, sys
from torch.utils.cpp_extension import load_inline
src = r'''
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
__global__ void __launch_bounds__(1024) rd(const uint4* p, long n, unsigned* o) {
  unsigned a = 0;
  #pragma unroll 4
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) { uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p + i)); a ^= v.x ^ v.w; }
  if (a == 0x12345) o[0] = a;
}
void rdl(torch::Tensor p, torch::Tensor o, int64_t g) { rd<<<g, 1024, 0, c10::cuda::getCurrentCUDAStream()>>>((const uint4*)p.data_ptr(), p.numel() / 16, (unsigned*)o.data_ptr()); }
'''
lib = load_inline("rd70", cpp_sources="void rdl(torch::Tensor, torch::Tensor, int64_t);", cuda_sources=src, functions=["rdl"], extra_cuda_cflags=["-arch=sm_90a"])
o = torch.zeros(4, dtype=torch.int32, device="cuda"); flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
def tg(fn, reps=40):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s): fn(); fn()
    torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    ts = []
    for _ in range(reps):
        flush.zero_(); torch.cuda.synchronize(); e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        with open("/tmp/okc-gpu.lock", "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX); e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3)
    ts.sort(); return ts[len(ts) // 2]
e = tg(lambda: None)
for mb in (28.4, 50.3):
    b = torch.empty(int(mb * 1e6) // 16 * 16, dtype=torch.uint8, device="cuda")
    for g in (132, 264):
        t = tg(lambda: lib.rdl(b, o, g)) - e
        print(f"read {mb} MB grid {g}x1024: {t:.2f} us = {mb / t:.2f} TB/s (empty graph {e:.2f})")
