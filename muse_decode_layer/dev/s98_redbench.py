import torch, fcntl, statistics, sys
from torch.utils.cpp_extension import load_inline
src = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
// grid (G, S): CTA (g,s) adds NF floats to out[g*NF .. (g+1)*NF) (S-way overlap across s), v4 or scalar reds
template<int V4>
__global__ void __launch_bounds__(256) redk(float* out, int NF) {
  float* o = out + (size_t)blockIdx.x * NF;
  float v = 1.0f + blockIdx.y;
  if (V4) {
    for (int i = threadIdx.x * 4; i < NF; i += blockDim.x * 4)
      asm volatile("red.global.add.v4.f32 [%0], {%1,%2,%3,%4};" :: "l"(o + i), "f"(v), "f"(v), "f"(v), "f"(v) : "memory");
  } else {
    for (int i = threadIdx.x; i < NF; i += blockDim.x) atomicAdd(o + i, v);
  }
}
void run(torch::Tensor out, int64_t G, int64_t S, int64_t NF, int64_t v4) {
  auto st = at::cuda::getCurrentCUDAStream();
  if (v4) redk<1><<<dim3(G, S), 256, 0, st>>>(out.data_ptr<float>(), NF);
  else redk<0><<<dim3(G, S), 256, 0, st>>>(out.data_ptr<float>(), NF);
}
'''
m = load_inline("redb1", cpp_sources="void run(torch::Tensor out, int64_t G, int64_t S, int64_t NF, int64_t v4);",
                cuda_sources=src, functions=["run"], extra_cuda_cflags=["-O3", "-gencode", "arch=compute_90a,code=sm_90a"])
def t(fn, reps=50):
    g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn(); torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=s): fn()
    torch.cuda.synchronize(); ts = []
    with open("/tmp/okc-gpu.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for _ in range(reps):
            a, b = torch.cuda.Event(True), torch.cuda.Event(True); a.record(); g.replay(); b.record(); torch.cuda.synchronize()
            ts.append(a.elapsed_time(b) * 1000)
    return statistics.median(ts)
for (name, N, G, S) in [("FR T128", 4608, 8, 16), ("pool T128", 4096, 8, 16), ("unp T128", 8192, 16, 8), ("FR T256", 4608, 8, 32)]:
    T = 256 if "256" in name else 128
    NF = N * T // G
    out = torch.zeros(G * NF, device="cuda")
    for v4 in (1, 0):
        us = t(lambda: m.run(out, G, S, NF, v4))
        nf = G * S * NF
        print(f"{name} G{G} S{S} v4={v4}: {nf/1e6:.1f}M floats ({nf*4/1e6:.0f} MB) {us:.1f} us = {nf*4/us/1e6:.2f} TB/s")
