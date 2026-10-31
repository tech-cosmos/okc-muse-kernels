"""fresh x9: cold i-cache cost of straight-line code (H100). One warp per CTA runs N dependent-free FFMA chains twice
(same code via a 2-trip loop); stamps pass 1 and pass 2. Optional background DRAM stream on the other SMs.
python tools/fresh/icache/ic.py"""
import torch, fcntl
from torch.utils.cpp_extension import load_inline
N = int(__import__("os").environ.get("ICN", "512"))
body = "\n".join(f"    a{i%8} = fmaf(a{i%8}, b, {i}.f);" for i in range(N))
src = r'''
#include <cuda_runtime.h>
#include <torch/extension.h>
__device__ __forceinline__ unsigned long long gt() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__global__ void __launch_bounds__(32) ic(float* out, unsigned long long* ts, float b) {
  float a0 = threadIdx.x, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3, a4 = a0 + 4, a5 = a0 + 5, a6 = a0 + 6, a7 = a0 + 7;
  unsigned long long t0 = gt();
#pragma unroll 1
  for (int p = 0; p < 2; ++p) {
''' + body + r'''
    unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) : "f"(a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7));
    if (threadIdx.x == 0) ts[blockIdx.x * 4 + 1 + p] = t;
  }
  if (threadIdx.x == 0) ts[blockIdx.x * 4] = t0;
  out[blockIdx.x * 32 + threadIdx.x] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}
__global__ void stream(const float4* __restrict__ x, size_t n, float* o) {
  float s = 0.f;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) { float4 v = __ldcs(x + i); s += v.x + v.w; }
  if (s == 12345.f) o[0] = s;
}
void run(torch::Tensor out, torch::Tensor ts, int ctas) { ic<<<ctas, 32>>>(out.data_ptr<float>(), (unsigned long long*)ts.data_ptr(), 1.0001f); }
void runs(torch::Tensor x, torch::Tensor o, int g) { stream<<<g, 1024>>>((const float4*)x.data_ptr(), x.numel() / 4, o.data_ptr<float>()); }
'''
m = load_inline(f"fresh_ic{N}", cpp_sources="void run(torch::Tensor, torch::Tensor, int); void runs(torch::Tensor, torch::Tensor, int);",
                cuda_sources=src, functions=["run", "runs"], extra_cuda_cflags=["-O3", "-arch=sm_90a"], verbose=False)
out = torch.zeros(132 * 32, device="cuda"); ts = torch.zeros(132 * 4, dtype=torch.int64, device="cuda")
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
big = torch.empty(1 << 30, dtype=torch.uint8, device="cuda").view(torch.float32)
o = torch.zeros(4, device="cuda")
s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
with open("/tmp/okc-gpu.lock", "w") as lk:
    fcntl.flock(lk, fcntl.LOCK_EX)
    for mode in ("warmL2", "coldL2", "coldL2+stream"):
        r1, r2 = [], []
        for it in range(20):
            m.run(out, ts, 1); torch.cuda.synchronize()  # code into L2 (and that SM)
            if mode != "warmL2":
                flush.add_(1)
            torch.cuda.synchronize()
            if mode.endswith("stream"):
                with torch.cuda.stream(s1):
                    m.runs(big, o, 120)
                with torch.cuda.stream(s2):
                    torch.cuda._sleep(20000)
                    m.run(out, ts, 4)
            else:
                m.run(out, ts, 4)  # 4 CTAs: land on SMs other than the warm-up's (mostly)
            torch.cuda.synchronize()
            t = ts.view(-1, 4)[:4].double()
            r1 += ((t[:, 1] - t[:, 0]) / 1000).tolist(); r2 += ((t[:, 2] - t[:, 1]) / 1000).tolist()
        r1.sort(); r2.sort()
        print(f"N={N} {mode:14s} pass1 med {r1[len(r1)//2]:.2f} us max {r1[-1]:.2f} | pass2 med {r2[len(r2)//2]:.2f} us")
