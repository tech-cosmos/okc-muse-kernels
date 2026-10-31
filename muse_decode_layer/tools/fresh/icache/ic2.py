"""fresh x9: is the instruction cache that a cold kernel misses per SM or shared (GPC/L2)? H100.
ic: N straight-line FFMAs run twice (pass1 cold, pass2 warm), 1 warp/CTA, 132 CTAs (1 per SM), %smid recorded.
Before each trial: an evictor kernel with ~EV different instructions runs on all SMs. Modes: none / warm 1 CTA /
warm 132 CTAs. python tools/fresh/icache/ic2.py"""
import os, torch, fcntl
from torch.utils.cpp_extension import load_inline
N = int(os.environ.get("ICN", "1024")); EV = int(os.environ.get("ICEV", "8192"))
body = "\n".join(f"    a{i%8} = fmaf(a{i%8}, b, {i}.f);" for i in range(N))
ev = "\n".join(f"    a{i%8} = fmaf(a{i%8}, b, {i + 0.5}f);" for i in range(EV))
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
  unsigned sm; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
  if (threadIdx.x == 0) { ts[blockIdx.x * 4] = t0; ts[blockIdx.x * 4 + 3] = sm; }
  out[blockIdx.x * 32 + threadIdx.x] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}
__global__ void __launch_bounds__(32) evict(float* out, float b) {
  float a0 = threadIdx.x, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3, a4 = a0 + 4, a5 = a0 + 5, a6 = a0 + 6, a7 = a0 + 7;
''' + ev + r'''
  out[blockIdx.x * 32 + threadIdx.x] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}
void run(torch::Tensor out, torch::Tensor ts, int ctas) { ic<<<ctas, 32>>>(out.data_ptr<float>(), (unsigned long long*)ts.data_ptr(), 1.0001f); }
void runev(torch::Tensor out) { evict<<<264, 32>>>(out.data_ptr<float>(), 1.0001f); }
'''
m = load_inline(f"fresh_ic2_{N}_{EV}", cpp_sources="void run(torch::Tensor, torch::Tensor, int); void runev(torch::Tensor);",
                cuda_sources=src, functions=["run", "runev"], extra_cuda_cflags=["-O3", "-arch=sm_90a"], verbose=False)
out = torch.zeros(264 * 32, device="cuda"); ts = torch.zeros(132 * 4, dtype=torch.int64, device="cuda")
with open("/tmp/okc-gpu.lock", "w") as lk:
    fcntl.flock(lk, fcntl.LOCK_EX)
    for mode in ("noevict", "evict", "evict+warm1", "evict+warm132"):
        r1, r2, wsm = [], [], None
        for it in range(10):
            if mode != "noevict":
                m.runev(out)
            if mode == "evict+warm1":
                m.run(out, ts, 1); torch.cuda.synchronize(); wsm = int(ts[3])
            if mode == "evict+warm132":
                m.run(out, ts, 132)
            torch.cuda.synchronize()
            m.run(out, ts, 132); torch.cuda.synchronize()
            t = ts.view(-1, 4).double()
            r1 += ((t[:, 1] - t[:, 0]) / 1000).tolist(); r2 += ((t[:, 2] - t[:, 1]) / 1000).tolist()
        r1.sort(); r2.sort()
        q = lambda r, f: r[int(f * (len(r) - 1))]
        print(f"N={N} ({N * 16 // 1024} KB) EV={EV} {mode:14s} pass1 p10/med/p90 {q(r1,.1):.2f}/{q(r1,.5):.2f}/{q(r1,.9):.2f} | pass2 med {q(r2,.5):.2f}")
