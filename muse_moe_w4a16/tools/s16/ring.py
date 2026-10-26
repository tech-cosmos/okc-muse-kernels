# per-SM cp.async.bulk ring ingest microbenchmark: python ring.py
import torch, fcntl, sys
from torch.utils.cpp_extension import load_inline
src = r"""
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cstdint>
__device__ __forceinline__ uint32_t su(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mwait(uint64_t* b, int ph) {
  asm volatile("{\n.reg .pred p;\nW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}\n" ::"r"(su(b)), "r"(ph) : "memory");
}
__global__ void __launch_bounds__(288, 1) ring(const char* __restrict__ src, size_t span, int stg, int nstg, int nchunk,
                                               int per, int nprod, long long* out) {
  extern __shared__ __align__(128) char sm[];
  __shared__ __align__(8) uint64_t full[16], empty[16];
  const int tid = threadIdx.x;
  if (tid == 0) {
    for (int i = 0; i < nstg; ++i) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(su(&full[i])), "r"(1));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(su(&empty[i])), "r"(256));
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  long long t0 = clock64();
  const char* base = src + ((size_t)blockIdx.x * per * stg) % span;
  if (tid >= 256) {
    if (tid != 256) return;
    int st = 0, ph = 0;
    for (int i = 0; i < per; ++i) {
      if (i >= nstg) mwait(&empty[st], ph ^ 1);
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(su(&full[st])), "r"(stg) : "memory");
      const int cs = stg / nchunk;
      for (int c = 0; c < nchunk; ++c) {
        const char* g = base + (size_t)i * stg + c * cs;
        if (g >= src + span) g -= span;
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                     ::"r"(su(sm + st * stg + c * cs)), "l"(g), "r"(cs), "r"(su(&full[st])) : "memory");
      }
      if (++st == nstg) { st = 0; ph ^= 1; }
    }
    return;
  }
  int st = 0, ph = 0;
  uint32_t acc = 0;
  for (int i = 0; i < per; ++i) {
    mwait(&full[st], ph);
    acc += *reinterpret_cast<const uint32_t*>(sm + st * stg + tid * 4);
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(su(&empty[st])) : "memory");
    if (++st == nstg) { st = 0; ph ^= 1; }
  }
  if (tid == 0) out[blockIdx.x] = clock64() - t0 + (acc == 12345);
}
void run(torch::Tensor src, int64_t span, int64_t stg, int64_t nstg, int64_t nchunk, int64_t per, int64_t grid, torch::Tensor out) {
  cudaFuncSetAttribute(ring, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(stg * nstg));
  ring<<<grid, 288, stg * nstg>>>((const char*)src.data_ptr(), span, stg, nstg, nchunk, per, 1, (long long*)out.data_ptr());
}
"""
m = load_inline("ringmb", cpp_sources="void run(torch::Tensor src, int64_t span, int64_t stg, int64_t nstg, int64_t nchunk, int64_t per, int64_t grid, torch::Tensor out);",
                cuda_sources=src, functions=["run"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])
big = torch.empty(1 << 30, dtype=torch.uint8, device="cuda")
out = torch.zeros(132, dtype=torch.int64, device="cuda")
def t(span, stg, nstg, nchunk, per, grid, reps=5):
    ts = []
    with open('/tmp/okc-gpu.lock', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for _ in range(reps):
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record(); m.run(big, span, stg, nstg, nchunk, per, grid, out); b.record(); torch.cuda.synchronize()
            ts.append(a.elapsed_time(b) * 1e3)
    us = sorted(ts)[len(ts) // 2]
    return us, grid * per * stg / us / 1e3  # TB/s
for span_name, span in (("L2 16MB", 16 << 20), ("HBM 1GB", 1 << 30)):
    for grid in (16, 132):
        for stg, nstg, nch in ((16384, 8, 1), (18432, 8, 1), (16384, 12, 1), (32768, 6, 1), (16384, 8, 4), (8192, 16, 1), (4096, 16, 1)):
            per = (4 << 20) // stg
            us, tb = t(span, stg, nstg, nch, per, grid)
            print(f"{span_name} grid {grid} stage {stg} x{nstg} chunks {nch}: {us:.1f} us, {tb:.2f} TB/s, {tb*1e3/grid:.1f} GB/s/SM", flush=True)
