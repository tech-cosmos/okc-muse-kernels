# consumer-side instrumentation of a twm-shaped cp.async.bulk ring: where do the per-stage cycles go?
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
__device__ __forceinline__ bool mtest(uint64_t* b, int ph) {
  uint32_t r;
  asm volatile("{\n.reg .pred p;\nmbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n" : "=r"(r) : "r"(su(b)), "r"(ph) : "memory");
  return r;
}
constexpr int STG = 26624, NSTG = 8;
__global__ void __launch_bounds__(288, 1) ring(const char* __restrict__ src, int per, int dly, int cnt, long long* out) {
  extern __shared__ __align__(128) char sm[];
  __shared__ __align__(8) uint64_t full[NSTG], empty[NSTG];
  const int tid = threadIdx.x;
  if (tid == 0) {
    for (int i = 0; i < NSTG; ++i) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(su(&full[i])), "r"(1));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(su(&empty[i])), "r"(cnt));
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (tid >= 256) {
    if (tid != 256) return;
    int st = 0, ph = 0;
    long long wsum = 0;
    for (int i = 0; i < per; ++i) {
      long long a = clock64();
      if (i >= NSTG) mwait(&empty[st], ph ^ 1);
      wsum += clock64() - a;
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(su(&full[st])), "r"(19456) : "memory");
      const char* gw = src + (((size_t)blockIdx.x * per + i) * 16384) % (256u << 20);
      const char* gs = src + (256u << 20) + (((size_t)blockIdx.x * per + i) * 2048) % (64u << 20);
      const char* gx = src + (512u << 20) + ((size_t)blockIdx.x * 8192 + (i & 63) * 1024);
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                   ::"r"(su(sm + st * STG)), "l"(gw), "r"(16384), "r"(su(&full[st])) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                   ::"r"(su(sm + st * STG + 16384)), "l"(gs), "r"(2048), "r"(su(&full[st])) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                   ::"r"(su(sm + st * STG + 18432)), "l"(gx), "r"(1024), "r"(su(&full[st])) : "memory");
      if (++st == NSTG) { st = 0; ph ^= 1; }
    }
    out[blockIdx.x * 8 + 4] = wsum;
    return;
  }
  int st = 0, ph = 0;
  uint32_t acc = 0;
  long long t0 = clock64(), wsum = 0, ready = 0, dsum = 0;
  for (int i = 0; i < per; ++i) {
    long long a = clock64();
    ready += mtest(&full[st], ph);
    mwait(&full[st], ph);
    long long b = clock64();
    wsum += b - a;
    acc += *reinterpret_cast<const uint32_t*>(sm + st * STG + tid * 4);
    if (dly) { long long c0 = clock64(); while (clock64() - c0 < dly) acc += 1; }
    dsum += clock64() - b;
    if (cnt == 256 || (tid & 31) == 0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(su(&empty[st])) : "memory");
    if (++st == NSTG) { st = 0; ph ^= 1; }
  }
  if (tid == 0) { out[blockIdx.x * 8] = clock64() - t0 + (acc == 12345); out[blockIdx.x * 8 + 1] = wsum; out[blockIdx.x * 8 + 2] = ready; out[blockIdx.x * 8 + 3] = dsum; }
}
void run(torch::Tensor src, int64_t per, int64_t grid, int64_t dly, int64_t cnt, torch::Tensor out) {
  cudaFuncSetAttribute(ring, cudaFuncAttributeMaxDynamicSharedMemorySize, STG * NSTG);
  ring<<<grid, 288, STG * NSTG>>>((const char*)src.data_ptr(), (int)per, (int)dly, (int)cnt, (long long*)out.data_ptr());
}
"""
m = load_inline("ringmb6", cpp_sources="void run(torch::Tensor src, int64_t per, int64_t grid, int64_t dly, int64_t cnt, torch::Tensor out);",
                cuda_sources=src, functions=["run"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])
big = torch.empty(1 << 30, dtype=torch.uint8, device="cuda")
out = torch.zeros(132 * 8, dtype=torch.int64, device="cuda")
per = 1024
with open('/tmp/okc-gpu.lock', 'w') as lk:
    fcntl.flock(lk, fcntl.LOCK_EX)
    for cnt in (256, 8):
        for dly in (0, 400, 1000):
            for grid in (16,):
                m.run(big, per, grid, dly, cnt, out); torch.cuda.synchronize()
                o = out.view(132, 8)[:grid].double().mean(0).tolist()
                print(f"cnt {cnt} dly {dly} G{grid}: cyc/stage {o[0]/per:.0f}  cons wait {o[1]/per:.0f}  work {o[3]/per:.0f}  ready-before-wait {o[2]/per:.2f}  prod empty-wait {o[4]/per:.0f}", flush=True)
