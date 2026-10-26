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
                                               int per, int mode, long long* out) {
  extern __shared__ __align__(128) char sm[];
  __shared__ __align__(8) uint64_t full[16], empty[16];
  const int tid = threadIdx.x;
  if (tid == 0) {
    for (int i = 0; i < nstg; ++i) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(su(&full[i])), "r"(1));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(su(&empty[i])), "r"((mode & 1) ? 8 : 256));
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
      if (i >= nstg && !(mode & 2)) mwait(&empty[st], ph ^ 1);
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(su(&full[st])), "r"((mode & 16) ? ((mode & 32) ? 18432 : 19456) : stg) : "memory");
      const int cs = stg / nchunk;
      if (mode & 16) {  // twm-like stage: W 16K (stream A), S 2K (stream B), x 1K (stream C, re-read)
        const char* gw = src + (((size_t)blockIdx.x * per + i) * 16384) % (256u << 20);
        const char* gs = src + (256u << 20) + (((size_t)blockIdx.x * per + i) * 2048) % (64u << 20);
        const char* gx = src + (512u << 20) + ((size_t)blockIdx.x * 8192 + (i & 63) * 1024);
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                     ::"r"(su(sm + st * stg)), "l"(gw), "r"(16384), "r"(su(&full[st])) : "memory");
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                     ::"r"(su(sm + st * stg + 16384)), "l"(gs), "r"(2048), "r"(su(&full[st])) : "memory");
        if (!(mode & 32))
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                     ::"r"(su(sm + st * stg + 18432)), "l"(gx), "r"(1024), "r"(su(&full[st])) : "memory");
        if (++st == nstg) { st = 0; ph ^= 1; }
        continue;
      }
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
  if (mode & 8) { if (tid == 0) out[blockIdx.x] = 0; return; }
  for (int i = 0; i < per; ++i) {
    mwait(&full[st], ph);
    acc += *reinterpret_cast<const uint32_t*>(sm + st * stg + tid * 4);
    if (mode & 1) { __syncwarp(); if ((tid & 31) == 0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(su(&empty[st])) : "memory"); }
    else asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(su(&empty[st])) : "memory");
    if (++st == nstg) { st = 0; ph ^= 1; }
  }
  if (tid == 0) out[blockIdx.x] = clock64() - t0 + (acc == 12345);
}
void run(torch::Tensor src, int64_t span, int64_t stg, int64_t nstg, int64_t nchunk, int64_t per, int64_t grid, int64_t mode, torch::Tensor out) {
  cudaFuncSetAttribute(ring, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(stg * nstg));
  ring<<<grid, 288, stg * nstg>>>((const char*)src.data_ptr(), span, stg, nstg, nchunk, per, mode, (long long*)out.data_ptr());
}
"""
m = load_inline("ringmb3", cpp_sources="void run(torch::Tensor src, int64_t span, int64_t stg, int64_t nstg, int64_t nchunk, int64_t per, int64_t grid, int64_t mode, torch::Tensor out);",
                cuda_sources=src, functions=["run"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])
big = torch.empty(1 << 30, dtype=torch.uint8, device="cuda")
out = torch.zeros(132, dtype=torch.int64, device="cuda")
def t(span, stg, nstg, nchunk, per, grid, mode=0, reps=5):
    ts = []
    with open('/tmp/okc-gpu.lock', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for _ in range(reps):
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record(); m.run(big, span, stg, nstg, nchunk, per, grid, mode, out); b.record(); torch.cuda.synchronize()
            ts.append(a.elapsed_time(b) * 1e3)
    us = sorted(ts)[len(ts) // 2]
    return us, grid * per * stg / us / 1e3  # TB/s
for mode, stg in ((0, 16384), (16, 19456), (48, 18432), (16, 26624)):
    for grid in (16, 132):
        per = 1024
        us, tb = t(1 << 30, stg, 8, 1, per, grid, mode)
        print(f"mode {mode} grid {grid} stride {stg}: {us:.1f} us, {us*1e3/per:.0f} ns/stage", flush=True)
