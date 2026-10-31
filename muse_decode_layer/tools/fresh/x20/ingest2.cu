// x20: per-SM L2->smem ingest vs bytes in flight. One CTA per SM, a ring of NS stages of SB bytes
// filled by cp.async.bulk (TMA bulk copy) from a REGION-byte buffer (L2-resident if small).
// Each CTA reads ITERS stages; consumer = one warp that waits each full barrier and re-arms.
// usage: ./ingest  (prints a table)
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); return 1; } } while (0)

__device__ __forceinline__ void mbar_init(uint64_t* b, int n) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(b)), "r"(n));
}
__device__ __forceinline__ void mbar_expect(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(b)), "r"(bytes));
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t ph) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(b);
  asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n @!p bra W;\n}" :: "r"(a), "r"(ph));
}
__device__ __forceinline__ void bulk(void* dst, const void* src, uint32_t bytes, uint64_t* b) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src), "r"(bytes), "r"((uint32_t)__cvta_generic_to_shared(b)) : "memory");
}

__global__ void ring(const char* src, size_t region, int ns, int sb, int iters, int chunk, unsigned long long* sink) {
  extern __shared__ __align__(128) char sm[];
  uint64_t* bar = (uint64_t*)sm;
  char* buf = sm + 1024;
  if (threadIdx.x == 0) {
    for (int i = 0; i < ns; i++) mbar_init(&bar[i], 1);
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  __syncthreads();
  if (threadIdx.x != 0) return;
  // stage s of this CTA reads region offset ((blockIdx * iters + s) * sb) % region
  size_t base = (size_t)blockIdx.x * 7919 * sb;
  auto issue = [&](int s) {
    int slot = s % ns;
    size_t off = (base + (size_t)s * sb) % (region - sb);
    off &= ~(size_t)127;
    mbar_expect(&bar[slot], sb);
    for (int c = 0; c < sb; c += chunk) bulk(buf + slot * sb + c, src + off + c, chunk, &bar[slot]);
  };
  for (int s = 0; s < ns && s < iters; s++) issue(s);
  unsigned long long acc = 0;
  for (int s = 0; s < iters; s++) {
    int slot = s % ns;
    mbar_wait(&bar[slot], (s / ns) & 1);
    acc += buf[slot * sb + (s & 127)];
    if (s + ns < iters) issue(s + ns);
  }
  if (acc == 0xdeadbeef) *sink = acc;
}

int main() {
  int nsm; CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  size_t big = (size_t)1 << 30;
  char* src; CK(cudaMalloc(&src, big)); CK(cudaMemset(src, 1, big));
  unsigned long long* sink; CK(cudaMalloc(&sink, 8));
  CK(cudaFuncSetAttribute(ring, cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024));
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  size_t regions[2] = {(size_t)8 << 20, big};
  int sbs[5] = {16384, 32768, 65536, 98304, 196608};
  printf("nsm %d\n", nsm);
  printf("%-6s %6s %4s %8s %10s %10s\n", "region", "SB", "NS", "inflight", "GB/s/SM", "TB/s chip");
  for (int r = 0; r < 1; r++)
    for (int si = 0; si < 5; si++)
      for (int ns = 1; ns <= 24; ns *= 2) {
        int sb = sbs[si];
        if ((size_t)ns * sb > 200 * 1024) continue;
        for (int chunk = 4096; chunk <= sb; chunk *= 4) {
        size_t per = (size_t)64 << 20;  // bytes per CTA total... keep runtime ~ms
        int iters = (int)(((size_t)8 << 20) / sb);  // 8 MB per CTA
        for (int rep = 0; rep < 2; rep++) {  // warm, then timed
          cudaEventRecord(a);
          ring<<<nsm, 32, 1024 + ns * sb>>>(src, regions[r], ns, sb, iters, chunk, sink);
          cudaEventRecord(b);
          CK(cudaEventSynchronize(b));
        }
        float ms; cudaEventElapsedTime(&ms, a, b);
        double bytes = (double)iters * sb;
        printf("%-6s %6d c%6d %4d %7dK %10.1f %10.2f\n", r ? "1GB" : "8MB", sb, chunk, ns, ns * sb / 1024, bytes / (ms * 1e-3) / 1e9,
               bytes * nsm / (ms * 1e-3) / 1e12);
        (void)per;
        }
      }
  // also NS 12 / 6 / 3 at 16K
  return 0;
}
