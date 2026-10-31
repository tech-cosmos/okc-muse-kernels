// x20: TMA bulk ring, P producer warps (lane 0 each) own stages s = w mod P. All CTAs read the same offsets (SHARED)
// or distinct ones. Prints GB/s per SM. Tests whether the ~350 ns/stage cap is per waiting thread.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__global__ void ring(const char* src, size_t region, int ns, int sb, int iters, int P, int shared, int hint, unsigned long long* sink) {
  extern __shared__ __align__(128) char sm[];
  uint64_t* bar = (uint64_t*)sm; char* buf = sm + 1024;
  if (threadIdx.x == 0) { for (int i = 0; i < ns; i++) asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(sa(&bar[i])));
    asm volatile("fence.mbarrier_init.release.cluster;"); }
  __syncthreads();
  int w = threadIdx.x / 32; if ((threadIdx.x & 31) || w >= P) return;
  size_t base = shared ? 0 : (size_t)blockIdx.x * 7919 * sb;
  auto issue = [&](int s) { int slot = s % ns; size_t off = ((base + (size_t)s * sb) % (region - sb)) & ~(size_t)127;
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"(sa(&bar[slot])), "r"(sb));
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
      :: "r"(sa(buf + (size_t)slot * sb)), "l"(src + off), "r"(sb), "r"(sa(&bar[slot])) : "memory"); };
  for (int s = w; s < ns && s < iters; s += P) issue(s);
  unsigned long long acc = 0;
  for (int s = w; s < iters; s += P) {
    int slot = s % ns; uint32_t ph = (s / ns) & 1;
    if (hint) asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared.b64 p, [%0], %1, %2;\n @!p bra W;\n}" :: "r"(sa(&bar[slot])), "r"(ph), "r"(hint));
    else asm volatile("{\n .reg .pred p;\n W: mbarrier.test_wait.parity.shared.b64 p, [%0], %1;\n @!p bra W;\n}" :: "r"(sa(&bar[slot])), "r"(ph));
    acc += buf[(size_t)slot * sb + (s & 127)];
    if (s + ns < iters) issue(s + ns);
  }
  if (acc == 0xdeadbeef) *sink = acc;
}
int main() {
  int nsm; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
  char* src; cudaMalloc(&src, 32 << 20); cudaMemset(src, 1, 32 << 20); unsigned long long* sink; cudaMalloc(&sink, 8);
  cudaFuncSetAttribute(ring, cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024);
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  int cfg[][3] = {{16384, 8, 1}, {16384, 8, 2}, {16384, 8, 4}, {16384, 8, 8}, {8192, 16, 1}, {8192, 16, 4}, {8192, 16, 16},
                  {32768, 4, 1}, {32768, 4, 2}, {32768, 4, 4}, {32768, 6, 6}, {65536, 2, 1}, {65536, 2, 2}};
  for (int shared = 0; shared < 2; shared++) for (int hint = 0; hint < 2; hint++) for (auto& c : cfg) {
    int sb = c[0], ns = c[1], P = c[2]; int iters = (8 << 20) / sb; float ms;
    for (int r = 0; r < 2; r++) { cudaEventRecord(a); ring<<<nsm, 32 * P, 1024 + ns * sb>>>(src, 8 << 20, ns, sb, iters, P, shared, hint ? 1000000 : 0, sink); cudaEventRecord(b); cudaEventSynchronize(b); }
    cudaEventElapsedTime(&ms, a, b);
    printf("shared %d %s SB %6d NS %2d P %2d  %6.1f GB/s/SM  %6.0f ns/stage/CTA\n", shared, hint ? "try_wait " : "test_wait", sb, ns, P,
           (double)iters * sb / ms / 1e6, ms * 1e6 / iters);
  }
  printf("%s\n", cudaGetErrorString(cudaGetLastError()));
}
