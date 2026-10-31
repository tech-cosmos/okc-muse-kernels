// x20: what stalls a single TMA issuer? Time (clock64) of the issue of NS bulk copies back to back, then of each wait.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__global__ void k(const char* src, int ns, int sb, int mode, long long* out) {
  extern __shared__ __align__(128) char sm[];
  uint64_t* bar = (uint64_t*)sm; char* buf = sm + 1024;
  if (threadIdx.x == 0) { for (int i = 0; i < ns; i++) asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(sa(&bar[i])));
    asm volatile("fence.mbarrier_init.release.cluster;"); }
  __syncthreads();
  if (threadIdx.x) return;
  size_t base = (size_t)blockIdx.x * 1048576 * 3;
  long long t0 = clock64(); long long ti[32], tw[32];
  for (int s = 0; s < ns; s++) {
    if (mode != 1) asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"(sa(&bar[s])), "r"(sb));
    if (mode != 2) asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
      :: "r"(sa(buf + (size_t)s * sb)), "l"(src + base + (size_t)s * sb), "r"(sb), "r"(sa(&bar[s])) : "memory");
    ti[s] = clock64();
  }
  for (int s = 0; s < ns && mode == 0; s++) {
    asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared.b64 p, [%0], 0;\n @!p bra W;\n}" :: "r"(sa(&bar[s])));
    tw[s] = clock64();
  }
  if (blockIdx.x == 0) { for (int s = 0; s < ns; s++) { out[s] = ti[s] - t0; out[32 + s] = mode == 0 ? tw[s] - t0 : 0; } }
}
int main() {
  char* src; cudaMalloc(&src, 1 << 30); cudaMemset(src, 1, 1 << 30); long long* out; cudaMallocManaged(&out, 64 * 8);
  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024);
  int nsm = 132;
  for (int mode = 0; mode < 3; mode++) for (int sb : {4096, 16384}) for (int g : {1, nsm}) {
    int ns = sb == 4096 ? 16 : 8;
    k<<<g, 32, 1024 + ns * sb>>>(src, ns, sb, mode, out); cudaDeviceSynchronize();
    k<<<g, 32, 1024 + ns * sb>>>(src, ns, sb, mode, out); cudaDeviceSynchronize();
    printf("mode %d (%s) SB %5d grid %3d issue clk:", mode, mode == 0 ? "expect+copy" : mode == 1 ? "copy only, no wait" : "expect only", sb, g);
    for (int s = 0; s < ns; s++) printf(" %lld", out[s]); printf("\n");
    if (mode == 0) { printf("   wait done clk:"); for (int s = 0; s < ns; s++) printf(" %lld", out[32 + s]); printf("\n"); }
  }
  printf("%s\n", cudaGetErrorString(cudaGetLastError()));
}
