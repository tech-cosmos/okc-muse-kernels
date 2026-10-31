// x20: g30-like producer/consumer TMA ring: producer = warp 0 lane 0 (waits EMPTY, expect_tx, bulk), consumer = warp 1
// (all lanes wait FULL, lane 0 arrives EMPTY; optional LDS of the stage). Optional 2nd producer thread for alternate stages.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void wait(uint64_t* b, uint32_t ph) {
  asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n @!p bra W;\n}" :: "r"(sa(b)), "r"(ph) : "memory"); }
__global__ void k(const char* src, size_t region, int ns, int sb, int iters, int npro, int shared, int lds, unsigned long long* sink) {
  extern __shared__ __align__(128) char sm[];
  uint64_t* full = (uint64_t*)sm; uint64_t* empty = full + 32; char* buf = sm + 1024;
  if (threadIdx.x == 0) { for (int i = 0; i < ns; i++) { asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(sa(&full[i])));
      asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(sa(&empty[i]))); }
    asm volatile("fence.mbarrier_init.release.cluster;"); }
  __syncthreads();
  int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
  size_t lim = region - sb;
  if (warp < npro) {  // producer p: stages s = p, p + npro, ..
    if (lane) return;
    size_t off = shared ? 0 : ((size_t)blockIdx.x * 7919 * sb) % lim; off &= ~(size_t)127;
    size_t step = (size_t)sb; off = (off + warp * step) % lim;
    int st = warp, ph = 0;
    for (int s = warp; s < iters; s += npro) {
      wait(&empty[st], ph ^ 1);
      asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"(sa(&full[st])), "r"(sb) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
        :: "r"(sa(buf + (size_t)st * sb)), "l"(src + off), "r"(sb), "r"(sa(&full[st])) : "memory");
      off += step * npro; if (off >= lim) off -= lim; off &= ~(size_t)127;
      st += npro; if (st >= ns) { st -= ns; ph ^= 1; }
    }
    return;
  }
  if (warp != npro) return;
  unsigned long long acc = 0; int st = 0, ph = 0;
  for (int s = 0; s < iters; s++) {
    wait(&full[st], ph);
    if (lds) acc += buf[(size_t)st * sb + lane * 4];
    __syncwarp();
    if (lane == 0) asm volatile("mbarrier.arrive.shared.b64 _, [%0];" :: "r"(sa(&empty[st])) : "memory");
    if (++st == ns) { st = 0; ph ^= 1; }
  }
  if (acc == 0xdeadbeef) *sink = acc;
}
int main() {
  int nsm; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
  char* src; cudaMalloc(&src, 1 << 30); cudaMemset(src, 1, 1 << 30); unsigned long long* sink; cudaMalloc(&sink, 8);
  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024);
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  int cfg[][3] = {{6144, 10, 1}, {6144, 10, 2}, {6144, 20, 1}, {6144, 20, 2}, {6144, 20, 4}, {16384, 10, 1}, {16384, 10, 2}, {32768, 6, 1}, {32768, 6, 2}};
  for (size_t region : {(size_t)8 << 20, (size_t)1 << 30}) for (int shared = 0; shared < 2; shared++) for (int lds = 0; lds < 2; lds++) for (auto& c : cfg) {
    if (region > (8u << 20) && shared) continue;
    int sb = c[0], ns = c[1], np = c[2]; int iters = (8 << 20) / sb; float ms;
    for (int r = 0; r < 2; r++) { cudaEventRecord(a); k<<<nsm, 32 * (np + 1), 1024 + ns * sb>>>(src, region, ns, sb, iters, np, shared, lds, sink); cudaEventRecord(b); cudaEventSynchronize(b); }
    cudaEventElapsedTime(&ms, a, b);
    printf("region %4zuMB shared %d lds %d SB %5d NS %2d producers %d  %6.1f GB/s/SM  %5.0f ns/stage\n", region >> 20, shared, lds, sb, ns, np,
           (double)iters * sb / ms / 1e6, ms * 1e6 / iters);
  }
  printf("%s\n", cudaGetErrorString(cudaGetLastError()));
}
