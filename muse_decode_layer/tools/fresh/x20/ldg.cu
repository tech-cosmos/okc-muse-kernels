// x20: chip L2->SM read rate with ld.global.v4 (LSU path) vs region size and sharing.
// mode 0: each CTA reads its own slice sequence of an R-byte region; mode 1: all CTAs read the SAME addresses in lockstep-ish order.
#include <cstdio>
#include <cuda_runtime.h>
__global__ void rd(const int4* __restrict__ src, size_t n16, int iters, int mode, int4* sink) {
  int4 acc = make_int4(0, 0, 0, 0);
  size_t per = blockDim.x * 8;
  size_t start = mode ? 0 : ((size_t)blockIdx.x * 7919 * per) % n16;
  for (int it = 0; it < iters; it++) {
    size_t o = (start + (size_t)it * per) % (n16 - per);
    int4 v[8];
#pragma unroll
    for (int u = 0; u < 8; u++) v[u] = __ldcg(src + o + u * blockDim.x + threadIdx.x);
#pragma unroll
    for (int u = 0; u < 8; u++) { acc.x ^= v[u].x; acc.y ^= v[u].y; acc.z ^= v[u].z; acc.w ^= v[u].w; }
  }
  if (acc.x == 0x12345678) sink[threadIdx.x] = acc;
}
int main() {
  int nsm; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
  size_t big = (size_t)1 << 30; int4* src; cudaMalloc(&src, big); cudaMemset(src, 1, big);
  int4* sink; cudaMalloc(&sink, 1 << 16);
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  size_t regs[3] = {(size_t)8 << 20, (size_t)32 << 20, big};
  for (int mode = 0; mode < 2; mode++) for (int r = 0; r < 3; r++) for (int thr = 256; thr <= 1024; thr *= 2) for (int cps = 1; cps <= 2; cps++) {
    int g = nsm * cps; if (thr * cps > 2048) continue;
    size_t per = (size_t)thr * 8 * 16; int iters = (int)(((size_t)8 << 20) / per / cps);
    float ms;
    for (int rep = 0; rep < 2; rep++) { cudaEventRecord(a); rd<<<g, thr>>>(src, regs[r] / 16, iters, mode, sink); cudaEventRecord(b); cudaEventSynchronize(b); }
    cudaEventElapsedTime(&ms, a, b);
    double bytes = (double)iters * per * g;
    printf("mode %d region %5zuMB thr %4d ctas/sm %d  chip %6.2f TB/s  per-SM %6.1f GB/s\n", mode, regs[r] >> 20, thr, cps, bytes / ms / 1e9, bytes / ms / 1e6 / nsm);
  }
  printf("%s\n", cudaGetErrorString(cudaGetLastError()));
}
