# x21: microbench of RS wgmma loops (no memory, A regs constant, B = smem garbage): T=256 forms per CTA, 64 W rows x K 4096.
# form "n128tt2": both WGs every stage, n128 each (s100 tt2); "n256alt": WG w takes stages w, w+2, .. with n256 (one dq per W elem).
import sys
def wg(n):
    nd = n // 2  # fp32 acc regs per thread for m64nN
    outs = ", ".join(f"%{i}" for i in range(nd))
    a = ", ".join(f"%{nd + i}" for i in range(4))
    s = f'asm volatile("{{\\n wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{outs}}}, {{{a}}}, %{nd + 4}, 1, 1, 1, 1;\\n}}\\n"'
    s += " : " + ", ".join(f'"+f"(d[{i}])' for i in range(nd))
    s += ' : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));'
    return f"__device__ __forceinline__ void mma{n}(float* d, unsigned a0, unsigned a1, unsigned a2, unsigned a3, uint64_t db) {{ {s} }}\n"
src = r'''
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
__device__ __forceinline__ uint64_t dsw128(uint32_t a) {
  return (uint64_t)((a & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)64 << 32) | ((uint64_t)1 << 62); }
''' + wg(128) + wg(256) + r'''
template <int N, int ALT>
__global__ void __launch_bounds__(256, 1) k(int KS, float* out) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const int t = threadIdx.x & 127, w = threadIdx.x >> 7;
  float d[N / 2];
#pragma unroll
  for (int j = 0; j < N / 2; j++) d[j] = 0.f;
  unsigned A[16];
#pragma unroll
  for (int j = 0; j < 16; j++) A[j] = 0x3f803f80u ^ (t * j);
  const uint32_t b0 = (uint32_t)__cvta_generic_to_shared(sm) + (ALT ? 0 : w * 16384);
  for (int i = ALT ? w : 0; i < KS; i += ALT ? 2 : 1) {
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    const uint32_t b = b0 + (i & 3) * 32768;
#pragma unroll
    for (int j = 0; j < 4; j++) { if (N == 256) mma256(d, A[4*j], A[4*j+1], A[4*j+2], A[4*j+3], dsw128(b + j * 32)); else mma128(d, A[4*j], A[4*j+1], A[4*j+2], A[4*j+3], dsw128(b + j * 32)); }
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
#pragma unroll
    for (int j = 0; j < 16; j++) A[j] += 0x00010001u;
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
  float s = 0; for (int j = 0; j < N / 2; j++) s += d[j];
  if (s == 1.2345f) out[threadIdx.x] = s;
}
int main() {
  int nsm; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
  float* o; cudaMalloc(&o, 4096);
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  auto run = [&](auto kern, const char* name, int grid) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 140 * 1024);
    float best = 1e9;
    for (int r = 0; r < 5; r++) { cudaEventRecord(a); kern<<<grid, 256, 140 * 1024>>>(64, o); cudaEventRecord(b); cudaEventSynchronize(b);
      float ms; cudaEventElapsedTime(&ms, a, b); best = ms < best ? ms : best; }
    double fl = 2.0 * 256 * 64 * 4096 * grid;
    printf("%-10s grid %3d  %6.2f us  %5.0f TFLOPS (chip-equiv for 128 CTAs: %5.2f us)\n", name, grid, best * 1e3, fl / best / 1e9, best * 1e3);
  };
  run(k<128, 0>, "n128tt2", 128); run(k<256, 1>, "n256alt", 128); run(k<128, 0>, "n128tt2", 132); run(k<256, 1>, "n256alt", 132);
  printf("%s\n", cudaGetErrorString(cudaGetLastError()));
}
'''
open(sys.argv[1], "w").write(src)
