# s18: dq + legacy mma.sync m16n8k16 (A = dequantized registers, B = x fragments by LDS) vs dq + RS wgmma (mb.py)
import torch, sys
from torch.utils.cpp_extension import load_inline
SRC = r"""
#include <cstdint>
__device__ __forceinline__ uint32_t bmul(uint32_t a, uint32_t b) { uint32_t r; asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
__device__ __forceinline__ void dq(uint32_t q, uint32_t s0, uint32_t s1, uint32_t* a) {
  const uint32_t M = 0x81C081C0u;
  const uint32_t o0 = q & M, o1 = __funnelshift_l(q, q, 10) & M, o2 = __funnelshift_l(q, q, 13) & M;
  const uint32_t o3 = ((q & 0x001B001Bu) * 16648u) & M;
  a[0] = bmul(o0, s0); a[1] = bmul(o1, s1); a[2] = bmul(o2, s0); a[3] = bmul(o3, s1);
}
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1)); }
// MODE 7: dq + mma.sync, 8: mma.sync only (A fixed), 9: dq only (A consumed by xor)
template <int MODE, int RT, int NB>
__global__ void __launch_bounds__(256, 1) mb(const uint32_t* __restrict__ w, unsigned long long* out, float* sinkf, uint32_t* sinku, int iters) {
  extern __shared__ __align__(1024) char sm[];
  char* sB = sm; char* sW = sm + 4 * 8192;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  for (int i = tid; i < (4 * 8192 + 4 * 8 * RT * 512) / 4; i += 256) reinterpret_cast<uint32_t*>(sm)[i] = w[i & 4095] & 0x3F3F3F3Fu;
  __syncthreads();
  float acc[RT][NB][4];
  for (int r = 0; r < RT; ++r) for (int b = 0; b < NB; ++b) for (int i = 0; i < 4; ++i) acc[r][b][i] = 0.f;
  uint32_t af[RT][4], xs = 0;
  for (int r = 0; r < RT; ++r) for (int i = 0; i < 4; ++i) af[r][i] = 0x3F803F80u;
  const uint32_t s0 = 0x3F803F80u ^ (lane & 1), s1 = 0x3F803F80u;
  const unsigned long long t0 = clock64();
#pragma unroll 1
  for (int it = 0; it < iters; ++it) {
    const int st = it & 3;
    uint4 wv[RT];
#pragma unroll
    for (int r = 0; r < RT; ++r) wv[r] = reinterpret_cast<const uint4*>(sW + st * 8 * RT * 512)[(warp * RT + r) * 32 + lane];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      uint32_t b[NB][2];
#pragma unroll
      for (int nb = 0; nb < NB; ++nb) {
        const uint2 v = reinterpret_cast<const uint2*>(sB + st * 8192)[(j * NB + nb) * 32 + lane];
        b[nb][0] = v.x; b[nb][1] = v.y;
      }
#pragma unroll
      for (int r = 0; r < RT; ++r) {
        if (MODE != 8) {
          const uint32_t q = (j == 0 ? wv[r].x : j == 1 ? wv[r].y : j == 2 ? wv[r].z : wv[r].w) ^ it;
          dq(q, s0, s1, af[r]);
        }
        if (MODE == 9) { xs ^= af[r][0] ^ af[r][1] ^ af[r][2] ^ af[r][3] ^ b[0][0]; continue; }
#pragma unroll
        for (int nb = 0; nb < NB; ++nb) mma16816(acc[r][nb], af[r], b[nb][0], b[nb][1]);
      }
    }
  }
  const unsigned long long t1 = clock64();
  float s = 0.f;
  for (int r = 0; r < RT; ++r) for (int b = 0; b < NB; ++b) for (int i = 0; i < 4; ++i) s += acc[r][b][i];
  sinkf[blockIdx.x * 256 + tid] = s; sinku[blockIdx.x * 256 + tid] = xs;
  if (tid == 0) out[blockIdx.x] = t1 - t0;
}
int64_t run(int64_t mode, int64_t rt, int64_t nb, torch::Tensor w, torch::Tensor out, torch::Tensor sf, torch::Tensor su, int64_t iters, int64_t grid) {
  const int SM = 4 * 8192 + 4 * 8 * 4 * 512 + 1024;
  #define GO(M, R, B) if (mode == M && rt == R && nb == B) { auto k = mb<M, R, B>; cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SM); \
    k<<<grid, 256, SM>>>((const uint32_t*)w.data_ptr(), (unsigned long long*)out.data_ptr(), (float*)sf.data_ptr(), (uint32_t*)su.data_ptr(), (int)iters); return 0; }
  #define GOM(R, B) GO(7, R, B) GO(8, R, B) GO(9, R, B)
  GOM(4, 1) GOM(4, 2) GOM(4, 4) GOM(2, 8)
  return -1;
}
"""
lib = load_inline("mb18_ms", cpp_sources="int64_t run(int64_t, int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t);",
                  cuda_sources="#include <torch/extension.h>\n" + SRC, functions=["run"],
                  extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"], verbose=False)
w = torch.randint(0, 2**31, (4096,), dtype=torch.int32, device="cuda")
G = int(sys.argv[1]) if len(sys.argv) > 1 else 1
out = torch.zeros(132, dtype=torch.int64, device="cuda"); sf = torch.zeros(132 * 256, device="cuda"); su = torch.zeros(132 * 256, dtype=torch.int32, device="cuda")
IT = 2000
names = {7: "dq+mma.sync", 8: "mma.sync only", 9: "dq only"}
for rt, nb in ((4, 1), (4, 2), (4, 4), (2, 8)):
    row = []
    for m in (7, 8, 9):
        assert lib.run(m, rt, nb, w, out, sf, su, 50, G) == 0
        lib.run(m, rt, nb, w, out, sf, su, IT, G); torch.cuda.synchronize()
        row.append(f"{names[m]} {out[:G].float().mean().item() / IT:6.0f}")
    print(f"RT{rt} N{nb * 8} cyc/stage: " + " | ".join(row), flush=True)
