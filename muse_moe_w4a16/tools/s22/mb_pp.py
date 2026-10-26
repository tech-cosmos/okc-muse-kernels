# s18 microbenchmark: does dequant ALU work overlap with wgmma on Hopper? RS (A in registers) vs SS (A in smem).
import torch, sys
from torch.utils.cpp_extension import load_inline
def gen_rs(n):
    na = n // 2; dl = ", ".join(f"%{i}" for i in range(na)); al = ", ".join(f"%{na + i}" for i in range(4))
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(na))
    return (f"template <> __device__ __forceinline__ void wg_rs<{n}>(float* d, const uint32_t* a, uint64_t desc) {{\n"
            f'  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, %{na + 5}, 0;\\n"\n'
            f'    "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{dl}}}, {{{al}}}, %{na + 4}, p, 1, 1, 0;\\n}}\\n"\n'
            f'    : {outs} : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));\n}}\n')
def gen_ss(n):
    na = n // 2; dl = ", ".join(f"%{i}" for i in range(na))
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(na))
    return (f"template <> __device__ __forceinline__ void wg_ss<{n}>(float* d, uint64_t da, uint64_t db) {{\n"
            f'  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, %{na + 2}, 0;\\n"\n'
            f'    "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{dl}}}, %{na}, %{na + 1}, p, 1, 1, 0, 0;\\n}}\\n"\n'
            f'    : {outs} : "l"(da), "l"(db), "r"(1));\n}}\n')
NS = (8, 32, 40, 48, 56, 64)
SRC = r"""
#include <cstdint>
#ifndef DQV
#define DQV 0
#endif
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ uint32_t bmul(uint32_t a, uint32_t b) {
  uint32_t r; asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
__device__ __forceinline__ void dq(uint32_t q, uint32_t s0, uint32_t s1, uint32_t* a) {
  const uint32_t M = 0x81C081C0u;
  const uint32_t o0 = q & M, o1 = __funnelshift_l(q, q, 10) & M, o2 = __funnelshift_l(q, q, 13) & M;
  const uint32_t o3 = ((q & 0x001B001Bu) * 16648u) & M;
#if DQV == 0
  a[0] = bmul(o0, s0); a[1] = bmul(o1, s1); a[2] = bmul(o2, s0); a[3] = bmul(o3, s1);
#elif DQV == 1
  a[0] = o0 ^ s0; a[1] = o1 ^ s1; a[2] = o2 ^ s0; a[3] = o3 ^ s1;
#elif DQV == 2
  a[0] = o0 * s0; a[1] = o1 * s1; a[2] = o2 * s0; a[3] = o3 * s1;
#else
  uint32_t r0, r1, r2, r3;
  asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(r0) : "r"(o0), "r"(s0)); asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(r1) : "r"(o1), "r"(s1));
  asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(r2) : "r"(o2), "r"(s0)); asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(r3) : "r"(o3), "r"(s1));
  a[0] = r0; a[1] = r1; a[2] = r2; a[3] = r3;
#endif
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int K> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(K) : "memory"); }
__device__ __forceinline__ void fence_r(uint32_t& r) { asm volatile("" : "+r"(r)::"memory"); }
__device__ __forceinline__ void fence_f(float& r) { asm volatile("" : "+f"(r)::"memory"); }
__device__ __forceinline__ uint64_t sw128_desc(const void* p) {
  return (uint64_t)((smem_u32(p) & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62); }
template <int N> __device__ __forceinline__ void wg_rs(float* d, const uint32_t* a, uint64_t desc);
template <int N> __device__ __forceinline__ void wg_ss(float* d, uint64_t da, uint64_t db);
@@GEN@@
// MODE 1 ALU only, 2 RS MMA only, 3 RS both (twm), 4 SS MMA only, 5 ALU + SS MMA, 6 ALU + STS A + SS MMA (realistic SS)
template <int MODE, int RT, int N, bool PP = true>
__global__ void __launch_bounds__(256, 1) mb(const uint32_t* __restrict__ w, unsigned long long* out, float* sinkf, uint32_t* sinku, int iters) {
  extern __shared__ __align__(1024) char sm[];
  char* sB = sm;                       // B: N rows x 128 B (SW128), 8 stages
  char* sW = sm + 4 * 8192;            // weights: 8 stages x 8 warps x RT x 32 lanes x 16 B
  char* sA = sW + 4 * 8 * RT * 512;    // A tiles for SS: 2 WGs x 2 buffers x RT x 8 KB (64 rows x 128 B)
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, wg = warp >> 2;
  for (int i = tid; i < (4 * 8192 + 4 * 8 * RT * 512) / 4; i += 256) reinterpret_cast<uint32_t*>(sm)[i] = w[i & 4095] & 0x3F3F3F3Fu;
  for (int i = tid; i < 2 * 2 * RT * 8192 / 4; i += 256) reinterpret_cast<uint32_t*>(sA)[i] = 0x3F803F80u;
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  constexpr int NA = N / 2;
  float acc[RT][NA];
  for (int r = 0; r < RT; ++r) for (int i = 0; i < NA; ++i) acc[r][i] = 0.f;
  uint32_t afA[RT][4], afB[RT][4], xs = 0;
  for (int r = 0; r < RT; ++r) for (int i = 0; i < 4; ++i) { afA[r][i] = 0x3F803F80u; afB[r][i] = 0x3F803F80u; }
  const uint32_t s0 = 0x3F803F80u ^ (lane & 1), s1 = 0x3F803F80u;
  uint32_t afS[RT][4][4];
  if (MODE == 7 && wg == 1) asm volatile("bar.arrive 2, 256;" ::: "memory");
  const unsigned long long t0 = clock64();
  if (MODE == 7) {
    auto dqs = [&](int it, uint32_t (&af)[RT][4][4]) {
      const int st = it & 3;
      uint4 wv[RT];
#pragma unroll
      for (int r = 0; r < RT; ++r) wv[r] = reinterpret_cast<const uint4*>(sW + st * 8 * RT * 512)[(warp * RT + r) * 32 + lane];
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int r = 0; r < RT; ++r) {
          const uint32_t q = (j == 0 ? wv[r].x : j == 1 ? wv[r].y : j == 2 ? wv[r].z : wv[r].w) ^ it;
          dq(q, s0, s1, af[r][j]);
        }
    };
    auto mma = [&](int it, uint32_t (&af)[RT][4][4]) {
      const uint64_t d0 = sw128_desc(sB + (it & 3) * 8192);
      if (PP) asm volatile("bar.sync %0, 256;" ::"r"(2 + wg) : "memory");
      wg_fence();
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int r = 0; r < RT; ++r) wg_rs<N>(acc[r], af[r][j], d0 + j * 2);
      wg_commit();
      if (PP && !(wg == 1 && it == iters - 1)) asm volatile("bar.arrive %0, 256;" ::"r"(3 - wg) : "memory");
    };
    uint32_t afT[RT][4][4];
    dqs(0, afS);
#pragma unroll 1
    for (int it = 0; it < iters; it += 2) {
      mma(it, afS);
      dqs(it + 1, afT);
      wg_wait<0>();
      mma(it + 1, afT);
      dqs(it + 2, afS);
      wg_wait<0>();
    }
  } else

#pragma unroll 1
  for (int it = 0; it < iters; ++it) {
    const int st = it & 3;
    const char* sb = sB + st * 8192;
    uint4 wv[RT];
#pragma unroll
    for (int r = 0; r < RT; ++r) wv[r] = reinterpret_cast<const uint4*>(sW + st * 8 * RT * 512)[(warp * RT + r) * 32 + lane];
    const uint64_t d0 = sw128_desc(sb);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const bool odd = j & 1;
      const uint32_t q0 = 0;
      if (MODE == 1 || MODE == 3 || MODE == 5 || MODE == 6) {
#pragma unroll
        for (int r = 0; r < RT; ++r) {
          const uint32_t q = (j == 0 ? wv[r].x : j == 1 ? wv[r].y : j == 2 ? wv[r].z : wv[r].w) ^ it;
          if (odd) dq(q, s0, s1, afB[r]); else dq(q, s0, s1, afA[r]);
        }
      }
      if (MODE == 1 || MODE == 5) {
#pragma unroll
        for (int r = 0; r < RT; ++r) xs ^= (odd ? afB[r][0] ^ afB[r][1] ^ afB[r][2] ^ afB[r][3] : afA[r][0] ^ afA[r][1] ^ afA[r][2] ^ afA[r][3]);
      }
      char* abuf = sA + ((wg * 2 + (odd ? 1 : 0)) * RT) * 8192;
      if (MODE == 6) {  // each lane stores its 4 A regs (fragment layout ignored: cost only)
#pragma unroll
        for (int r = 0; r < RT; ++r)
          reinterpret_cast<uint4*>(abuf + r * 8192)[(warp & 3) * 32 + lane + (j & 2) * 64] =
              odd ? make_uint4(afB[r][0], afB[r][1], afB[r][2], afB[r][3]) : make_uint4(afA[r][0], afA[r][1], afA[r][2], afA[r][3]);
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        asm volatile("bar.sync %0, 128;" ::"r"(1 + wg) : "memory");
      }
      if (MODE >= 2 && MODE != 1) {
        wg_fence();
#pragma unroll
        for (int r = 0; r < RT; ++r) {
          if (MODE == 2 || MODE == 3) wg_rs<N>(acc[r], odd ? afB[r] : afA[r], d0 + j * 2);
          else wg_ss<N>(acc[r], sw128_desc(abuf + r * 8192) + j * 2, d0 + j * 2);
        }
        wg_commit();
        wg_wait<1>();
#pragma unroll
        for (int r = 0; r < RT; ++r)
#pragma unroll
          for (int i = 0; i < 4; ++i) { if (odd) fence_r(afA[r][i]); else fence_r(afB[r][i]); }
      }
    }
  }
  wg_wait<0>();
  const unsigned long long t1 = clock64();
  float s = 0.f;
  for (int r = 0; r < RT; ++r) for (int i = 0; i < NA; ++i) s += acc[r][i];
  sinkf[blockIdx.x * 256 + tid] = s; sinku[blockIdx.x * 256 + tid] = xs;
  if (tid == 0) out[blockIdx.x] = t1 - t0;
}
#define I(M, R, N) template __global__ void mb<M, R, N>(const uint32_t*, unsigned long long*, float*, uint32_t*, int);
int64_t run(int64_t mode, int64_t rt, int64_t n, torch::Tensor w, torch::Tensor out, torch::Tensor sf, torch::Tensor su, int64_t iters, int64_t grid) {
  const int SM = 4 * 8192 + 4 * 8 * 4 * 512 + 2 * 2 * 4 * 8192 + 1024;
  #define GO(M, R, NN) if (mode == M && rt == R && n == NN) { auto k = mb<M, R, NN>; cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SM); \
    k<<<grid, 256, SM>>>((const uint32_t*)w.data_ptr(), (unsigned long long*)out.data_ptr(), (float*)sf.data_ptr(), (uint32_t*)su.data_ptr(), (int)iters); return 0; }
  #define GP(R, NN) if (mode == 8 && rt == R && n == NN) { auto k = mb<7, R, NN, false>; cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SM); \
    k<<<grid, 256, SM>>>((const uint32_t*)w.data_ptr(), (unsigned long long*)out.data_ptr(), (float*)sf.data_ptr(), (uint32_t*)su.data_ptr(), (int)iters); return 0; }
  #define GOM(R, NN) GO(1, R, NN) GO(2, R, NN) GO(3, R, NN) GO(4, R, NN) GO(5, R, NN) GO(6, R, NN)
  GO(1, 2, 64) GO(1, 4, 8) GO(1, 4, 32)
  GO(2, 2, 64) GO(3, 2, 64) GO(7, 2, 64) GP(2, 64)
  GO(2, 4, 8) GO(3, 4, 8) GO(7, 4, 8) GP(4, 8)
  GO(2, 4, 32) GO(3, 4, 32) GO(7, 4, 32) GP(4, 32)
  return -1;
}
"""
SRC = SRC.replace("@@GEN@@", "".join(gen_rs(n) + gen_ss(n) for n in NS))
import os
DQV = int(os.environ.get("DQV", "0"))
lib = load_inline(f"mb22_pp{DQV}", cpp_sources="int64_t run(int64_t, int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t);",
                  cuda_sources="#include <torch/extension.h>\n" + SRC, functions=["run"],
                  extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3", f"-DDQV={DQV}"], verbose=False)
w = torch.randint(0, 2**31, (4096,), dtype=torch.int32, device="cuda")
G = int(sys.argv[1]) if len(sys.argv) > 1 else 1
out = torch.zeros(132, dtype=torch.int64, device="cuda"); sf = torch.zeros(132 * 256, device="cuda"); su = torch.zeros(132 * 256, dtype=torch.int32, device="cuda")
IT = 2000
names = {1: "ALU", 2: "RS MMA", 3: "RS both", 7: "pingpong", 8: "stage-batch no-pp"}
for rt, n in ((2, 64), (4, 8), (4, 32)):
    row = []
    for m in (1, 2, 3, 8):
        assert lib.run(m, rt, n, w, out, sf, su, 50, G) == 0
        lib.run(m, rt, n, w, out, sf, su, IT, G); torch.cuda.synchronize()
        row.append(f"{names[m]} {out[:G].float().mean().item() / IT:6.0f}")
    print(f"RT{rt} N{n} cyc/stage: " + " | ".join(row), flush=True)
