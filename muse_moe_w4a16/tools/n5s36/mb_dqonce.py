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
NS = (8, 32, 64, 128)
SRC = r"""
#include <cstdint>
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ uint32_t bmul(uint32_t a, uint32_t b) {
  uint32_t r; asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
__device__ __forceinline__ void dq(uint32_t q, uint32_t s0, uint32_t s1, uint32_t* a) {
  const uint32_t M = 0x81C081C0u;
  const uint32_t o0 = q & M, o1 = __funnelshift_l(q, q, 10) & M, o2 = __funnelshift_l(q, q, 13) & M;
  const uint32_t o3 = ((q & 0x001B001Bu) * 16648u) & M;
  a[0] = bmul(o0, s0); a[1] = bmul(o1, s1); a[2] = bmul(o2, s0); a[3] = bmul(o3, s1);
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

template <int RT, int N, int C>
__global__ void __launch_bounds__(256, 1) mbc(const uint32_t* __restrict__ w, unsigned long long* out, float* sinkf, int iters) {
  extern __shared__ __align__(1024) char sm[];
  char* sB = sm; char* sW = sm + 4 * 4 * 8192;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  for (int i = tid; i < (16 * 8192 + 4 * 8 * RT * 512) / 4; i += 256) reinterpret_cast<uint32_t*>(sm)[i] = w[i & 4095] & 0x3F3F3F3Fu;
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  constexpr int NA = N / 2;
  float acc[RT][C][NA];
  for (int r = 0; r < RT; ++r) for (int c = 0; c < C; ++c) for (int i = 0; i < NA; ++i) acc[r][c][i] = 0.f;
  uint32_t afA[RT][4], afB[RT][4];
  for (int r = 0; r < RT; ++r) for (int i = 0; i < 4; ++i) { afA[r][i] = 0x3F803F80u; afB[r][i] = 0x3F803F80u; }
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
      const bool odd = j & 1;
#pragma unroll
      for (int r = 0; r < RT; ++r) {
        const uint32_t q = (j == 0 ? wv[r].x : j == 1 ? wv[r].y : j == 2 ? wv[r].z : wv[r].w) ^ it;
        if (odd) dq(q, s0, s1, afB[r]); else dq(q, s0, s1, afA[r]);
      }
      wg_fence();
#pragma unroll
      for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int c = 0; c < C; ++c) wg_rs<N>(acc[r][c], odd ? afB[r] : afA[r], sw128_desc(sB + (st * 4 + c) * 8192) + j * 2);
      wg_commit();
      wg_wait<1>();
#pragma unroll
      for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int i = 0; i < 4; ++i) { if (odd) fence_r(afA[r][i]); else fence_r(afB[r][i]); }
    }
  }
  wg_wait<0>();
  const unsigned long long t1 = clock64();
  float s = 0.f;
  for (int r = 0; r < RT; ++r) for (int c = 0; c < C; ++c) for (int i = 0; i < NA; ++i) s += acc[r][c][i];
  sinkf[blockIdx.x * 256 + tid] = s;
  if (tid == 0) out[blockIdx.x] = t1 - t0;
}
int64_t run(int64_t rt, int64_t n, int64_t c, torch::Tensor w, torch::Tensor out, torch::Tensor sf, int64_t iters, int64_t grid) {
  const int SM = 16 * 8192 + 4 * 8 * 4 * 512 + 1024;
  #define GO(R, NN, CC) if (rt == R && n == NN && c == CC) { auto k = mbc<R, NN, CC>; cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SM); \
    k<<<grid, 256, SM>>>((const uint32_t*)w.data_ptr(), (unsigned long long*)out.data_ptr(), (float*)sf.data_ptr(), (int)iters); return 0; }
  GO(4,8,1) GO(4,32,1) GO(2,64,1) GO(2,64,2) GO(2,32,2) GO(2,32,3) GO(2,32,4) GO(2,128,1) GO(1,64,2) GO(1,64,3) GO(1,64,4) GO(1,128,2) GO(4,32,2)
  return -1;
}
"""

import os
SRC = SRC.replace("@@GEN@@", "".join(gen_rs(n) for n in NS))
lib = load_inline("mb36_" + os.environ.get("TAG","a"), cpp_sources="int64_t run(int64_t, int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t);",
                  cuda_sources="#include <torch/extension.h>\n" + SRC, functions=["run"],
                  extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3", "-Xptxas", "-v", "-maxrregcount=" + os.environ.get("MAXR", "168")], verbose=True)
import fcntl
w = torch.randint(0, 2**31, (4096,), dtype=torch.int32, device="cuda")
G = int(sys.argv[1]) if len(sys.argv) > 1 else 1
out = torch.zeros(132, dtype=torch.int64, device="cuda"); sf = torch.zeros(132 * 256, device="cuda")
IT = 2000
CFGS = [(4, 8, 1), (4, 32, 1), (2, 64, 1), (2, 64, 2), (2, 32, 2), (2, 32, 3), (2, 32, 4), (2, 128, 1), (1, 64, 2), (1, 64, 3), (1, 64, 4), (1, 128, 2), (4, 32, 2)]
for rt, n, c in CFGS:
    assert lib.run(rt, n, c, w, out, sf, 50, G) == 0
    with open("/tmp/okc-gpu.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        lib.run(rt, n, c, w, out, sf, IT, G); torch.cuda.synchronize()
    cyc = out[:G].float().mean().item() / IT
    kb = rt * 4  # KB of W per stage (8 warps x RT x 32 lanes x 16 B)
    tok = n * c
    print(f"RT{rt} N{n} x{c} (acc {rt*c*n//2} regs): {cyc:6.0f} cyc/stage = {cyc/kb*8:6.0f} per 8KB, {cyc/kb*8/tok:5.2f} cyc per 8KB per token", flush=True)
