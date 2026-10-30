# fresh x3 (rfport step 1): LOP3 register-source-count variants next to HGMMA. Derived from conc.py (fresh x1).
# Roles 7-11 (WG1, 112 LOP3/stage each): 7 = 3 regs (a), 8 = 2 regs + imm (b; = role 3), 9 = 1 reg + c[] + RZ (c),
# 10 = 3 regs, b/c slots the same register for 16 back-to-back instrs (d, reuse), 11 = 2 regs + imm, b slot reused (e).
# was x1: does an HGMMA in flight (WG0) slow pure ALU/FMA work in another WG (WG1) on the same SMs? RS vs SS, N sweep.
# Roles per WG: 0 idle, 1 HGMMA RS loop (A const in regs), 2 HGMMA SS loop (A from smem), 3 LOP3 loop, 4 HMUL2 loop,
# 5 dq-mix loop (twm's dq: 5 LOP3 + 2 SHF + 1 IMAD + 4 HMUL2 per 8 weights), 6 HGMMA RS + its own dq (twm-like, one WG).
import torch, os, fcntl, sys
from torch.utils.cpp_extension import load_inline
NS = [int(x) for x in os.environ.get("NS", "8,32,64").split(",")]
def gen(n):
    na = n // 2; dl = ", ".join(f"%{i}" for i in range(na)); outs = ", ".join(f'"+f"(d[{i}])' for i in range(na))
    rs = (f"template <> __device__ __forceinline__ void wrs<{n}>(float* d, const uint32_t* a, uint64_t b) {{\n"
          f'  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, 1, 0;\\n"\n'
          f'    "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{dl}}}, {{%{na},%{na+1},%{na+2},%{na+3}}}, %{na+4}, p, 1, 1, 0;\\n}}\\n"\n'
          f'    : {outs} : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b));\n}}\n')
    ss = (f"template <> __device__ __forceinline__ void wss<{n}>(float* d, uint64_t a, uint64_t b) {{\n"
          f'  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, 1, 0;\\n"\n'
          f'    "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{dl}}}, %{na}, %{na+1}, p, 1, 1, 0, 0;\\n}}\\n"\n'
          f'    : {outs} : "l"(a), "l"(b));\n}}\n')
    return rs + ss
SRC = r"""
#include <cstdint>
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ uint64_t sw128_desc(const void* p) {
  return (uint64_t)((smem_u32(p) & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62); }
template <int N> __device__ __forceinline__ void wrs(float* d, const uint32_t* a, uint64_t b);
template <int N> __device__ __forceinline__ void wss(float* d, uint64_t a, uint64_t b);
@@GEN@@
__device__ __forceinline__ void fence_r(uint32_t& r) { asm volatile("" : "+r"(r)::"memory"); }
__device__ __forceinline__ uint32_t bmul(uint32_t a, uint32_t b) { uint32_t d; asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b)); return d; }
__device__ __forceinline__ void dq(uint32_t q, uint32_t s0, uint32_t s1, uint32_t* a) {
  const uint32_t M = 0x81C081C0u;
  const uint32_t o0 = q & M, o1 = __funnelshift_l(q, q, 10) & M, o2 = __funnelshift_l(q, q, 13) & M;
  const uint32_t o3 = ((q & 0x001B001Bu) * 16648u) & M;
  a[0] = bmul(o0, s0); a[1] = bmul(o1, s1); a[2] = bmul(o2, s0); a[3] = bmul(o3, s1);
}
// one "stage" = RT 4 x 4 k16 = 16 HGMMA m64nNk16 per WG (= the twm RT4 stage); ALU roles: 16 x the dq's instruction count
template <int ROLE, int N>
__device__ unsigned long long body(const char* sm, int nst, uint32_t* sink, const uint32_t m0, const uint32_t m1, const uint32_t m2, const uint32_t m3, const uint32_t m4, const uint32_t m5, const uint32_t m6) {
  constexpr int RT = 4;
  float acc[RT][N / 2];
  for (int r = 0; r < RT; ++r) for (int i = 0; i < N / 2; ++i) acc[r][i] = 0.f;
  uint32_t af[2][RT][4];
  const int lane = threadIdx.x & 31, warp = (threadIdx.x >> 5) & 3;
  for (int d = 0; d < 2; ++d) for (int r = 0; r < RT; ++r) for (int i = 0; i < 4; ++i) af[d][r][i] = 0x3C003C00u ^ (lane << 2) ^ i;
  uint32_t v[16];
  for (int i = 0; i < 16; ++i) v[i] = threadIdx.x * 7 + i;
  uint32_t sv[7], tv[7];
  for (int u = 0; u < 7; ++u) { sv[u] = threadIdx.x * 0x9E3779B1u + u * 0x85EBCA6Bu; tv[u] = (threadIdx.x << 3) ^ (u * 0xC2B2AE35u); }
  const uint64_t bd = sw128_desc(sm), ad = sw128_desc(sm + 16384);
  __syncwarp();
  const unsigned long long t0 = clock64();
  if (ROLE != 0) {
#pragma unroll 1
  for (int kc = 0; kc < nst; ++kc) {
    if (ROLE == 1 || ROLE == 2 || ROLE == 6) {
#pragma unroll
      for (int j0 = 0; j0 < 4; ++j0) {
        if (ROLE == 6) {
          const uint4 wv = reinterpret_cast<const uint4*>(sm + 32768)[(warp * 4 + j0) * 32 + lane];
          dq(wv.x, v[0], v[1], af[j0 & 1][0]); dq(wv.y, v[0], v[1], af[j0 & 1][1]);
          dq(wv.z, v[2], v[3], af[j0 & 1][2]); dq(wv.w, v[2], v[3], af[j0 & 1][3]);
        }
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
        for (int r = 0; r < RT; ++r) {
          if (ROLE == 2) wss<N>(acc[r], ad + r * 128, bd + j0 * 2);
          else wrs<N>(acc[r], af[j0 & 1][r], bd + j0 * 2);
        }
        asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
        for (int r = 0; r < RT; ++r) for (int i = 0; i < 4; ++i) fence_r(af[(j0 + 1) & 1][r][i]);
      }
    } else if (ROLE == 3) {  // 16 dq x 7 ALU instrs = 112 LOP3 per stage
#pragma unroll
      for (int u = 0; u < 7; ++u)
#pragma unroll
        for (int i = 0; i < 16; ++i) asm volatile("lop3.b32 %0, %0, %1, 0x81c081c0, 0x96;" : "+r"(v[i]) : "r"(v[(i + 1) & 15]));
    } else if (ROLE == 4) {  // 64 HMUL2 per stage
#pragma unroll
      for (int u = 0; u < 4; ++u)
#pragma unroll
        for (int i = 0; i < 16; ++i) asm volatile("mul.rn.bf16x2 %0, %0, %1;" : "+r"(v[i]) : "r"(v[(i + 3) & 15]));
    } else if (ROLE == 7) {
#pragma unroll
      for (int u = 0; u < 7; ++u)
#pragma unroll
        for (int i = 0; i < 16; ++i) asm volatile("lop3.b32 %0, %0, %1, %2, 0xe8;" : "+r"(v[i]) : "r"(v[(i + 1) & 15]), "r"(v[(i + 7) & 15]));
    } else if (ROLE == 8) {
#pragma unroll
      for (int u = 0; u < 7; ++u)
#pragma unroll
        for (int i = 0; i < 16; ++i) asm volatile("lop3.b32 %0, %0, %1, 0x81c081c0, 0x96;" : "+r"(v[i]) : "r"(v[(i + 1) & 15]));
    } else if (ROLE == 9) {
      const uint32_t mm[7] = {m0, m1, m2, m3, m4, m5, m6};
#pragma unroll
      for (int u = 0; u < 7; ++u)
#pragma unroll
        for (int i = 0; i < 16; ++i) asm volatile("lop3.b32 %0, %0, %1, 0, 0x6c;" : "+r"(v[i]) : "r"(mm[u]));
    } else if (ROLE == 10) {
#pragma unroll
      for (int u = 0; u < 7; ++u)
#pragma unroll
        for (int i = 0; i < 16; ++i) asm volatile("lop3.b32 %0, %0, %1, %2, 0xe8;" : "+r"(v[i]) : "r"(sv[u]), "r"(tv[u]));
    } else if (ROLE == 11) {
#pragma unroll
      for (int u = 0; u < 7; ++u)
#pragma unroll
        for (int i = 0; i < 16; ++i) asm volatile("lop3.b32 %0, %0, %1, 0x81c081c0, 0xe8;" : "+r"(v[i]) : "r"(sv[u]));
    } else if (ROLE == 5) {  // 16 dq per stage from registers (no LDS), outputs folded so nothing is dead
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        uint32_t a[4][4];
#pragma unroll
        for (int r = 0; r < 4; ++r) dq(v[r * 4 + u] ^ kc, v[(r + 1) & 15], v[(r + 2) & 15], a[r]);
#pragma unroll
        for (int r = 0; r < 4; ++r) for (int i = 0; i < 4; ++i) fence_r(a[r][i]);
        v[u] ^= a[0][0]; v[u + 4] ^= a[1][1]; v[u + 8] ^= a[2][2]; v[u + 12] ^= a[3][3];
      }
    }
  }
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  const unsigned long long t1 = clock64();
  uint32_t s = 0;
  for (int r = 0; r < RT; ++r) for (int i = 0; i < N / 2; ++i) s ^= __float_as_uint(acc[r][i]);
  for (int i = 0; i < 16; ++i) s ^= v[i];
  for (int u = 0; u < 7; ++u) s ^= sv[u] + tv[u];
  for (int d = 0; d < 2; ++d) for (int r = 0; r < RT; ++r) s ^= af[d][r][0] ^ af[d][r][3];
  sink[threadIdx.x] = s;
  return t1 - t0;
}
template <int R0, int R1, int N>
__global__ void __launch_bounds__(256, 1) conc(unsigned long long* out, uint32_t* sink, int n0, int n1, uint32_t m0, uint32_t m1, uint32_t m2, uint32_t m3, uint32_t m4, uint32_t m5, uint32_t m6) {
  extern __shared__ __align__(1024) char sm[];
  for (int i = threadIdx.x; i < 65536 / 4; i += 256) reinterpret_cast<uint32_t*>(sm)[i] = (i * 2654435761u) & 0x3F3F3F3Fu;
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  const int wg = threadIdx.x >> 7;
  unsigned long long t = wg == 0 ? body<R0, N>(sm, n0, sink, m0, m1, m2, m3, m4, m5, m6) : body<R1, N>(sm, n1, sink, m0, m1, m2, m3, m4, m5, m6);
  if ((threadIdx.x & 127) == 0) out[wg] = t;
}
int64_t run(int64_t cfg, torch::Tensor out, torch::Tensor sink, int64_t n0, int64_t n1) {
  int c = 0;
  #define GO(R0, R1, NN) if (cfg == c++) { auto k = conc<R0, R1, NN>; \
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536); \
    k<<<1, 256, 65536>>>((unsigned long long*)out.data_ptr(), (uint32_t*)sink.data_ptr(), (int)n0, (int)n1, 0x81c081c0u, 0x12345678u, 0x0f0f0f0fu, 0x33cc33ccu, 0x5a5aa5a5u, 0x76543210u, 0xdeadbeefu); return 0; }
  @@CFGS@@
  return -1;
}
"""
PAIRS = [(0, r) for r in (7, 8, 9, 10, 11)] + [(1, r) for r in (7, 8, 9, 10, 11)] + [(1, 0)]
NAMES = {0: 'idle', 1: 'HGMMA-RS', 2: 'HGMMA-SS', 3: 'LOP3x112', 4: 'HMUL2x64', 5: 'dq16', 6: 'dq+RS (twm)', 7: 'a:3reg', 8: 'b:2reg+imm', 9: 'c:1reg+c[]', 10: 'd:3reg reuse', 11: 'e:2reg+imm reuse'}
ALL = [(a, b, n) for n in NS for (a, b) in PAIRS]
SRC = SRC.replace("@@GEN@@", "".join(gen(n) for n in NS)).replace("@@CFGS@@", "\n  ".join(f"GO({a}, {b}, {n})" for a, b, n in ALL))
lib = load_inline("fresh_conc4_" + os.environ.get("TAG", "a"), cpp_sources="int64_t run(int64_t, torch::Tensor, torch::Tensor, int64_t, int64_t);",
                  cuda_sources="#include <torch/extension.h>\n" + SRC, functions=["run"],
                  extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"])
out = torch.zeros(2, dtype=torch.int64, device="cuda"); sink = torch.zeros(256, dtype=torch.int32, device="cuda")
NST = int(os.environ.get("NST", "400"))
for ci, (a, b, n) in enumerate(ALL):
    lib.run(ci, out, sink, 8, 8); torch.cuda.synchronize()
    best = None
    for _ in range(3):
        with open('/tmp/okc-gpu.lock', 'w') as lk:
            fcntl.flock(lk, fcntl.LOCK_EX); lib.run(ci, out, sink, NST, NST); torch.cuda.synchronize()
        v = [out[i].item() / NST for i in range(2)]
        if best is None or max(v) < max(best): best = v
    print(f"N{n:3d} WG0 {NAMES[a]:12s} {best[0]:7.1f} | WG1 {NAMES[b]:12s} {best[1]:7.1f} cyc/stage", flush=True)
