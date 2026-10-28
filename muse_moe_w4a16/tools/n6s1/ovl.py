# n6s1: FA3-style overlap premise test. WG1: chain of SS wgmma m64nNk16 (A, B in smem, KG per commit, wait<1>);
# WG0: independent FFMA chains (8 per thread) or int LOP3/SHF chains. MODE 0 = WG0 only, 1 = WG1 only, 2 = both.
# Prints cycles per outer iteration for each WG. Same SM as the H200 (GH100), so the H100 answers it.
import torch, sys, os, fcntl
from torch.utils.cpp_extension import load_inline
def gen_ss(n):
    na = n // 2; dl = ", ".join(f"%{i}" for i in range(na)); outs = ", ".join(f'"+f"(d[{i}])' for i in range(na))
    return (f"template <> __device__ __forceinline__ void wg_ss<{n}>(float* d, uint64_t da, uint64_t db) {{\n"
            f'  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, %{na + 2}, 0;\\n"\n'
            f'    "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{dl}}}, %{na}, %{na + 1}, p, 1, 1, 0, 0;\\n}}\\n"\n'
            f'    : {outs} : "l"(da), "l"(db), "r"(1));\n}}\n')
def gen_rs(n):
    na = n // 2; dl = ", ".join(f"%{i}" for i in range(na)); al = ", ".join(f"%{na + i}" for i in range(4))
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(na))
    return (f"template <> __device__ __forceinline__ void wg_rs<{n}>(float* d, const uint32_t* a, uint64_t desc) {{\n"
            f'  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, %{na + 5}, 0;\\n"\n'
            f'    "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{dl}}}, {{{al}}}, %{na + 4}, p, 1, 1, 0;\\n}}\\n"\n'
            f'    : {outs} : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));\n}}\n')
NS = (32, 64, 128, 256)
SRC = r"""
#include <cstdint>
#ifndef DISTA
#define DISTA 0
#endif
#ifndef NORM
#define NORM 0
#endif
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ uint64_t sw128_desc(const void* p) {
  return (uint64_t)((smem_u32(p) & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62); }
template <int N> __device__ __forceinline__ void wg_ss(float* d, uint64_t da, uint64_t db);
template <int N> __device__ __forceinline__ void wg_rs(float* d, const uint32_t* a, uint64_t desc);
@@GEN@@
template <int MODE, int N, int ALU, int KG, bool RS>
__global__ void __launch_bounds__(256, 1) ov(unsigned long long* out, float* sink, int iters, int alun) {
  extern __shared__ __align__(1024) char sm[];
  const int tid = threadIdx.x, wg = tid >> 7;
  for (int i = tid; i < (64 + 256) * 128 / 4; i += 256) reinterpret_cast<uint32_t*>(sm)[i] = 0x3F803F80u;
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  constexpr int NACC = KG >= 100 ? 4 : 1;  // KG 104: 4 wgmmas per group on 4 DIFFERENT accumulators (twm's RT4 order)
  float acc[NACC][N / 2];
#pragma unroll
  for (int r = 0; r < NACC; ++r)
#pragma unroll
  for (int i = 0; i < N / 2; ++i) acc[r][i] = 0.f;
  float f[8]; uint32_t u[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) { f[i] = tid * 0.001f + i; u[i] = tid * 2654435761u + i; }
  const uint64_t da = sw128_desc(sm), db = sw128_desc(sm + 64 * 128);
  const unsigned long long t0 = clock64();
  if (MODE >= 4 && MODE <= 7) {  // twm pattern: each active WG does dq (alun rounds of 4 x 8-weight dq per k16) -> RS wgmma on those regs.
    // MODE 4: both WGs; MODE 5: only WG1 (WG0 idle). Double-buffered A, wait<1> (as twm).
    if (MODE == 4 || MODE == 6 || wg == 1) {
      uint32_t aC[NACC][4];
#pragma unroll
      for (int r = 0; r < NACC; ++r)
#pragma unroll
        for (int i = 0; i < 4; ++i) aC[r][i] = u[i] | 0x3F803F80u;
      uint32_t aA[NACC][4], aB[NACC][4];
#pragma unroll
      for (int r = 0; r < NACC; ++r)
#pragma unroll
        for (int i = 0; i < 4; ++i) { aA[r][i] = u[i]; aB[r][i] = u[i + 4]; }
      auto dqa = [&](uint32_t (&a)[NACC][4], uint32_t it) {
#pragma unroll
        for (int r = 0; r < NACC; ++r)
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            uint32_t q = a[r][i] ^ it;
            {
              const uint32_t M = 0x81C081C0u;
              const uint32_t o0 = q & M, o1 = __funnelshift_l(q, q, 10) & M, o2 = __funnelshift_l(q, q, 13) & M, o3 = ((q & 0x001B001Bu) * 16648u) & M;
              uint32_t r0, r1, r2, r3, s0 = 0x3F803F80u;
              asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r0) : "r"(o0), "r"(s0)); asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r1) : "r"(o1), "r"(s0));
              asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r2) : "r"(o2), "r"(s0)); asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r3) : "r"(o3), "r"(s0));
              q = (r0 ^ r1) + (r2 ^ r3);
            }
            a[r][i] = NORM ? ((q & 0x007F007Fu) | 0x3F803F80u) : q;  // NORM: normal bf16 in [1, 2)
          }
      };
      auto mma = [&](uint32_t (&a)[NACC][4]) {
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
        for (int k = 0; k < (KG % 100); ++k) wg_rs<N>(acc[k % NACC], MODE >= 6 ? aC[k % NACC] : a[k % NACC], db + (k & 3) * 2);
        asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
      };
      if (MODE == 4 && alun == 9) {  // n6s1: triple-buffered A, dq runs 1 group ahead of the wgmma that reads it
        uint32_t aD[NACC][4];
#pragma unroll
        for (int r = 0; r < NACC; ++r)
#pragma unroll
          for (int i = 0; i < 4; ++i) aD[r][i] = u[i] + 3;
        dqa(aA, 0);
#pragma unroll 1
        for (int it = 0; it < iters; it += 3) {
          dqa(aB, it + 1); mma(aA); asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
          dqa(aD, it + 2); mma(aB); asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
          dqa(aA, it + 3); mma(aD); asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
        }
        asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
#pragma unroll
        for (int r = 0; r < NACC; ++r) u[r & 7] ^= aD[r][1];
      } else
#pragma unroll 1
      for (int it = 0; it < iters; it += 2) {
        dqa(aA, it); mma(aA);
        asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
        dqa(aB, it + 1); mma(aB);
        asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
      }
      asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
#pragma unroll
      for (int r = 0; r < NACC; ++r) u[r & 7] ^= aA[r][0] ^ aB[r][3];
    }
  } else
  if (MODE == 3) {  // SHARED loop: both WGs in one it-loop, branch per iteration (mb_x structure)
#pragma unroll 1
    for (int it = 0; it < iters; ++it) {
      if (wg == 0) {
#pragma unroll 1
      for (int k = 0; k < alun; ++k)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          if (ALU == 0) f[i] = fmaf(f[i], 1.0001f, 0.5f);
          else if (ALU == 1) { u[i] = (u[i] & 0x81C081C0u) ^ __funnelshift_l(u[i], u[i], 10); }
          else if (ALU == 3) {  // exact twm dq (s27 mb_x DQV 0): 2 int masks + 2 rotl + IMAD + 4 HMUL2 bf16x2 per 8 weights
            const uint32_t q = u[i], M = 0x81C081C0u;
            const uint32_t o0 = q & M, o1 = __funnelshift_l(q, q, 10) & M, o2 = __funnelshift_l(q, q, 13) & M, o3 = ((q & 0x001B001Bu) * 16648u) & M;
            uint32_t r0, r1, r2, r3, s0 = 0x3F803F80u ^ (tid & 1);
            asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r0) : "r"(o0), "r"(s0)); asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r1) : "r"(o1), "r"(s0));
            asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r2) : "r"(o2), "r"(s0)); asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r3) : "r"(o3), "r"(s0));
            u[i] = (r0 ^ r1 ^ r2 ^ r3) + q;
          } else if (ALU == 4) {  // IMAD only chains
            u[i] = u[i] * 16648u + 0x1B001Bu;
          } else if (ALU == 5) {  // HMUL2 bf16 only chains
            uint32_t r; asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(u[i]), "r"(0x3F813F81u)); u[i] = r;
          } else {  // LDS.128 every 2 chains + int ops (dq-like smem traffic: 16 B per 32 int ops per thread)
            if ((i & 1) == 0) { const uint4 v = reinterpret_cast<const uint4*>(sm)[((tid + k * 37 + i) & 1023)]; u[i] ^= v.x; u[i + 1] ^= v.y; }
            u[i] = (u[i] & 0x81C081C0u) ^ __funnelshift_l(u[i], u[i], 10);
          }
        }
      } else {
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
        for (int k = 0; k < (KG % 100); ++k) {
          if (RS) { uint32_t a4[4] = {u[0] | 0x3F803F80u, u[1] | 0x3F803F80u, u[2] | 0x3F803F80u, u[3] | 0x3F803F80u}; wg_rs<N>(acc[k % NACC], a4, db + (k & 3) * 2); }
          else wg_ss<N>(acc[k % NACC], da + (k & 3) * 2, db + (k & 3) * 2);
        }
        asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
      }
    }
    if (wg == 1) asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  } else {
  if (wg == 0 && MODE != 1) {
#pragma unroll 1
    for (int it = 0; it < iters; ++it)
#pragma unroll 1
      for (int k = 0; k < alun; ++k)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          if (ALU == 0) f[i] = fmaf(f[i], 1.0001f, 0.5f);
          else if (ALU == 1) { u[i] = (u[i] & 0x81C081C0u) ^ __funnelshift_l(u[i], u[i], 10); }
          else if (ALU == 3) {  // exact twm dq (s27 mb_x DQV 0): 2 int masks + 2 rotl + IMAD + 4 HMUL2 bf16x2 per 8 weights
            const uint32_t q = u[i], M = 0x81C081C0u;
            const uint32_t o0 = q & M, o1 = __funnelshift_l(q, q, 10) & M, o2 = __funnelshift_l(q, q, 13) & M, o3 = ((q & 0x001B001Bu) * 16648u) & M;
            uint32_t r0, r1, r2, r3, s0 = 0x3F803F80u ^ (tid & 1);
            asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r0) : "r"(o0), "r"(s0)); asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r1) : "r"(o1), "r"(s0));
            asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r2) : "r"(o2), "r"(s0)); asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r3) : "r"(o3), "r"(s0));
            u[i] = (r0 ^ r1 ^ r2 ^ r3) + q;
          } else if (ALU == 4) {  // IMAD only chains
            u[i] = u[i] * 16648u + 0x1B001Bu;
          } else if (ALU == 5) {  // HMUL2 bf16 only chains
            uint32_t r; asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(u[i]), "r"(0x3F813F81u)); u[i] = r;
          } else {  // LDS.128 every 2 chains + int ops (dq-like smem traffic: 16 B per 32 int ops per thread)
            if ((i & 1) == 0) { const uint4 v = reinterpret_cast<const uint4*>(sm)[((tid + k * 37 + i) & 1023)]; u[i] ^= v.x; u[i + 1] ^= v.y; }
            u[i] = (u[i] & 0x81C081C0u) ^ __funnelshift_l(u[i], u[i], 10);
          }
        }
  }
  if (wg == 1 && MODE != 0) {
    uint32_t aR[4][4];  // n6s1: 4 DISTINCT constant A sets (DISTA=1) so ptxas can't collapse them into one
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
      for (int i = 0; i < 4; ++i) aR[r][i] = (u[i] | 0x3F803F80u) ^ (DISTA ? (r + 1) << 1 : 0);
#pragma unroll 1
    for (int it = 0; it < iters; ++it) {
      asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
      for (int k = 0; k < (KG % 100); ++k) {
        if (RS) wg_rs<N>(acc[k % NACC], aR[k & 3], db + (k & 3) * 2);
        else wg_ss<N>(acc[k % NACC], da + (k & 3) * 2, db + (k & 3) * 2);
      }
      asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
      asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  }
  }
  const unsigned long long t1 = clock64();
  float s = 0.f;
#pragma unroll
  for (int r = 0; r < NACC; ++r)
#pragma unroll
  for (int i = 0; i < N / 2; ++i) s += acc[r][i];
#pragma unroll
  for (int i = 0; i < 8; ++i) s += f[i] + (float)u[i];
  sink[blockIdx.x * 256 + tid] = s;
  if ((tid & 127) == 0) out[blockIdx.x * 2 + wg] = t1 - t0;
}
int64_t run(int64_t kg, int64_t rs, int64_t mode, int64_t n, int64_t alu, torch::Tensor out, torch::Tensor sink, int64_t iters, int64_t alun, int64_t grid) {
  const int SM = (64 + 256) * 128 + 1024;
  #define GO(M, NN, A) if (mode == M && n == NN && alu == A) { auto k = rs ? (kg == 104 ? ov<M, NN, A, 104, true> : kg == 100 ? ov<M, NN, A, 100, true> : ov<M, NN, A, 4, true>) : (kg == 104 ? ov<M, NN, A, 104, false> : ov<M, NN, A, 4, false>); cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SM); \
    k<<<grid, 256, SM>>>((unsigned long long*)out.data_ptr(), (float*)sink.data_ptr(), (int)iters, (int)alun); return 0; }
  #define GA(NN, A) GO(0, NN, A) GO(1, NN, A) GO(2, NN, A) GO(3, NN, A) GO(4, NN, A) GO(5, NN, A) GO(6, NN, A) GO(7, NN, A)
  GA(32, 0) GA(64, 0) GA(32, 1) GA(64, 1) GA(32, 2) GA(64, 2) GA(32, 3) GA(64, 3) GA(128, 0) GA(256, 0)
  return -1;
}
"""
SRC = SRC.replace("@@GEN@@", "".join(gen_ss(n) + gen_rs(n) for n in NS))
lib = load_inline("n6s1_ovl13n" + os.environ.get("NORM", "0") + "d" + os.environ.get("DISTA", "0"), cpp_sources="int64_t run(int64_t, int64_t, int64_t, int64_t, int64_t, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t);",
                  cuda_sources="#include <torch/extension.h>\n" + SRC, functions=["run"],
                  extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3", "-DNORM=" + os.environ.get("NORM", "0"), "-DDISTA=" + os.environ.get("DISTA", "0")], verbose=False)
G = int(sys.argv[1]) if len(sys.argv) > 1 else 1
out = torch.zeros(2 * 132, dtype=torch.int64, device="cuda"); sink = torch.zeros(132 * 256, device="cuda")
if os.environ.get("TWM"):
    IT = 400
    for n in (32, 64):
        for alun in (1,):
            lib.run(104, 1, 5, n, 0, out, sink, 20, alun, G); torch.cuda.synchronize()
            with open('/tmp/okc-gpu.lock', 'w') as lk:
                fcntl.flock(lk, fcntl.LOCK_EX)
                lib.run(104, 1, 1, n, 0, out, sink, IT, alun, G); torch.cuda.synchronize(); m = out[1].item() / IT  # MMA only (const A)
                lib.run(104, 1, 5, n, 0, out, sink, IT, alun, G); torch.cuda.synchronize(); one = out[1].item() / IT
                lib.run(104, 1, 4, n, 0, out, sink, IT, alun, G); torch.cuda.synchronize(); b0, b1 = out[0].item() / IT, out[1].item() / IT
                lib.run(104, 1, 4, n, 0, out, sink, IT - IT % 3, 9, G); torch.cuda.synchronize(); t2 = max(out[0].item(), out[1].item()) / (IT - IT % 3)
                lib.run(104, 1, 7, n, 0, out, sink, IT, alun, G); torch.cuda.synchronize(); c1 = out[1].item() / IT
                lib.run(104, 1, 6, n, 0, out, sink, IT, alun, G); torch.cuda.synchronize(); c2 = max(out[0].item(), out[1].item()) / IT
                lib.run(100, 1, 5, n, 0, out, sink, IT, alun, G); torch.cuda.synchronize(); d1 = out[1].item() / IT
                lib.run(100, 1, 4, n, 0, out, sink, IT, alun, G); torch.cuda.synchronize(); d2 = max(out[0].item(), out[1].item()) / IT
            print(f"twm-pattern N{n} dq x{alun}: MMA-only WG {m:6.0f} | ONE WG dq+mma {one:6.0f} | TWO WGs dq+mma {b0:6.0f}/{b1:6.0f} "
                  f"| TRIPLE-BUF TWO {t2:5.0f} | CONST-A dq+mma ONE {c1:5.0f} TWO {c2:5.0f} | dq-only ONE WG {d1:5.0f} TWO WGs {d2:5.0f} | TWO dq+mma minus TWO dq-only = {max(b0, b1) - d2:5.0f} vs 2 x MMA {2 * m:5.0f}", flush=True)
    sys.exit(0)
IT = 400
RS = int(os.environ.get("RS", "0")); KG = int(os.environ.get("KG", "4"))
for alu in [int(a) for a in os.environ.get("ALUS", "0,1,2").split(",")]:
    for n in [int(x) for x in os.environ.get("NN", "32,64,128,256").split(",")]:
        # alun chosen so WG0 alone ~ WG1 alone: measure WG1 first, then scale
        lib.run(KG, RS, 1, n, alu, out, sink, 20, 1, G); torch.cuda.synchronize()
        with open('/tmp/okc-gpu.lock', 'w') as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            lib.run(KG, RS, 1, n, alu, out, sink, IT, 1, G); torch.cuda.synchronize(); m1 = out[1].item() / IT
            lib.run(KG, RS, 0, n, alu, out, sink, IT, 8, G); torch.cuda.synchronize(); a8 = out[0].item() / IT
            alun = max(1, round(8 * m1 / a8))
            lib.run(KG, RS, 0, n, alu, out, sink, IT, alun, G); torch.cuda.synchronize(); a = out[0].item() / IT
            lib.run(KG, RS, 2, n, alu, out, sink, IT, alun, G); torch.cuda.synchronize(); b0, b1 = out[0].item() / IT, out[1].item() / IT
            lib.run(KG, RS, 3, n, alu, out, sink, IT, alun, G); torch.cuda.synchronize(); c0, c1 = out[0].item() / IT, out[1].item() / IT
        print(f"{['FFMA', 'INT ', 'LDS+INT', 'DQ', 'IMAD', 'HMUL2'][alu]} {'RS' if RS else 'SS'} KG{KG} N{n:3d}: WG0 alone {a:7.0f} | WG1 SS-MMA alone {m1:7.0f} | both WG0 {b0:7.0f} WG1 {b1:7.0f} "
              f"| overlap {(a + m1 - max(b0, b1)) / min(a, m1) * 100:5.1f}% of the smaller | SHARED-LOOP both {c0:5.0f}/{c1:5.0f} overlap {(a + m1 - max(c0, c1)) / min(a, m1) * 100:5.1f}%", flush=True)
