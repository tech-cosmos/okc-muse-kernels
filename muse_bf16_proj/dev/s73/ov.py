# s73 proj: why don't ALU (WG0) and RS wgmma (WG1) overlap? variants of the MMA side / ALU side.
import torch, sys, os, fcntl
from torch.utils.cpp_extension import load_inline
def gen(n, acc):  # acc: 'f32' (bf16 in) or 'f16' (f16 in, f16 acc)
    na = n // 2 if acc == 'f32' else n // 4
    dl = ", ".join(f"%{i}" for i in range(na)); al = ", ".join(f"%{na + i}" for i in range(4))
    ty = "f32.bf16.bf16" if acc == 'f32' else "f16.f16.f16"
    c = "f" if acc == 'f32' else "r"; T = "float" if acc == 'f32' else "uint32_t"
    outs = ", ".join(f'"+{c}"(d[{i}])' for i in range(na))
    return (f"template <> __device__ __forceinline__ void wg_{acc}<{n}>({T}* d, const uint32_t* a, uint64_t desc) {{\n"
            f'  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, %{na + 5}, 0;\\n"\n'
            f'    "wgmma.mma_async.sync.aligned.m64n{n}k16.{ty} {{{dl}}}, {{{al}}}, %{na + 4}, p, 1, 1, 0;\\n}}\\n"\n'
            f'    : {outs} : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));\n}}\n')
SRC = r"""
#include <cstdint>
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ uint64_t sw128_desc(const void* p) {
  return (uint64_t)((smem_u32(p) & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62); }
template <int N> __device__ __forceinline__ void wg_f32(float* d, const uint32_t* a, uint64_t desc);
template <int N> __device__ __forceinline__ void wg_f16(uint32_t* d, const uint32_t* a, uint64_t desc);
@@GEN@@
// MMA side (WG1): per iter, RT tiles x 4 k16 steps, one commit, wait<1>.  ORD 0 = j outer (acc alternates), 1 = r outer.
// ALU side (WG0): ALUK = 0 none, 1 = 8 independent LOP3/IADD chains (ILP 8), 2 = FFMA chains (ILP 8), per iter ALUN ops/chain
// WHO: 0 both, 1 ALU only, 2 MMA only
template <int RT, int N, int ACC, int ORD, int ALUK, int ALUN>
__global__ void __launch_bounds__(256, 1) mb(unsigned long long* out, uint32_t* sink, int iters, int who) {
  extern __shared__ __align__(1024) char sm[];
  const int tid = threadIdx.x, wg = tid >> 7;
  for (int i = tid; i < 4 * 8192 / 4; i += 256) reinterpret_cast<uint32_t*>(sm)[i] = 0x3C003C00u;
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  constexpr int NA = ACC == 0 ? N / 2 : N / 4;
  float accf[RT][ACC == 0 ? NA : 1]; uint32_t acch[RT][ACC == 1 ? NA : 1];
  for (int r = 0; r < RT; ++r) { for (int i = 0; i < (ACC == 0 ? NA : 1); ++i) accf[r][i] = 0.f; for (int i = 0; i < (ACC == 1 ? NA : 1); ++i) acch[r][i] = 0; }
  uint32_t af[4] = {0x3C003C00u ^ (tid & 1), 0x3C003C00u, 0x3C003C00u, 0x3C003C00u};
  uint32_t x[8]; float f[8];
  for (int i = 0; i < 8; ++i) { x[i] = tid * 7 + i; f[i] = (float)(tid + i); }
  const unsigned long long t0 = clock64();
  if (wg == 1 && who != 1) {
#pragma unroll 1
    for (int it = 0; it < iters; ++it) {
      const uint64_t d0 = sw128_desc(sm + (it & 3) * 8192);
      asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
      if (ORD == 0) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int r = 0; r < RT; ++r) { if (ACC == 0) wg_f32<N>(accf[r], af, d0 + j * 2); else wg_f16<N>(acch[r], af, d0 + j * 2); }
      } else {
#pragma unroll
        for (int r = 0; r < RT; ++r)
#pragma unroll
          for (int j = 0; j < 4; ++j) { if (ACC == 0) wg_f32<N>(accf[r], af, d0 + j * 2); else wg_f16<N>(acch[r], af, d0 + j * 2); }
      }
      asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
      asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    float fs = 0.f;
    for (int r = 0; r < RT; ++r) { if (ACC == 0) for (int i = 0; i < NA; ++i) fs += accf[r][i]; else for (int i = 0; i < NA; ++i) x[0] ^= acch[r][i]; }
    f[0] += fs;
  }
  if (wg == 0 && who != 2 && ALUK) {
#pragma unroll 1
    for (int it = 0; it < iters; ++it) {
#pragma unroll
      for (int k = 0; k < ALUN; ++k)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          if (ALUK == 1) x[i] = (__funnelshift_l(x[i], x[i], 10) & 0x81C081C0u) ^ (x[i] + k);
          else f[i] = fmaf(f[i], 1.0001f, 0.5f);
        }
    }
  }
  const unsigned long long t1 = clock64();
  uint32_t s = 0;
  for (int i = 0; i < 8; ++i) s ^= x[i] ^ __float_as_uint(f[i]);
  sink[blockIdx.x * 256 + tid] = s;
  if ((tid & 127) == 0) out[blockIdx.x * 2 + wg] = t1 - t0;
}
int64_t run(int64_t cfg, torch::Tensor out, torch::Tensor sink, int64_t iters, int64_t grid, int64_t who) {
  const int SM = 4 * 8192 + 1024;
  #define GO(RT, N, ACC, ORD, AK, AN) if (cfg == RT * 100000 + N * 1000 + ACC * 100 + ORD * 10 + AK) { auto k = mb<RT, N, ACC, ORD, AK, AN>; \
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SM); \
    k<<<grid, 256, SM>>>((unsigned long long*)out.data_ptr(), (uint32_t*)sink.data_ptr(), (int)iters, (int)who); return 0; }
  @@INST@@
  return -1;
}
"""
AN = int(os.environ.get("ALUN", "4"))
CFGS = []
for rt, n in ((2, 64), (4, 32), (4, 8)):
    for acc in (0, 1):
        for ordr in (0, 1):
            for ak in (1, 2):
                CFGS.append((rt, n, acc, ordr, ak))
inst = "".join(f"GO({rt},{n},{acc},{o},{ak},{AN})\n" for rt, n, acc, o, ak in CFGS)
SRC = SRC.replace("@@GEN@@", "".join(gen(n, a) for n in (8, 32, 64) for a in ('f32', 'f16'))).replace("@@INST@@", inst)
lib = load_inline(f"pj30_ov2_{AN}", cpp_sources="int64_t run(int64_t, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t);",
                  cuda_sources="#include <torch/extension.h>\n" + SRC, functions=["run"],
                  extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"], verbose=False)
G = int(sys.argv[1]) if len(sys.argv) > 1 else 132
out = torch.zeros(264, dtype=torch.int64, device="cuda"); sink = torch.zeros(132 * 256, dtype=torch.int32, device="cuda")
IT = 2000
def t(cfg, who):
    assert lib.run(cfg, out, sink, 20, G, who) == 0; torch.cuda.synchronize(); out.zero_()
    with open('/tmp/okc-gpu.lock', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX); lib.run(cfg, out, sink, IT, G, who); torch.cuda.synchronize()
    o = out[:2 * G].view(G, 2).float().mean(0) / IT
    return o[0].item(), o[1].item()
for rt, n, acc, o, ak in CFGS:
    cfg = rt * 100000 + n * 1000 + acc * 100 + o * 10 + ak
    a = t(cfg, 1)[0]; m = t(cfg, 2)[1]; b0, b1 = t(cfg, 0)
    print(f"RT{rt} N{n:2d} {'f32' if acc == 0 else 'f16'} ord{o} {'LOP' if ak == 1 else 'FMA'}x{AN}: ALU {a:5.0f} MMA {m:5.0f} both {b0:5.0f}/{b1:5.0f}  overlap={(a + m - max(b0, b1)) / min(a, m):4.2f}", flush=True)
