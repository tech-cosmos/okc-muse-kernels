# s19 microbenchmark: warp-specialized dequant. 2 "T" warpgroups dequantize NVFP4 words (LDS.128 + twm's dq) and store
# the BF16 A tiles to smem (SW128, STS.128) into an A ring; 2 "M" warpgroups run SS wgmma on them. Compare cycles per
# stage with twm's RS-both (s18 mb_wgmma: same warps do dq + RS wgmma): RT4 N8 792, RT4 N32 1046, RT2 N64 737.
# stage = 2 WGs x RT m64 tiles x 64 k (= one twm ring stage). MODE 0 full, 1 T only (M just releases), 2 M only (T no dq/STS).
import torch, sys, fcntl, os
from torch.utils.cpp_extension import load_inline
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "s18"))
def gen_ss(n):
    na = n // 2; dl = ", ".join(f"%{i}" for i in range(na))
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(na))
    return (f"template <> __device__ __forceinline__ void wg_ss<{n}>(float* d, uint64_t da, uint64_t db) {{\n"
            f'  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, %{na + 2}, 0;\\n"\n'
            f'    "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{dl}}}, %{na}, %{na + 1}, p, 1, 1, 0, 0;\\n}}\\n"\n'
            f'    : {outs} : "l"(da), "l"(db), "r"(1));\n}}\n')
NS = (8, 16, 32, 64)
SRC = r"""
#include <cstdint>
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ uint32_t bmul(uint32_t a, uint32_t b) {
  uint32_t r; asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
__device__ __forceinline__ uint4 dq(uint32_t q, uint32_t s0, uint32_t s1) {
  const uint32_t M = 0x81C081C0u;
  const uint32_t o0 = q & M, o1 = __funnelshift_l(q, q, 10) & M, o2 = __funnelshift_l(q, q, 13) & M;
  const uint32_t o3 = ((q & 0x001B001Bu) * 16648u) & M;
  return make_uint4(bmul(o0, s0), bmul(o1, s1), bmul(o2, s0), bmul(o3, s1));
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int K> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(K) : "memory"); }
__device__ __forceinline__ uint64_t sw128_desc(const void* p) {
  return (uint64_t)((smem_u32(p) & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62); }
__device__ __forceinline__ void mb_init(uint64_t* b, int c) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(b)), "r"(c)); }
__device__ __forceinline__ void mb_arrive(uint64_t* b) { asm volatile("{\n.reg .b64 s;\nmbarrier.arrive.shared::cta.b64 s, [%0];\n}" ::"r"(smem_u32(b)) : "memory"); }
__device__ __forceinline__ void mb_wait(uint64_t* b, uint32_t ph) {
  asm volatile("{\n.reg .pred p;\nW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}" ::"r"(smem_u32(b)), "r"(ph) : "memory"); }
template <int N> __device__ __forceinline__ void wg_ss(float* d, uint64_t da, uint64_t db);
@@GEN@@
template <int MODE, int RT, int N, int NA, int NT>
__global__ void __launch_bounds__(128 * (2 + NT), 1) mbws(const uint32_t* __restrict__ w, unsigned long long* out, float* sinkf, int iters) {
  extern __shared__ __align__(1024) char sm[];
  constexpr int ASZ = 2 * RT * 8192;            // one A stage: 2 M WGs x RT tiles x 64 rows x 128 B
  char* sA = sm;                                // NA stages
  char* sB = sA + NA * ASZ;                     // 4 x 8 KB (x tile, SW128 rows)
  char* sW = sB + 4 * 8192;                     // 2 stages x 2 WGs x RT x 512 words
  uint64_t* bar = reinterpret_cast<uint64_t*>(sW + 2 * 2 * RT * 2048);  // full[NA][2], empty[NA][2]
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int MT = 128 * NT;                      // first M thread
  for (int i = tid; i < (NA * ASZ + 4 * 8192 + 2 * 2 * RT * 2048) / 4; i += blockDim.x) reinterpret_cast<uint32_t*>(sm)[i] = w[i & 4095] & 0x3F3F3F3Fu;
  if (tid == 0) for (int i = 0; i < NA * 2; ++i) { mb_init(bar + i, 128 * NT / 2); mb_init(bar + 2 * NA + i, 128); }
  asm volatile("fence.mbarrier_init.release.cluster;\nfence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  if (tid < MT) {  // T: NT warpgroups, NT/2 per M WG
    const int g = tid / (64 * NT);              // which M WG this T thread serves
    const int t = tid % (64 * NT);              // index within the T group (64*NT threads per M WG)
    const uint32_t s0 = 0x3F803F80u ^ (lane & 1), s1 = 0x3F803F80u;
#pragma unroll 1
    for (int it = 0; it < iters; ++it) {
      const int b = it % NA; const uint32_t ph = (it / NA) & 1;
      if (MODE != 2) {
        // weights of this M WG's RT tiles: RT x 64 rows x 64 k = RT*512 words = RT*128 uint4; NT=2 -> 1 uint4 per tile per thread
        uint4 wv[RT * 2 / NT];
        const uint4* ws = reinterpret_cast<const uint4*>(sW + (it & 1) * 2 * RT * 2048 + g * RT * 2048);
#pragma unroll
        for (int r = 0; r < RT * 2 / NT; ++r) wv[r] = ws[r * 64 * NT + t];
        uint4 av[RT * 2 / NT][4];
#pragma unroll
        for (int r = 0; r < RT * 2 / NT; ++r) {
          av[r][0] = dq(wv[r].x ^ it, s0, s1); av[r][1] = dq(wv[r].y ^ it, s0, s1);
          av[r][2] = dq(wv[r].z ^ it, s0, s1); av[r][3] = dq(wv[r].w ^ it, s0, s1);
        }
        mb_wait(bar + 2 * NA + b * 2 + g, ph ^ 1);
        char* ab = sA + b * ASZ + g * RT * 8192;
#pragma unroll
        for (int r = 0; r < RT * 2 / NT; ++r)
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const int e = (r * 64 * NT + t) * 4 + k;       // 16 B chunk index within the M WG's RT*8 KB
            const int row = e >> 3, c = e & 7;
            *reinterpret_cast<uint4*>(ab + row * 128 + ((c ^ (row & 7)) << 4)) = av[r][k];
          }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      } else {
        mb_wait(bar + 2 * NA + b * 2 + g, ph ^ 1);
      }
      mb_arrive(bar + b * 2 + g);
    }
  } else {  // M
    const int g = (tid - MT) >> 7;
    constexpr int NAC = N / 2;
    float acc[RT][NAC];
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int i = 0; i < NAC; ++i) acc[r][i] = 0.f;
    const unsigned long long t0 = clock64();
#pragma unroll 1
    for (int it = 0; it < iters; ++it) {
      const int b = it % NA; const uint32_t ph = (it / NA) & 1;
      mb_wait(bar + b * 2 + g, ph);
      if (MODE != 1) {
        const uint64_t db = sw128_desc(sB + (it & 3) * 8192);
        const char* ab = sA + b * ASZ + g * RT * 8192;
        wg_fence();
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int r = 0; r < RT; ++r) wg_ss<N>(acc[r], sw128_desc(ab + r * 8192) + j * 2, db + j * 2);
        wg_commit();
        wg_wait<1>();
      }
      if (it > 0) mb_arrive(bar + 2 * NA + ((it - 1) % NA) * 2 + g);
    }
    wg_wait<0>();
    const unsigned long long t1 = clock64();
    float s = 0.f;
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int i = 0; i < NAC; ++i) s += acc[r][i];
    sinkf[blockIdx.x * 1024 + tid] = s;
    if (tid == MT) out[blockIdx.x] = t1 - t0;
  }
}
int64_t run(int64_t mode, int64_t rt, int64_t n, int64_t na, int64_t nt, torch::Tensor w, torch::Tensor out, torch::Tensor sf, int64_t iters, int64_t grid) {
  #define GO(M, R, NN, A, T) if (mode == M && rt == R && n == NN && na == A && nt == T) { auto k = mbws<M, R, NN, A, T>; \
    const int SM = A * 2 * R * 8192 + 4 * 8192 + 2 * 2 * R * 2048 + 256; \
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SM); \
    k<<<grid, 128 * (2 + T), SM>>>((const uint32_t*)w.data_ptr(), (unsigned long long*)out.data_ptr(), (float*)sf.data_ptr(), (int)iters); \
    return (int64_t)cudaGetLastError(); }
  #define GOM(R, NN, A, T) GO(0, R, NN, A, T) GO(1, R, NN, A, T) GO(2, R, NN, A, T)
  GOM(4, 8, 2, 2) GOM(4, 16, 2, 2) GOM(4, 32, 2, 2) GOM(2, 64, 3, 2) GOM(2, 64, 4, 2) GOM(2, 32, 4, 2)
  GOM(4, 8, 2, 4) GOM(4, 32, 2, 4) GOM(2, 64, 4, 4)
  return -1;
}
"""
SRC = SRC.replace("@@GEN@@", "".join(gen_ss(n) for n in NS))
lib = load_inline("mb19_ws", cpp_sources="int64_t run(int64_t, int64_t, int64_t, int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t);",
                  cuda_sources="#include <torch/extension.h>\n" + SRC, functions=["run"],
                  extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3", "-Xptxas=-v"], verbose=False)
w = torch.randint(0, 2**31, (4096,), dtype=torch.int32, device="cuda")
G = int(sys.argv[1]) if len(sys.argv) > 1 else 1
out = torch.zeros(132, dtype=torch.int64, device="cuda"); sf = torch.zeros(132 * 1024, device="cuda")
IT = 2000
names = {0: "full", 1: "T only", 2: "M only"}
cfgs = [(4, 8, 2, 2), (4, 16, 2, 2), (4, 32, 2, 2), (2, 32, 4, 2), (2, 64, 3, 2), (2, 64, 4, 2), (4, 8, 2, 4), (4, 32, 2, 4), (2, 64, 4, 4)]
for rt, n, na, nt in cfgs:
    row = []
    for m in (0, 1, 2):
        assert lib.run(m, rt, n, na, nt, w, out, sf, 50, G) == 0
        torch.cuda.synchronize()
        with open("/tmp/okc-gpu.lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            lib.run(m, rt, n, na, nt, w, out, sf, IT, G); torch.cuda.synchronize()
            fcntl.flock(lk, fcntl.LOCK_UN)
        row.append(f"{names[m]} {out[:G].float().mean().item() / IT:6.0f}")
    print(f"RT{rt} N{n} NA{na} NT{nt} cyc/stage: " + " | ".join(row), flush=True)
