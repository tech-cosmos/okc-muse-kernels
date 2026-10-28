# n6s2: what limits twm's "dq -> RS wgmma on fresh registers" pattern? NWG consumer WGs each run the same loop:
# per iteration update 4 A quads (UPD 0: none (distinct A loaded from smem once), 1: one XOR per reg, 2: exact twm dq
# 12 ops / 8 w, 3: dq but into the OTHER buffer one group ahead), then 4 RS wgmma m64nNk16 on 4 accumulators,
# commit, wait<1>. Double-buffered A. Prints cycles per iteration (max over WGs) and checks the SASS for serialization.
import torch, sys, os, fcntl, subprocess, glob
from torch.utils.cpp_extension import load_inline
def gen_rs(n):
    na = n // 2; dl = ", ".join(f"%{i}" for i in range(na)); al = ", ".join(f"%{na + i}" for i in range(4))
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(na))
    return (f"template <> __device__ __forceinline__ void wg_rs<{n}>(float* d, const uint32_t* a, uint64_t desc) {{\n"
            f'  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, %{na + 5}, 0;\\n"\n'
            f'    "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{dl}}}, {{{al}}}, %{na + 4}, p, 1, 1, 0;\\n}}\\n"\n'
            f'    : {outs} : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));\n}}\n')
SRC = r"""
#include <cstdint>
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ uint64_t sw128_desc(const void* p) {
  return (uint64_t)((smem_u32(p) & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62); }
template <int N> __device__ __forceinline__ void wg_rs(float* d, const uint32_t* a, uint64_t desc);
@@GEN@@
__device__ __forceinline__ uint32_t dq1(uint32_t q, uint32_t s0) {
  const uint32_t M = 0x81C081C0u;
  const uint32_t o0 = q & M, o1 = __funnelshift_l(q, q, 10) & M, o2 = __funnelshift_l(q, q, 13) & M, o3 = ((q & 0x001B001Bu) * 16648u) & M;
  uint32_t r0, r1, r2, r3;
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r0) : "r"(o0), "r"(s0)); asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r1) : "r"(o1), "r"(s0));
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r2) : "r"(o2), "r"(s0)); asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r3) : "r"(o3), "r"(s0));
  // twm consumes all 4 outputs as 4 A regs; here one A reg per dq so the 4:1 output fan-in is a cheap merge (2 extra ops)
  return (r0 ^ r1) + (r2 ^ r3);
}
template <int NWG, int N, int UPD>
__global__ void __launch_bounds__(NWG * 128, 1) rsa(unsigned long long* out, float* sink, int iters) {
  extern __shared__ __align__(1024) char sm[];
  const int tid = threadIdx.x, wg = tid >> 7;
  for (int i = tid; i < (64 + 256) * 128 / 4; i += NWG * 128) reinterpret_cast<uint32_t*>(sm)[i] = 0x3F803F80u ^ (i * 2654435761u & 0x00070007u);
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  float acc[4][N / 2];
#pragma unroll
  for (int r = 0; r < 4; ++r)
#pragma unroll
    for (int i = 0; i < N / 2; ++i) acc[r][i] = 0.f;
  uint32_t aA[4][4], aB[4][4];
  const volatile uint32_t* vs = reinterpret_cast<const volatile uint32_t*>(sm);
#pragma unroll
  for (int r = 0; r < 4; ++r)
#pragma unroll
    for (int i = 0; i < 4; ++i) { aA[r][i] = vs[(tid * 16 + r * 4 + i) & 4095]; aB[r][i] = vs[(tid * 16 + r * 4 + i + 7) & 4095]; }
  const uint64_t db = sw128_desc(sm + 64 * 128);
  const uint32_t s0 = 0x3F803F80u ^ (tid & 1);
  uint32_t sk[4][4];
#pragma unroll
  for (int r = 0; r < 4; ++r)
#pragma unroll
    for (int i = 0; i < 4; ++i) sk[r][i] = vs[(tid * 16 + r * 4 + i + 3) & 4095];
  auto upd = [&](uint32_t (&a)[4][4], uint32_t it) {
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        if (UPD == 1) a[r][i] ^= it;
        else if (UPD == 3 || UPD == 7) sk[r][i] = dq1(sk[r][i] ^ it, s0);
        else if (UPD >= 2) a[r][i] = dq1(a[r][i] ^ it, s0);
      }
  };
  auto mma = [&](uint32_t (&a)[4][4]) {
    if (UPD == 5) return;
    if (UPD == 6 || UPD == 7) {
#pragma unroll
      for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int i = 0; i < 4; ++i) { asm volatile("" : "+r"(a[r][i])); asm volatile("" : "+r"(sk[r][i])); }
      return;
    }
      asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
    for (int k = 0; k < 4; ++k) wg_rs<N>(acc[k], a[k], db + k * 2);
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
      for (int i = 0; i < 4; ++i) asm volatile("" : "+r"(sk[r][i]));
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
  };
  __syncthreads();
  const unsigned long long t0 = clock64();
#pragma unroll 1
  for (int it = 0; it < iters; it += 2) {
    upd(aA, it); mma(aA); asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
    upd(aB, it + 1); mma(aB); asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  const unsigned long long t1 = clock64();
  float s = 0.f;
#pragma unroll
  for (int r = 0; r < 4; ++r)
#pragma unroll
    for (int i = 0; i < N / 2; ++i) s += acc[r][i];
#pragma unroll
  for (int r = 0; r < 4; ++r) s += (float)(aA[r][0] ^ aB[r][3] ^ sk[r][1] ^ sk[r][2]);
  sink[blockIdx.x * 512 + tid] = s;
  if ((tid & 127) == 0) out[blockIdx.x * 4 + wg] = t1 - t0;
}
int64_t run(int64_t nwg, int64_t n, int64_t upd, torch::Tensor out, torch::Tensor sink, int64_t iters, int64_t grid) {
  const int SM = (64 + 256) * 128 + 1024;
  #define GO(W, NN, U) if (nwg == W && n == NN && upd == U) { auto k = rsa<W, NN, U>; cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SM); \
    k<<<grid, W * 128, SM>>>((unsigned long long*)out.data_ptr(), (float*)sink.data_ptr(), (int)iters); return 0; }
  #define GU(W, NN) GO(W, NN, 0) GO(W, NN, 2) GO(W, NN, 3) GO(W, NN, 6) GO(W, NN, 7)
  GU(1, 32) GU(2, 32) GU(3, 32) GU(1, 64) GU(2, 64) GU(3, 64)
  return -1;
}
"""
SRC = SRC.replace("@@GEN@@", "".join(gen_rs(n) for n in (32, 64)))
NAME = "n6s2_rsa3"
lib = load_inline(NAME, cpp_sources="int64_t run(int64_t, int64_t, int64_t, torch::Tensor, torch::Tensor, int64_t, int64_t);",
                  cuda_sources="#include <torch/extension.h>\n" + SRC, functions=["run"],
                  extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"], verbose=False)
so = glob.glob(os.path.expanduser(f"~/.cache/torch_extensions/*/{NAME}/{NAME}.so"))[0]
sass = subprocess.run(["/usr/local/cuda-13.0/bin/cuobjdump", "-sass", so], capture_output=True, text=True).stdout
ser = {}
for blk in sass.split("Function : ")[1:]:
    name = blk.split()[0]; ser[name] = blk.count("DEPBAR.LE gsb0, 0x0")
G = int(sys.argv[1]) if len(sys.argv) > 1 else 1
out = torch.zeros(4 * 132, dtype=torch.int64, device="cuda"); sink = torch.zeros(132 * 512, device="cuda")
IT = 400
for n in (32, 64):
    for upd in [int(x) for x in os.environ.get('UPDS', '0,2,6,3,7').split(',')]:
        row = []
        for nwg in (1, 2, 3):
            lib.run(nwg, n, upd, out, sink, 20, G); torch.cuda.synchronize()
            with open('/tmp/okc-gpu.lock', 'w') as lk:
                fcntl.flock(lk, fcntl.LOCK_EX)
                lib.run(nwg, n, upd, out, sink, IT, G); torch.cuda.synchronize()
            c = max(out[i].item() for i in range(nwg)) / IT
            nm = f"_Z3rsaILi{nwg}ELi{n}ELi{upd}EEvPyPfi"
            row.append(f"{nwg}WG {c:6.0f} ({c / nwg:5.0f}/WG, ser0={ser.get(nm, '?')})")
        print(f"N{n} UPD{upd} {['MMA only', 'xor', 'dq->A + MMA(A)', 'dq->sk + MMA(constA)', '', 'dq-no-mma', 'dq->A, no MMA', 'dq->sk, no MMA'][upd]:22s}: " + " | ".join(row), flush=True)
