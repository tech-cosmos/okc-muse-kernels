# n6s2: twm stage loop microbench (tw_item structure: RT row tiles, KG 1, 4 groups/stage of RT dq -> fence -> RT RS wgmma
# m64nNk16 -> commit -> wait<D-1>), production dq(), weights + scales LDS'd per stage from 8 smem stages (static data, no
# mbarriers), x tile (B) from smem. 384 threads: 2 consumer WGs (setmaxnreg.inc 232) + 1 idle producer WG (dec 40), as twm.
# D = A-ring depth: 2 (production) or 4 (slot = group index j0, wait<3>). MODE 0 both, 1 dq only, 2 mma only.
# Prints cycles/stage (max over consumer WGs) and the loop's DEPBAR 0x0 count (serialization check) + ptxas C7512.
import torch, os, sys, fcntl, subprocess, glob, re
from torch.utils.cpp_extension import load_inline
def gen_rs(n):
    na = n // 2; dl = ", ".join(f"%{i}" for i in range(na)); al = ", ".join(f"%{na + i}" for i in range(4))
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(na))
    return (f"template <> __device__ __forceinline__ void wg_rs<{n}>(float* d, const uint32_t* a, uint64_t desc) {{\n"
            f'  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, %{na + 5}, 0;\\n"\n'
            f'    "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{dl}}}, {{{al}}}, %{na + 4}, p, 1, 1, 0;\\n}}\\n"\n'
            f'    : {outs} : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));\n}}\n')
NS = (8, 16, 32)
SRC = r"""
#include <cstdint>
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ uint64_t sw128_desc(const void* p) {
  return (uint64_t)((smem_u32(p) & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62); }
template <int N> __device__ __forceinline__ void wg_rs(float* d, const uint32_t* a, uint64_t desc);
@@GEN@@
template <int K> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(K) : "memory"); }
__device__ __forceinline__ void fence_r(uint32_t& r) { asm volatile("" : "+r"(r)::"memory"); }
__device__ __forceinline__ uint32_t bmul(uint32_t a, uint32_t b) {
  uint32_t d; asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u)); return d; }
__device__ __forceinline__ void dq(uint32_t q, uint32_t s0, uint32_t s1, uint32_t* a) {
  const uint32_t M = 0x81C081C0u;
  const uint32_t o0 = q & M, o1 = __funnelshift_l(q, q, 10) & M, o2 = __funnelshift_l(q, q, 13) & M;
  const uint32_t o3 = ((q & 0x001B001Bu) * 16648u) & M;
  a[0] = bmul(o0, s0); a[1] = bmul(o1, s1); a[2] = bmul(o2, s0); a[3] = bmul(o3, s1);
}
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t u4get(const uint4& v, int j) { return j == 0 ? v.x : j == 1 ? v.y : j == 2 ? v.z : v.w; }
constexpr int NSTG = 8;
template <int MODE, int RT, int N, int D>
__global__ void __launch_bounds__(384, 1) ring(unsigned long long* out, float* sink, int nst) {
  extern __shared__ __align__(1024) char sm[];
  // per stage: x tile 64 k x N rows bf16 SW128 (N * 128 B, padded to 4 KB) + weights 8 warps x RT x 32 x 16 B + scales
  constexpr int XB = 4096, WB = 8 * RT * 512, STG = XB + WB + 8 * 32 * 8;
  const int tid = threadIdx.x, wg = tid >> 7, warp = tid >> 5, lane = tid & 31;
  for (int i = tid; i < NSTG * STG / 4; i += 384) reinterpret_cast<uint32_t*>(sm)[i] = (i * 2654435761u) & 0x3F3F3F3Fu;
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  if (wg == 2) { asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n"); return; }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");
  float acc[RT][N / 2];
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int i = 0; i < N / 2; ++i) acc[r][i] = 0.f;
  uint32_t af[D][RT][4];
#pragma unroll
  for (int d = 0; d < D; ++d)
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int i = 0; i < 4; ++i) af[d][r][i] = 0x3F803F80u ^ ((d * 4 + r) << 1) ^ i;
  int st = 0;
  const unsigned long long t0 = clock64();
#pragma unroll 1
  for (int kc = 0; kc < nst; ++kc) {
    const char* sb = sm + st * STG;
    uint4 wv[RT];
#pragma unroll
    for (int r = 0; r < RT; ++r) wv[r] = reinterpret_cast<const uint4*>(sb + XB)[(warp * RT + r) * 32 + lane];
    const uint2 sv = reinterpret_cast<const uint2*>(sb + XB + WB)[warp * 32 + lane];
    uint32_t s0[RT], s1[RT];
#pragma unroll
    for (int r = 0; r < RT; ++r) { const uint32_t w = (r & 2) ? sv.y : sv.x; s0[r] = (w & 0x00FF00FFu) | 0x3F003F00u; s1[r] = ((w >> 8) & 0x00FF00FFu) | 0x3F003F00u; }
    const uint64_t d0 = sw128_desc(sb);
#pragma unroll
    for (int j0 = 0; j0 < 4; ++j0) {
      const int sl = D == 4 ? j0 : (j0 & 1);
      if (MODE != 2 && MODE != 4) {
#pragma unroll
        for (int r = 0; r < RT; ++r) dq(u4get(wv[r], j0), s0[r], s1[r], af[sl][r]);
      }
      if (MODE >= 3) {
#pragma unroll
        for (int nb = 0; nb < N / 8; ++nb) {
          const uint2 b = reinterpret_cast<const uint2*>(sb)[(j0 * (N / 8) + nb) * 32 + lane];
#pragma unroll
          for (int r = 0; r < RT; ++r) mma16816(&acc[r][nb * 4], af[sl][r], b.x, b.y);
        }
      } else if (MODE != 1) {
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
        for (int r = 0; r < RT; ++r) wg_rs<N>(acc[r], af[sl][r], d0 + j0 * 2);
        asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
        if (D == 4) wg_wait<3>(); else wg_wait<1>();
      }
#pragma unroll
      for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int i = 0; i < 4; ++i) fence_r(af[D == 4 ? ((j0 + 1) & 3) : ((j0 + 1) & 1)][r][i]);
    }
    if (++st == NSTG) st = 0;
  }
  wg_wait<0>();
  const unsigned long long t1 = clock64();
  float s = 0.f;
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int i = 0; i < N / 2; ++i) s += acc[r][i];
#pragma unroll
  for (int d = 0; d < D; ++d)
#pragma unroll
    for (int r = 0; r < RT; ++r) s += __uint_as_float(af[d][r][0] ^ af[d][r][3]);
  sink[blockIdx.x * 384 + tid] = s;
  if ((tid & 127) == 0) out[blockIdx.x * 2 + wg] = t1 - t0;
}
int64_t run(int64_t mode, int64_t rt, int64_t n, int64_t d, torch::Tensor out, torch::Tensor sink, int64_t nst, int64_t grid) {
  constexpr int SMB = NSTG * (4096 + 8 * 4 * 512 + 8 * 32 * 8) + 1024;
  #define GO(M, R, NN, DD) if (mode == M && rt == R && n == NN && d == DD) { auto k = ring<M, R, NN, DD>; \
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SMB); \
    k<<<grid, 384, SMB>>>((unsigned long long*)out.data_ptr(), (float*)sink.data_ptr(), (int)nst); return 0; }
  #define GM(NN, DD) GO(0, 4, NN, DD) GO(1, 4, NN, DD) GO(2, 4, NN, DD) GO(3, 4, NN, DD) GO(4, 4, NN, DD)
  GM(8, 2) GM(16, 2) GM(32, 2)
  return -1;
}
"""
SRC = SRC.replace("@@GEN@@", "".join(gen_rs(n) for n in NS))
NAME = "n6s2_ring2"
lib = load_inline(NAME, cpp_sources="int64_t run(int64_t, int64_t, int64_t, int64_t, torch::Tensor, torch::Tensor, int64_t, int64_t);",
                  cuda_sources="#include <torch/extension.h>\n" + SRC, functions=["run"],
                  extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3", "-Xptxas=-v"], verbose=False)
bdir = glob.glob(os.path.expanduser(f"~/.cache/torch_extensions/*/{NAME}"))[0]
so = f"{bdir}/{NAME}.so"
sass = subprocess.run(["/usr/local/cuda-13.0/bin/cuobjdump", "-sass", so], capture_output=True, text=True).stdout
info = {}
for blk in sass.split("Function : ")[1:]:
    nm = blk.split()[0]
    info[nm] = (blk.count("DEPBAR.LE gsb0, 0x0"), blk.count("HGMMA"), len(re.findall(r"\bSTL\b|\bLDL\b", blk)))
G = int(sys.argv[1]) if len(sys.argv) > 1 else 1
out = torch.zeros(2 * 132, dtype=torch.int64, device="cuda"); sink = torch.zeros(132 * 384, device="cuda")
NST = 512
for n in NS:
    row = []
    for d in (2,):
        for mode in (0, 2, 3, 4):
            lib.run(mode, 4, n, d, out, sink, 16, G); torch.cuda.synchronize()
            with open('/tmp/okc-gpu.lock', 'w') as lk:
                fcntl.flock(lk, fcntl.LOCK_EX)
                lib.run(mode, 4, n, d, out, sink, NST, G); torch.cuda.synchronize()
            c = max(out[0].item(), out[1].item()) / NST
            dep0, hg, spill = info.get(f"_Z4ringILi{mode}ELi4ELi{n}ELi{d}EEvPyPfi", ("?", "?", "?"))
            row.append(f"D{d} {['wgmma+dq', 'dq', 'wgmma only', 'mma.sync+dq', 'mma.sync only'][mode]} {c:6.0f} (dep0 {dep0} hg {hg} lmem {spill})")
    print(f"RT4 N{n:2d} cyc/stage: " + " | ".join(row), flush=True)
