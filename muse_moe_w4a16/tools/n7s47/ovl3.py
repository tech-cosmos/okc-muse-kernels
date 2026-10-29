# n7s47 (lead 14:30Z): dq/wgmma overlap re-open. twm stage microbench (n6s2 ring.py structure: RT row tiles, 4 k16
# groups per stage, production dq(), weights + scales LDS'd from 8 static smem stages, x tile = SW128 B from smem), but
# every consumer warpgroup runs its OWN ROLE in its own loop (no shared loop body), and each WG times itself.
# Roles: 0 idle | 1 twm (dq -> fence -> RT RS wgmma -> commit -> wait<D-1>) | 2 dq only (xor sink) | 3 RS wgmma on
# constant distinct A (fence each group) | 4 RS wgmma on "fresh" A (1 LOP3 per A reg, then fence) | 5 twm + FA3 pingpong
# (WG w bar.sync's on bar 1+w before its wgmma batch, bar.arrive's bar 2-w after it) | 6 twm with dq of group j+1 hoisted
# between commit(j) and wait (explicit 2-set software pipeline).
# Config: (R0, R1, R2, RT, N, D, SMR). R2 = role of WG2 (-1 = twm's idle producer WG with setmaxnreg 40/232 when SMR=1).
# Output: cycles per stage per WG (clock64 over 512 stages, 1 CTA), and the kernel's DEPBAR gsb0 0x0 count / lmem.
import torch, os, sys, fcntl, subprocess, glob, re
from torch.utils.cpp_extension import load_inline
def gen_rs(n):
    na = n // 2; dl = ", ".join(f"%{i}" for i in range(na)); al = ", ".join(f"%{na + i}" for i in range(4))
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(na))
    return (f"template <> __device__ __forceinline__ void wg_rs<{n}>(float* d, const uint32_t* a, uint64_t desc) {{\n"
            f'  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, %{na + 5}, 0;\\n"\n'
            f'    "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{dl}}}, {{{al}}}, %{na + 4}, p, 1, 1, 0;\\n}}\\n"\n'
            f'    : {outs} : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));\n}}\n')
NS = (8, 16, 24, 32, 40, 48, 56, 64)
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
__device__ __forceinline__ uint32_t u4get(const uint4& v, int j) { return j == 0 ? v.x : j == 1 ? v.y : j == 2 ? v.z : v.w; }
constexpr int NSTG = 8, XB = 8192;
template <int RT> struct SZ { static constexpr int WB = 8 * RT * 512, STG = XB + WB + 8 * 32 * 8; };

template <int ROLE, int RT, int N, int D>
__device__ __forceinline__ unsigned long long body(const char* sm, int warp, int lane, int wg, int nst, float* sinkp) {
  constexpr int STG = SZ<RT>::STG, XBW = XB + SZ<RT>::WB;
  float acc[RT][N / 2];
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int i = 0; i < N / 2; ++i) acc[r][i] = 0.f;
  uint32_t af[D][RT][4];
  const uint32_t seed = (uint32_t)clock() & 0x10u;  // runtime value: ptxas can't fold or merge the A quads
#pragma unroll
  for (int d = 0; d < D; ++d)
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int i = 0; i < 4; ++i) af[d][r][i] = (0x3F803F80u ^ ((d * 16 + r * 4 + i) << 1)) | seed;
  uint32_t xs = 0;
  int st = 0;
  if (ROLE == 5 && wg == 1) asm volatile("bar.arrive 1, 256;" ::: "memory");
  const unsigned long long t0 = clock64();
#pragma unroll 1
  for (int kc = 0; kc < nst; ++kc) {
    const char* sb = sm + st * STG;
    uint4 wv[RT];
    uint32_t s0[RT], s1[RT];
    if (ROLE == 1 || ROLE == 2 || ROLE == 5 || ROLE == 6) {
#pragma unroll
      for (int r = 0; r < RT; ++r) wv[r] = reinterpret_cast<const uint4*>(sb + XB)[(warp * RT + r) * 32 + lane];
      const uint2 sv = reinterpret_cast<const uint2*>(sb + XBW)[warp * 32 + lane];
#pragma unroll
      for (int r = 0; r < RT; ++r) { const uint32_t w = (r & 2) ? sv.y : sv.x; s0[r] = (w & 0x00FF00FFu) | 0x3F003F00u; s1[r] = ((w >> 8) & 0x00FF00FFu) | 0x3F003F00u; }
    }
    const uint64_t d0 = sw128_desc(sb);
    if (ROLE == 6 && true) {
      // explicit pipeline: group 0 dq'd here, then per group: fence, mma(j), commit, dq(j+1) into the other set, wait<1>
#pragma unroll
      for (int r = 0; r < RT; ++r) dq(wv[r].x, s0[r], s1[r], af[0][r]);
#pragma unroll
      for (int j0 = 0; j0 < 4; ++j0) {
        const int sl = j0 & 1;
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
        for (int r = 0; r < RT; ++r) wg_rs<N>(acc[r], af[sl][r], d0 + j0 * 2);
        asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
        if (j0 < 3) {
          wg_wait<1>();  // mma(j-1) done reading set sl^1
#pragma unroll
          for (int r = 0; r < RT; ++r) dq(u4get(wv[r], j0 + 1), s0[r], s1[r], af[sl ^ 1][r]);
        }
      }
      wg_wait<0>();
    } else {
#pragma unroll
    for (int j0 = 0; j0 < 4; ++j0) {
      const int sl = D == 4 ? j0 : D == 2 ? (j0 & 1) : 0;
      if (ROLE == 1 || ROLE == 2 || ROLE == 5) {
#pragma unroll
        for (int r = 0; r < RT; ++r) dq(u4get(wv[r], j0), s0[r], s1[r], af[sl][r]);
      }
      if (ROLE == 4) {
#pragma unroll
        for (int r = 0; r < RT; ++r)
#pragma unroll
          for (int i = 0; i < 4; ++i) af[sl][r][i] ^= 0x00010001u;
      }
      if (ROLE == 2) {
#pragma unroll
        for (int r = 0; r < RT; ++r) xs ^= af[sl][r][0] ^ af[sl][r][1] ^ af[sl][r][2] ^ af[sl][r][3];
      } else if (ROLE != 0) {
        if (ROLE == 5) asm volatile("bar.sync %0, 256;" ::"r"(1 + wg) : "memory");
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
        for (int r = 0; r < RT; ++r) wg_rs<N>(acc[r], af[sl][r], d0 + j0 * 2);
        asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
        if (ROLE == 5) asm volatile("bar.arrive %0, 256;" ::"r"(2 - wg) : "memory");
        if (D == 4) wg_wait<3>(); else if (D == 2) wg_wait<1>(); else wg_wait<0>();
      }
#pragma unroll
      for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int i = 0; i < 4; ++i) fence_r(af[D == 4 ? ((j0 + 1) & 3) : D == 2 ? ((j0 + 1) & 1) : 0][r][i]);
    }
    }
    if (++st == 4) st = 0;
  }
  if (ROLE == 5 && wg == 0) asm volatile("bar.sync 1, 256;" ::: "memory");  // absorb WG1's last arrive
  wg_wait<0>();
  const unsigned long long t1 = clock64();
  float s = __uint_as_float(xs);
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int i = 0; i < N / 2; ++i) s += acc[r][i];
#pragma unroll
  for (int d = 0; d < D; ++d)
#pragma unroll
    for (int r = 0; r < RT; ++r) s += __uint_as_float(af[d][r][0] ^ af[d][r][3]);
  *sinkp = s;
  return t1 - t0;
}

template <int ROLE, int RT, int N, int D, int NWG, int CPB, int NSG>
__global__ void __launch_bounds__(NWG * 128, CPB) ovl(unsigned long long* out, float* sink, int nst) {
  extern __shared__ __align__(1024) char sm[];
  constexpr int STG = SZ<RT>::STG;
  const int tid = threadIdx.x, wg = tid >> 7, warp = (tid >> 5) & 3, lane = tid & 31;
  for (int i = tid; i < NSG * STG / 4; i += NWG * 128) reinterpret_cast<uint32_t*>(sm)[i] = (i * 2654435761u) & 0x3F3F3F3Fu;
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  float* sp = sink + blockIdx.x * 512 + tid;
  // every WG uses warps 0..3's slots of the stage (same LDS pattern as WG0 of twm); ring of NSG stages
  unsigned long long t = body<ROLE, RT, N, D>(sm, warp + 4 * (wg & 1), lane, wg, nst, sp);
  if ((tid & 127) == 0) { unsigned smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid)); out[blockIdx.x * 8 + wg] = t; out[blockIdx.x * 8 + 7] = smid; }
}
int64_t run(int64_t cfg, torch::Tensor out, torch::Tensor sink, int64_t nst) {
  int c = 0;
  #define GO(RO, RT, NN, DD, W, C, G, S) if (cfg == c++) { auto k = ovl<RO, RT, NN, DD, W, C, S>; \
    constexpr int SMB = S * SZ<RT>::STG + 1024; \
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SMB); \
    k<<<G, W * 128, SMB>>>((unsigned long long*)out.data_ptr(), (float*)sink.data_ptr(), (int)nst); return 0; }
  @@CFGS@@
  return -1;
}
"""
# (label, ROLE, D, NWG, CPB, GRID, NSG)
CF = [("1cta 2WG twm (base)", 1, 2, 2, 1, 1, 4), ("1cta 4WG twm", 1, 2, 4, 1, 1, 4),
      ("2cta x 2WG twm", 1, 2, 2, 2, 264, 4), ("2cta x 1WG twm", 1, 2, 1, 2, 264, 4), ("1cta 1WG twm", 1, 2, 1, 1, 1, 4),
      ("1cta 4WG dq", 2, 2, 4, 1, 1, 4), ("1cta 2WG dq", 2, 2, 2, 1, 1, 4),
      ("1cta 4WG mmaC", 3, 2, 4, 1, 1, 4), ("1cta 2WG mmaC", 3, 2, 2, 1, 1, 4),
      ("1cta 4WG twm D4", 1, 4, 4, 1, 1, 4), ("1cta 2WG twm D4", 1, 4, 2, 1, 1, 4)]
SH = [(4, 32), (2, 64), (2, 32), (4, 16)] if not os.environ.get("SH") else [tuple(int(x) for x in t.split("x")) for t in os.environ["SH"].split(",")]
ALLC = [(lab, ro, rt, n, d, w, cpb, g, nsg) for (rt, n) in SH for (lab, ro, d, w, cpb, g, nsg) in CF]
gos = "\n  ".join(f"GO({ro}, {rt}, {n}, {d}, {w}, {cpb}, {g}, {nsg})" for (lab, ro, rt, n, d, w, cpb, g, nsg) in ALLC)
SRC = SRC.replace("@@GEN@@", "".join(gen_rs(n) for n in NS)).replace("@@CFGS@@", gos)
NAME = "n7s48_ovl3_" + os.environ.get("TAG", "a")
lib = load_inline(NAME, cpp_sources="int64_t run(int64_t, torch::Tensor, torch::Tensor, int64_t);",
                  cuda_sources="#include <torch/extension.h>\n" + SRC, functions=["run"],
                  extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"], verbose=bool(os.environ.get("VERB")))
bdir = glob.glob(os.path.expanduser(f"~/.cache/torch_extensions/*/{NAME}"))[0]
so = f"{bdir}/{NAME}.so"
sass = subprocess.run(["/usr/local/cuda-13.0/bin/cuobjdump", "-sass", so], capture_output=True, text=True).stdout
info = {}
for blk in sass.split("Function : ")[1:]:
    nm = blk.split()[0]
    m = re.match(r"_Z3ovlILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)E", nm)
    if m: info[tuple(int(x) for x in m.groups())] = (blk.count("DEPBAR.LE gsb0, 0x0"), blk.count("HGMMA"), len(re.findall(r"\bSTL\b|\bLDL\b", blk)))
out = torch.zeros(264 * 8, dtype=torch.int64, device="cuda"); sink = torch.zeros(264 * 512, device="cuda")
NST, REP = int(os.environ.get("NST", "512")), int(os.environ.get("REP", "3"))
for ci, (lab, ro, rt, n, d, w, cpb, g, nsg) in enumerate(ALLC):
    out.zero_(); lib.run(ci, out, sink, 16); torch.cuda.synchronize()
    best = None
    for _ in range(REP):
        out.zero_()
        with open('/tmp/okc-gpu.lock', 'w') as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            lib.run(ci, out, sink, NST); torch.cuda.synchronize()
        o = out.view(264, 8).cpu()
        per = o[:g, :w].double() / NST
        sms = o[:g, 7]
        v = per.max().item(); med = per.median().item()
        if best is None or v < best[0]: best = (v, med, len(set(sms.tolist())))
    dep0, hg, lm = info.get((ro, rt, n, d, w, cpb, nsg), ("?", "?", "?"))
    nwg_sm = w * cpb
    print(f"RT{rt} N{n:2d} {lab:22s} cyc/stage/WG max {best[0]:6.0f} med {best[1]:6.0f}  WG/SM {nwg_sm}  SM cyc per 2 WG-stages {best[0]*2/nwg_sm:6.0f}  (sms {best[2]} dep0 {dep0} hg {hg} lmem {lm})", flush=True)
