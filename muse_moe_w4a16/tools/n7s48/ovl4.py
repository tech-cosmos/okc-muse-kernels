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

// compact per-WG stage: x 8192 | W 4 warps x RT x 512 | S 4 x 32 x 8
template <int RT> struct CZ { static constexpr int XB = 8192, WB = 4 * RT * 512, STG = XB + WB + 1024; };
__device__ __forceinline__ void mb_init(uint64_t* b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(b)), "r"(n)); }
__device__ __forceinline__ void mb_expect(uint64_t* b, int tx) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(b)), "r"(tx) : "memory"); }
__device__ __forceinline__ void mb_wait(uint64_t* b, int ph) {
  asm volatile("{\n.reg .pred p;\nW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}\n" ::"r"(smem_u32(b)), "r"(ph) : "memory"); }
__device__ __forceinline__ void bulk(void* dst, const void* src, int bytes, uint64_t* b) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(b)) : "memory"); }

// SELF = 1: WG feeds its own ring (thread 0 of the WG issues the bulk copy of stage kc+NSG-1 after a WG barrier at
// the start of stage kc). SELF = 0: same compute, no copies (static smem, ring index only).
template <int RT, int N, int D, int NSG, int SELF>
__device__ __forceinline__ unsigned long long body(char* sm, uint64_t* bars, const char* gsrc, int gst, int warp, int lane, int wg, int nst, float* sinkp) {
  constexpr int STG = CZ<RT>::STG, XBW = CZ<RT>::XB + CZ<RT>::WB;
  constexpr int CO = SELF == 1 ? 0 : CZ<RT>::XB, CB = SELF == 1 ? STG : SELF == 3 ? STG - CZ<RT>::XB : 16;
  float acc[RT][N / 2];
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int i = 0; i < N / 2; ++i) acc[r][i] = 0.f;
  uint32_t af[D][RT][4];
  const uint32_t seed = (uint32_t)clock() & 0x10u;
#pragma unroll
  for (int d = 0; d < D; ++d)
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int i = 0; i < 4; ++i) af[d][r][i] = (0x3F803F80u ^ ((d * 16 + r * 4 + i) << 1)) | seed;
  const bool el = (threadIdx.x & 127) == 0;
  int gi = 0;
  if ((SELF == 1 || (SELF >= 3 && SELF != 6 && SELF != 8)) && el) for (int s = 0; s < NSG - 1; ++s) { mb_expect(&bars[s], CB); bulk(sm + s * STG + CO, gsrc + (gi++ % gst) * STG, CB, &bars[s]); }
  int st = 0, ph = 0;
  const unsigned long long t0 = clock64();
#pragma unroll 1
  for (int kc = 0; kc < nst; ++kc) {
    const char* sb = sm + st * STG;
    const char* sw = SELF == 2 ? sb + (wg & 1) * 4 * RT * 512 : sb;
    if (SELF == 1 || SELF >= 3) {
      if (SELF != 5) asm volatile("bar.sync %0, 128;" ::"r"(1 + wg) : "memory");  // all warps of this WG are done with stage kc-1's slot
      if (el && SELF != 6 && SELF != 8) { const int ns = st == 0 ? NSG - 1 : st - 1; mb_expect(&bars[ns], CB); bulk(sm + ns * STG + CO, gsrc + (gi++ % gst) * STG, CB, &bars[ns]); }
      if (el && SELF == 8) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(&bars[st])) : "memory");
      if (SELF != 6 && SELF != 7) mb_wait(&bars[st], ph);
    }
    uint4 wv[RT];
    uint32_t s0[RT], s1[RT];
#pragma unroll
    for (int r = 0; r < RT; ++r) wv[r] = reinterpret_cast<const uint4*>(sw + CZ<RT>::XB)[(warp * RT + r) * 32 + lane];
    const uint2 sv = reinterpret_cast<const uint2*>(sb + XBW)[warp * 32 + lane];
#pragma unroll
    for (int r = 0; r < RT; ++r) { const uint32_t w = (r & 2) ? sv.y : sv.x; s0[r] = (w & 0x00FF00FFu) | 0x3F003F00u; s1[r] = ((w >> 8) & 0x00FF00FFu) | 0x3F003F00u; }
    const uint64_t d0 = sw128_desc(sb);
#pragma unroll
    for (int j0 = 0; j0 < 4; ++j0) {
      const int sl = D == 4 ? j0 : D == 2 ? (j0 & 1) : 0;
#pragma unroll
      for (int r = 0; r < RT; ++r) dq(u4get(wv[r], j0), s0[r], s1[r], af[sl][r]);
      asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
      for (int r = 0; r < RT; ++r) wg_rs<N>(acc[r], af[sl][r], d0 + j0 * 2);
      asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
      if (D == 4) wg_wait<3>(); else if (D == 2) wg_wait<1>(); else wg_wait<0>();
#pragma unroll
      for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int i = 0; i < 4; ++i) fence_r(af[D == 4 ? ((j0 + 1) & 3) : D == 2 ? ((j0 + 1) & 1) : 0][r][i]);
    }
    if (++st == NSG) { st = 0; ph ^= 1; }
  }
  wg_wait<0>();
  const unsigned long long t1 = clock64();
  float s = 0.f;
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int i = 0; i < N / 2; ++i) s += acc[r][i];
  *sinkp = s;
  return t1 - t0;
}
// NWG consumer WGs, each with its own ring of NSG compact stages; XT extra idle warp (models a producer warp's reg cap)
template <int RT, int N, int D, int NWG, int NSG, int SELF, int XT>
__global__ void __launch_bounds__(NWG * 128 + XT * 32, 1) ovl(unsigned long long* out, float* sink, const char* gsrc, int gst, int nst) {
  extern __shared__ __align__(1024) char sm[];
  uint64_t (*bars)[8] = reinterpret_cast<uint64_t (*)[8]>(sm + NWG * NSG * CZ<RT>::STG);
  constexpr int STG = CZ<RT>::STG;
  const int tid = threadIdx.x, wg = tid >> 7, warp = (tid >> 5) & 3, lane = tid & 31;
  for (int i = tid; i < NWG * NSG * STG / 4; i += blockDim.x) reinterpret_cast<uint32_t*>(sm)[i] = FILLX;
  if (tid < NWG * NSG) mb_init(&bars[tid / NSG][tid % NSG], 1);
  asm volatile("fence.mbarrier_init.release.cluster;\nfence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  if (tid >= NWG * 128) return;
  unsigned long long t;
#define BW(I) t = body<RT, N, D, NSG, SELF>(sm + (SELF == 2 ? 0 : I * NSG * STG), bars[I], gsrc, gst, warp, lane, I, nst, sink + tid)
  if (wg == 0) BW(0); else if (wg == 1) BW(1); else if (wg == 2) BW(2); else BW(3);
  if ((tid & 127) == 0) out[wg] = t;
}
int64_t run(int64_t cfg, torch::Tensor out, torch::Tensor sink, torch::Tensor src, int64_t nst) {
  int c = 0;
  #define GO(RT, NN, DD, W, S, SF, XT) if (cfg == c++) { auto k = ovl<RT, NN, DD, W, S, SF, XT>; \
    constexpr int SMB = W * S * CZ<RT>::STG + 1024; \
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SMB); \
    k<<<1, W * 128 + XT * 32, SMB>>>((unsigned long long*)out.data_ptr(), (float*)sink.data_ptr(), (const char*)src.data_ptr(), (int)(src.numel() / CZ<RT>::STG), (int)nst); return 0; }
  @@CFGS@@
  return -1;
}
"""
# (label, D, NWG, NSG, SELF, XT)
CF = [("2WG static", 2, 2, 3, 0, 0), ("2WG self sync", 2, 2, 3, 4, 0), ("2WG issue only", 2, 2, 3, 7, 0), ("2WG arrive+wait only", 2, 2, 3, 8, 0),
      ("4WG static", 2, 4, 3, 0, 0), ("4WG self sync", 2, 4, 3, 4, 0), ("4WG issue only", 2, 4, 3, 7, 0), ("4WG arrive+wait only", 2, 4, 3, 8, 0)]
SH = [(4, 32), (2, 64), (2, 32), (4, 16)] if not os.environ.get("SH") else [tuple(int(x) for x in t.split("x")) for t in os.environ["SH"].split(",")]
ALLC = [(lab, rt, n, d, w, s, sf, xt) for (rt, n) in SH for (lab, d, w, s, sf, xt) in CF if w * s * (8192 + 4 * rt * 512 + 1024) + 1024 <= 227 * 1024]
gos = "\n  ".join(f"GO({rt}, {n}, {d}, {w}, {s}, {sf}, {xt})" for (lab, rt, n, d, w, s, sf, xt) in ALLC)
SRC = SRC.replace("FILLX", {"h": "(i * 2654435761u) & 0x3F3F3F3Fu", "z": "0u", "c": "0x11111111u"}[os.environ.get("FILL", "h")]).replace("@@GEN@@", "".join(gen_rs(n) for n in NS)).replace("@@CFGS@@", gos)
NAME = "n7s48_ovl4j_" + os.environ.get("FILL", "h") + os.environ.get("TAG", "a")
lib = load_inline(NAME, cpp_sources="int64_t run(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, int64_t);",
                  cuda_sources="#include <torch/extension.h>\n" + SRC, functions=["run"],
                  extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3", "-Xptxas=-v"], verbose=bool(os.environ.get("VERB")))
bdir = glob.glob(os.path.expanduser(f"~/.cache/torch_extensions/*/{NAME}"))[0]
sass = subprocess.run(["/usr/local/cuda-13.0/bin/cuobjdump", "-sass", f"{bdir}/{NAME}.so"], capture_output=True, text=True).stdout
info = {}
for blk in sass.split("Function : ")[1:]:
    nm = blk.split()[0]
    m = re.match(r"_Z3ovlILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)E", nm)
    if m: info[tuple(int(x) for x in m.groups())] = (blk.count("DEPBAR.LE gsb0, 0x0"), blk.count("HGMMA"), len(re.findall(r"\bSTL\b|\bLDL\b", blk)))
out = torch.zeros(8, dtype=torch.int64, device="cuda"); sink = torch.zeros(1024, device="cuda")
src = (torch.randint(0, 256, (2 << 20,), dtype=torch.uint8, device="cuda") & 0x3F)  # 2 MB, L2-resident
NST, REP = int(os.environ.get("NST", "512")), int(os.environ.get("REP", "3"))
for ci, (lab, rt, n, d, w, s, sf, xt) in enumerate(ALLC):
    out.zero_(); lib.run(ci, out, sink, src, 16); torch.cuda.synchronize()
    best = None
    for _ in range(REP):
        with open('/tmp/okc-gpu.lock', 'w') as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            lib.run(ci, out, sink, src, NST); torch.cuda.synchronize()
        v = max(out[i].item() / NST for i in range(w))
        if best is None or v < best: best = v
    dep0, hg, lm = info.get((rt, n, d, w, s, sf, xt), ("?", "?", "?"))
    wpc = w * rt * 4096 / best
    print(f"RT{rt} N{n:2d} {lab:20s} cyc/stage/WG {best:6.0f}  SM weights/clk {wpc:5.1f}  (dep0 {dep0} hg {hg} lmem {lm})", flush=True)
