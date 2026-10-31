# o9 (lead 2026-10-31T12:25Z TWM NON-DQ OVERHEAD): real tw_item (r-layer moe_src.py, twm config: KG 1, DYN, SA, STRIDE =
# CG::STG, KS false) in a harness with a data-free mbarrier ring (producer warp only arrives full[st] after empty[st]),
# epilogue skipped at run time (TWM_ORC 16: code kept, so the dq stays). Variants (timing only, V via -DOVRV):
#   0 real | 1 scales hoisted (no per-stage scale LDS + SA decode; constants) | 2 no ring protocol (no full waits for kc > 0,
#   no empty arrives; producer idle) | 3 = 1 + 2 | 4 = 3 + no per-stage desc/stage-pointer math (fixed stage 0)
# usage: python ovr.py <moe_src.py> ; prints cyc/stage per (RT, N) and variant
import sys, os, re, fcntl, torch
from torch.utils.cpp_extension import load_inline
src = open(sys.argv[1]).read()
cu = re.search(r'CUDA_SRC = r"""(.*?)\n"""', src, re.S).group(1)
cut = cu.index("template <int RT, int KG, int NSTG, bool GATED, int NW>\n__global__ void __launch_bounds__(TWC<RT, GATED, NW>::THREADS, 1) moe_tw(")
cu = cu[:cut]
ns = {}; exec(re.search(r"(def _gen_wgmma\(\):.*?return .*?\n)", src, re.S).group(1), ns)
cu = cu.replace("//@@WGMMA@@", ns["_gen_wgmma"]())
# variant knobs inside tw_item
a = """    if (kc) mbar_wait(&full[st], ph);"""
assert a in cu; cu = cu.replace(a, """    if (!(OVRV & 2)) { if (kc && (!(OVRV & 16) || !(kc & 1))) mbar_wait(&full[st], ph); }""", 1)
a = """    uint32_t s0[RT], s1[RT];
#pragma unroll
    for (int r = 0; r < RT; ++r) {
      const uint32_t sw = sc[r >> 1] >> ((r & 1) * 16);"""
assert a in cu
cu = cu.replace(a, """    uint32_t s0[RT], s1[RT];
#pragma unroll
    for (int r = 0; r < RT; ++r) {
      if (OVRV & 1) { s0[r] = 0x3F803F80u ^ (lane << 1); s1[r] = 0x3F003F00u ^ (lane << 1); continue; }
      const uint32_t sw = sc[r >> 1] >> ((r & 1) * 16);""", 1)
a = "    uint32_t sc[(RT + 1) / 2];\n    if (NW == 16) {"
assert a in cu; cu = cu.replace(a, "    uint32_t sc[(RT + 1) / 2] = {};\n    if (OVRV & 1) {} else if (NW == 16) {", 1)
a = "      if (j0 == 0 && pst >= 0) mbar_arrive(&empty[pst]);"
assert a in cu; cu = cu.replace(a, "      if (!(OVRV & 2)) { if (j0 == 0 && pst >= 0 && (!(OVRV & 16) || (kc & 1) || kc == 0 || true)) { if (OVRV & 8) mbar_arrive_u(eb + pst * 8); else mbar_arrive(&empty[pst]); } }", 1)
a = "  int pst = -1;\n#pragma unroll 1\n  for (int kc = 0; kc < nkc; ++kc) {"
assert a in cu; cu = cu.replace(a, "  int pst = -1;\n  uint32_t eb; asm volatile(\"mov.u32 %0, %1;\" : \"=r\"(eb) : \"r\"(smem_u32(empty)));\n#pragma unroll 1\n  for (int kc = 0; kc < nkc; ++kc) {", 1)
a = "__device__ __forceinline__ void mbar_wait(uint64_t* b, int parity) {"
assert a in cu; cu = cu.replace(a, """__device__ __forceinline__ void mbar_arrive_u(uint32_t b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(b) : "memory");
}
""" + a, 1)
a = "  mbar_arrive(&empty[pst]);\n#if TWM_ORC & 4"
assert a in cu; cu = cu.replace(a, "  if (!(OVRV & 2)) mbar_arrive(&empty[pst]);\n#if TWM_ORC & 4", 1)
a = "    const char* sb = smem + st * (STRIDE ? STRIDE : C::STG);\n    if (!(OVRV"
assert a in cu; cu = cu.replace(a, "    const char* sb = (OVRV & 4) ? smem : smem + st * (STRIDE ? STRIDE : C::STG);\n    if (!(OVRV", 1)
H = r"""
constexpr int HNSTG = 8;
using HCG = TWC<4, true, 8>;
template <int RT, int N>
__global__ void __launch_bounds__(384, 1) ovr(unsigned long long* out, int nitems, const uint32_t* glut, float gs) {
  extern __shared__ __align__(1024) uint4 smem_tw[];
  char* smem = reinterpret_cast<char*>(smem_tw) + ((1024 - (smem_u32(smem_tw) & 1023)) & 1023);
  __shared__ uint32_t slut[256];
  __shared__ __align__(8) uint64_t full[HNSTG], empty[HNSTG];
  __shared__ int s_flag;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  for (int i = tid; i < HNSTG * HCG::STG / 4; i += 384) {
    uint32_t v = (i * 2654435761u) ^ (i >> 3);
    reinterpret_cast<uint32_t*>(smem)[i] = v;
  }
  // scale bytes: positive normal e4m3 (0x30..0x5F)
  for (int s = 0; s < HNSTG; ++s)
    for (int i = tid; i < HCG::SB; i += 384) smem[s * HCG::STG + HCG::XB + HCG::WB + i] = (char)(0x30 + (i * 7) % 48);
  if (tid < 256) slut[tid] = glut[tid];
  if (tid == 0) for (int i = 0; i < HNSTG; ++i) { mbar_init(&full[i], 1); mbar_init(&empty[i], 8 * 32); }
  asm volatile("fence.mbarrier_init.release.cluster;\nfence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  const int total = nitems * 64;
  if (warp >= 8) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n" ::: "memory");
    if (warp == 8 && lane == 0) {
      int st = 0, ph = 0;
      const int lim = (OVRV & 2) ? HNSTG : total;
      for (int i = 0; i < lim; ++i) {
        if (!(OVRV & 2) && i >= HNSTG) mbar_wait(&empty[st], ph ^ 1);
        mbar_arrive(&full[st]);
        if (++st == HNSTG) { st = 0; ph ^= 1; }
      }
    }
    return;
  }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n" ::: "memory");
  int st = 0, ph = 0;
  const unsigned long long t0 = clock64();
  for (int it = 0; it < nitems; ++it) {
    if (!(OVRV & 2) || it == 0) mbar_wait(&full[st], ph);
    tw_item<RT, 1, N, HNSTG, true, 8, true, true, HCG::STG, 0, false>(smem, full, empty, st, ph, slut, warp, lane, N, 0, 0,
        gs, nullptr, nullptr, false, 64, nullptr, nullptr, &s_flag, 0);
  }
  const unsigned long long t1 = clock64();
  if ((tid & 127) == 0) out[tid >> 7] = t1 - t0;
}
int64_t run(int64_t cfg, torch::Tensor out, torch::Tensor lut, int64_t nitems) {
  constexpr int SMB = HNSTG * HCG::STG + 1024 + 8 * 1024;
  int c = 0;
#define GO(RT, NN) if (cfg == c++) { auto k = ovr<RT, NN>; \
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SMB); \
    k<<<1, 384, SMB>>>((unsigned long long*)out.data_ptr(), (int)nitems, (const uint32_t*)lut.data_ptr(), 1.0f + 1e-7f * (float)nitems); return 0; }
  GO(4, 8) GO(4, 16) GO(4, 24) GO(4, 32) GO(2, 32) GO(2, 64)
  return -1;
}
"""
SH = ["RT4 N8", "RT4 N16", "RT4 N24", "RT4 N32", "RT2 N32", "RT2 N64"]
VS = [int(v) for v in os.environ.get("VS", "0,1,2,3,4").split(",")]
NI = 6
res = {}
for v in VS:
    full = "#include <torch/extension.h>\n#define TWM_ORC 16\n#define OVRV %d\n" % v + cu + H
    lib = load_inline("o9_ovr_v%d_%s" % (v, __import__("hashlib").sha1(full.encode()).hexdigest()[:8]),
                      cpp_sources="int64_t run(int64_t, torch::Tensor, torch::Tensor, int64_t);", cuda_sources=full,
                      functions=["run"], extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3", "-std=c++20", "--expt-relaxed-constexpr", "-w"])
    out = torch.zeros(4, dtype=torch.int64, device="cuda"); lut = torch.arange(256, dtype=torch.int32, device="cuda")
    for c, nm in enumerate(SH):
        lib.run(c, out, lut, 2); torch.cuda.synchronize()
        if os.environ.get("SMOKE"): print("ok", v, nm, out[0].item(), flush=True); continue
        ts = []
        with open("/tmp/okc-gpu.lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            for _ in range(15):
                lib.run(c, out, lut, NI); torch.cuda.synchronize(); ts.append(max(out[0].item(), out[1].item()) / (NI * 64))
        res[(v, nm)] = sorted(ts)[len(ts) // 2]
    if os.environ.get("SMOKE"): continue
    print("V%d " % v + " | ".join("%s %.0f" % (nm, res[(v, nm)]) for nm in SH), flush=True)
if 0 in VS and not os.environ.get('SMOKE'):
    for v in VS:
        if v: print("V%d/V0 " % v + " | ".join("%s %.3f" % (nm, res[(v, nm)] / res[(0, nm)]) for nm in SH))
