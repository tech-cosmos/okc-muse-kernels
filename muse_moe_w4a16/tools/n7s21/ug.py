# n7s21 gate: T=256 unpool skip of zero-local tokens needs a GEMM whose per-CTA work scales with the kept count.
# ug<NT>: f[t][n] = sum_k x[t][k] W[n][k], swap-AB (A = W 128 rows via TMA SW128, B = x NT tokens via TMA SW128,
# wgmma m64nNTk16 SS), CTA = 128 W rows x NT tokens (token tile blockIdx.y), producer warp + 2 consumer WGs.
# Compare NT=128 (all 256 tokens, 2 tiles) vs cuBLAS, then NT = 104/112 (kept ~201-212 tokens in 2 tiles).
import torch, fcntl, statistics, sys, os
from torch.utils.cpp_extension import load_inline
NTS = [int(a) for a in os.environ.get("NTS", "128,112,104").split(",")]
NS = int(os.environ.get("NS", "6"))
MC = int(os.environ.get("MC", "1")); VAR = int(os.environ.get("VAR", "0"))

def mma(nt):
    n = nt // 2
    regs = ",".join(f"%{i}" for i in range(n))
    cons = ", ".join(f'"+f"(d[{i}])' for i in range(n))
    return f'''
__device__ __forceinline__ void mma{nt}(float* d, uint64_t a, uint64_t b, int sc) {{
  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, %{n+2}, 0;\\nwgmma.mma_async.sync.aligned.m64n{nt}k16.f32.bf16.bf16 {{{regs}}}, %{n}, %{n+1}, p, 1, 1, 0, 0;\\n}}\\n"
    : {cons} : "l"(a), "l"(b), "r"(sc));
}}'''

SRC = r'''
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>
#define NS_ @NS@
#define DBG_ @DBG@
#define XT_ @XT@
__device__ __forceinline__ uint32_t su(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint64_t* b, int c) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(su(b)), "r"(c)); }
__device__ __forceinline__ void mbar_tx(uint64_t* b, uint32_t tx) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(su(b)), "r"(tx) : "memory"); }
__device__ __forceinline__ void mbar_arrive(uint64_t* b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(su(b)) : "memory"); }
__device__ __forceinline__ bool mbar_try(uint64_t* b, uint32_t ph) {
  uint32_t ok; asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n" : "=r"(ok) : "r"(su(b)), "r"(ph) : "memory");
  return ok; }
#if DBG_
#define mbar_wait(b, ph) do { long long _c0 = clock64(); bool _p = false; while (!mbar_try(b, ph)) if (!_p && clock64() - _c0 > 2000000000LL) { _p = true; printf("HANG %s blk %d,%d tid %d ph %d\n", #b, blockIdx.x, blockIdx.y, threadIdx.x, (int)(ph)); } } while (0)
#define mbar_wait_cl mbar_wait
#else
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t ph) {
  asm volatile("{\n .reg .pred p;\n W1:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W1;\n}\n" :: "r"(su(b)), "r"(ph) : "memory"); }
#if !DBG_
__device__ __forceinline__ void mbar_wait_cl(uint64_t* b, uint32_t ph) {
  asm volatile("{\n .reg .pred p;\n W2:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W2;\n}\n" :: "r"(su(b)), "r"(ph) : "memory"); }
#endif
#endif
__device__ __forceinline__ void tma2d(void* dst, const CUtensorMap* m, int c0, int c1, uint64_t* b) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
      :: "r"(su(dst)), "l"(m), "r"(c0), "r"(c1), "r"(su(b)) : "memory"); }
__device__ __forceinline__ void tma2d_mc(void* dst, const CUtensorMap* m, int c0, int c1, uint64_t* b, uint16_t mask) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, %3}], [%4], %5;"
      :: "r"(su(dst)), "l"(m), "r"(c0), "r"(c1), "r"(su(b)), "h"(mask) : "memory"); }
__device__ __forceinline__ void arrive_remote(uint64_t* b, uint32_t rank) {
  asm volatile("{\n .reg .b32 ra;\n mapa.shared::cluster.u32 ra, %0, %1;\n mbarrier.arrive.shared::cluster.b64 _, [ra];\n}\n"
      :: "r"(su(b)), "r"(rank) : "memory"); }
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory"); }
#define MC_ @MC@
#define VAR_ @VAR@
#define WG1_ @WG1@
#define ST_ @ST@
__device__ __forceinline__ uint64_t dsw(uint32_t a) {
  return (uint64_t)((a & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)64 << 32) | ((uint64_t)1 << 62); }
@MMA@
template <int NT>
__global__ void __launch_bounds__(384, 1) ug(const __grid_constant__ CUtensorMap tw, const __grid_constant__ CUtensorMap tx,
                                             const __grid_constant__ CUtensorMap tx1, __nv_bfloat16* out, int N, int KS) {
  constexpr int H0 = ((NT / 2) + 7) / 8 * 8;  // MC 2: x rows loaded (and multicast) by cluster x-rank 0; rank 1 the rest
  extern __shared__ __align__(1024) uint8_t sm_raw[];
  uint8_t* sm = (uint8_t*)(((uintptr_t)sm_raw + 1023) & ~(uintptr_t)1023);
  constexpr int WB = 128 * 128, XB = NT * 128, SB = WB + XB;
  uint64_t* full = (uint64_t*)(sm + NS_ * SB);
  uint64_t* empty = full + NS_;
  const int tid = threadIdx.x, wg = tid / 128, n0 = blockIdx.x * 128, t0 = blockIdx.y * NT;
  if (tid == 0) {
    for (int s = 0; s < NS_; s++) { mbar_init(&full[s], 1); mbar_init(&empty[s], 2 * (MC_ == 2 ? 3 : MC_ ? 2 : 1)); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
#if MC_
  uint32_t rank; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
#if DBG_
  { uint32_t ccx, ccy, cn; asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(ccx)); asm volatile("mov.u32 %0, %%cluster_ctaid.y;" : "=r"(ccy)); asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(cn));
    if (tid == 0 && blockIdx.x < 2) printf("RANK blk %d,%d rank %u ctaid %u,%u n %u\n", blockIdx.x, blockIdx.y, rank, ccx, ccy, cn); }
#endif
  cluster_sync();
#endif
  if (wg == 0) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
    if (tid == 0) {
      for (int i = 0; i < KS; i++) {
        const int s = i % NS_;
#if MC_ == 2
        if (i >= NS_) mbar_wait_cl(&empty[s], ((i / NS_) - 1) & 1);
        mbar_tx(&full[s], SB);
        {
          const uint32_t cx = rank & 1, cy = rank >> 1;
          tma2d_mc(sm + s * SB + cy * 8192, &tw, i * 64, n0 + cy * 64, &full[s], (uint16_t)((1u << cx) | (1u << (cx + 2))));
#if XT_ == 1
          tma2d(sm + s * SB + WB, &tx, i * 64, t0, &full[s]); tma2d(sm + s * SB + WB + H0 * 128, &tx1, i * 64, t0 + H0, &full[s]);
#else
          if (cx) tma2d_mc(sm + s * SB + WB + H0 * 128, &tx1, i * 64, t0 + H0, &full[s], (uint16_t)(3u << (2 * cy)));
          else tma2d_mc(sm + s * SB + WB, &tx, i * 64, t0, &full[s], (uint16_t)(3u << (2 * cy)));
#endif
        }
#elif MC_
        if (i >= NS_) mbar_wait_cl(&empty[s], ((i / NS_) - 1) & 1);
        mbar_tx(&full[s], SB);
        tma2d_mc(sm + s * SB + rank * 8192, &tw, i * 64, n0 + rank * 64, &full[s], 3);
        tma2d(sm + s * SB + WB, &tx, i * 64, t0, &full[s]);
#else
        if (i >= NS_) mbar_wait(&empty[s], ((i / NS_) - 1) & 1);
        mbar_tx(&full[s], SB);
        tma2d(sm + s * SB, &tw, i * 64, n0, &full[s]);
        tma2d(sm + s * SB + WB, &tx, i * 64, t0, &full[s]);
#endif
      }
    }
    __syncwarp();
#if MC_
    cluster_sync();
#endif
    return;
  }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
  const int c = wg - 1, t = tid & 127;
  float d[NT / 2];
#pragma unroll
  for (int j = 0; j < NT / 2; j++) asm volatile("mov.b32 %0, 0;" : "=f"(d[j]));  // (scale-d 0 on the first mma)
#pragma unroll
  for (int j = 0; j < NT / 2; j++) asm volatile("" : "+f"(d[j]) :: "memory");
#pragma unroll 1
  for (int i = 0; i < KS; i++) {
    const int s = i % NS_;
    mbar_wait(&full[s], (i / NS_) & 1);
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    const uint32_t a = su(sm + s * SB) + c * 8192, b = su(sm + s * SB + WB);
#pragma unroll
    for (int j = 0; j < 4; j++) mma@NT@_dispatch<NT>(d, dsw(a + 32 * j), dsw(b + 32 * j), (int)(i > 0 || j > 0));
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(WG1_) : "memory");
#if MC_ == 2
    if (i > 0 && t == 0) { arrive_remote(&empty[(i - 1) % NS_], rank); arrive_remote(&empty[(i - 1) % NS_], rank ^ 1); arrive_remote(&empty[(i - 1) % NS_], rank ^ 2); }
#elif MC_
    if (i > 0 && t == 0) { arrive_remote(&empty[(i - 1) % NS_], 0); arrive_remote(&empty[(i - 1) % NS_], 1); }
#else
    if (i > 0 && t == 0) mbar_arrive(&empty[(i - 1) % NS_]);
#endif
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
#if DBG_
  if (t == 0) printf("CDONE blk %d,%d wg %d\n", blockIdx.x, blockIdx.y, wg);
#endif
#pragma unroll
  for (int j = 0; j < NT / 2; j++) asm volatile("" : "+f"(d[j]) :: "memory");
  if (VAR_ == 4) { float acc = 0.f;
#pragma unroll
    for (int j = 0; j < NT / 2; j++) acc += d[j];
    if (acc == 1234.5f) out[tid] = __float2bfloat16_rn(acc);
#if MC_
    cluster_sync();
#endif
    return; }
  if (VAR_ == 5) { if (d[0] + d[NT / 2 - 1] == 1234.5f) out[tid] = __float2bfloat16_rn(d[1]);
#if MC_
    cluster_sync();
#endif
    return; }
  if (VAR_ == 1) { if (d[0] == 1234.5f) out[tid] = __float2bfloat16_rn(d[1]);
#if MC_
    cluster_sync();
#endif
    return; }
  if (VAR_ == 3) {
    const int w = t / 32, l = t & 31, r = c * 64 + w * 16 + l / 4;
#pragma unroll
    for (int j = 0; j < NT / 8; j++) {
      const int col = j * 8 + (l & 3) * 2;
      out[(size_t)(t0 + col) * N + n0 + r] = __float2bfloat16_rn(d[4 * j]);
      out[(size_t)(t0 + col + 1) * N + n0 + r] = __float2bfloat16_rn(d[4 * j + 1]);
      out[(size_t)(t0 + col) * N + n0 + r + 8] = __float2bfloat16_rn(d[4 * j + 2]);
      out[(size_t)(t0 + col + 1) * N + n0 + r + 8] = __float2bfloat16_rn(d[4 * j + 3]);
    }
#if MC_
    cluster_sync();
#endif
    return;
  }
  asm volatile("bar.sync 1, 256;" ::: "memory");  // all stages consumed: reuse stage smem for the epilogue
  __nv_bfloat16* stg = (__nv_bfloat16*)sm;  // [NT][136]
  float e[NT / 2];  // copies (ptxas C7514: converting d in place serializes every wgmma)
#pragma unroll
  for (int j = 0; j < NT / 2; j++) asm volatile("mov.b32 %0, %1;" : "=f"(e[j]) : "f"(d[j]));
  const int w = t / 32, l = t & 31, r = c * 64 + w * 16 + l / 4;
#pragma unroll
  for (int j = 0; j < NT / 8; j++) {
    uint32_t p0, p1;
    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(p0) : "f"(e[4 * j + 1]), "f"(e[4 * j]));
    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(p1) : "f"(e[4 * j + 3]), "f"(e[4 * j + 2]));
    // stmatrix.trans: matrix 0 = features w*16+0..7 x tokens j*8..+7 (p0), matrix 1 = features +8..15 (p1)
    const int li = l & 7, hi8 = (l >> 3) & 1;
    const uint32_t ad = su(stg + (j * 8 + li) * 136 + c * 64 + w * 16 + hi8 * 8);
    asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};" :: "r"(ad), "r"(p0), "r"(p1) : "memory");
  }
  asm volatile("bar.sync 1, 256;" ::: "memory");
  const int u = tid - 128;
  for (int q = u; q < NT * 16; q += 256) {
    const int tk = q >> 4, cc = q & 15;
    *(uint4*)(out + (size_t)(t0 + tk) * N + n0 + cc * 8) = *(const uint4*)(stg + tk * 136 + cc * 8);
  }
#if MC_
  cluster_sync();
#endif
}
static PFN_cuTensorMapEncodeTiled_v12000 g_enc = nullptr;
static void mk(CUtensorMap* m, const void* p, uint64_t inner, uint64_t outer, uint32_t b0, uint32_t b1) {
  if (!g_enc) { cudaDriverEntryPointQueryResult q; cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", (void**)&g_enc, 12000, cudaEnableDefault, &q); }
  cuuint64_t dims[2] = {inner, outer}, strides[1] = {inner * 2};
  cuuint32_t box[2] = {b0, b1}, es[2] = {1, 1};
  CUresult r = g_enc(m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)p, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                     CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "encode ", (int)r);
}
template <int NT>
void run(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t ntile) {
  const int K = x.size(1), N = w.size(0);
  CUtensorMap tw, tx, tx1;
  constexpr int H0 = ((NT / 2) + 7) / 8 * 8;
  mk(&tw, w.data_ptr(), K, N, 64, MC_ ? 64 : 128);
  mk(&tx, x.data_ptr(), K, x.size(0), 64, MC_ == 2 ? H0 : NT);
  mk(&tx1, x.data_ptr(), K, x.size(0), 64, MC_ == 2 ? NT - H0 : NT);
  const int smem = NS_ * (128 * 128 + NT * 128) + 2048;
  static bool once = false;
  if (!once) { cudaFuncSetAttribute(ug<NT>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); once = true; }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(N / 128, ntile); cfg.blockDim = dim3(384); cfg.dynamicSmemBytes = smem; cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = MC_ == 2 ? 2 : 1; at[0].val.clusterDim.y = MC_ ? 2 : 1; at[0].val.clusterDim.z = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, ug<NT>, tw, tx, tx1, (__nv_bfloat16*)out.data_ptr(), N, K / 64) == cudaSuccess, "launch");
}
void ugemm(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t nt, int64_t ntile) {
  switch (nt) { @CASES@ default: TORCH_CHECK(false, "nt"); }
}
'''
def stage(nt):
    n = nt // 2
    regs = ",".join(f"%{i}" for i in range(n))
    cons = ", ".join(f'"+f"(d[{i}])' for i in range(n))
    body = "wgmma.fence.sync.aligned;\\n"
    for j in range(4):
        body += f"wgmma.mma_async.sync.aligned.m64n{nt}k16.f32.bf16.bf16 {{{regs}}}, %{n+2*j}, %{n+2*j+1}, {'p' if j == 0 else '1'}, 1, 1, 0, 0;\\n"
    body += "wgmma.commit_group.sync.aligned;\\n"
    ins = ", ".join(f'"l"(a{j}), "l"(b{j})' for j in range(4))
    return f"""
__device__ __forceinline__ void stage{nt}(float* d, uint64_t a0, uint64_t b0, uint64_t a1, uint64_t b1, uint64_t a2, uint64_t b2, uint64_t a3, uint64_t b3, int sc) {{
  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, %{n+8}, 0;\\n{body}}}\\n" : {cons} : {ins}, "r"(sc) : "memory");
}}"""
mmas = "\n".join(mma(nt) + stage(nt) for nt in NTS)
disp2 = "template <int NT> __device__ __forceinline__ void stage_dispatch(float* d, uint32_t a, uint32_t b, int sc) {\n" + \
    "".join(f"  if constexpr (NT == {nt}) stage{nt}(d, dsw(a), dsw(b), dsw(a + 32), dsw(b + 32), dsw(a + 64), dsw(b + 64), dsw(a + 96), dsw(b + 96), sc);\n" for nt in NTS) + "}\n"
disp = "template <int NT> __device__ __forceinline__ void mma_dispatch(float* d, uint64_t a, uint64_t b, int sc) {\n" + \
    "".join(f"  if constexpr (NT == {nt}) mma{nt}(d, a, b, sc);\n" for nt in NTS) + "}\n"
src = SRC.replace("@NS@", str(NS)).replace("@MC@", str(MC)).replace("@DBG@", os.environ.get("DBG", "0")).replace("@XT@", os.environ.get("XT", "0")).replace("@VAR@", str(VAR)).replace("@WG1@", os.environ.get("WG1", "1")).replace("@ST@", os.environ.get("ST", "1")).replace("@MMA@", mmas + disp + disp2).replace("mma@NT@_dispatch<NT>", "mma_dispatch<NT>") \
    .replace("@CASES@", " ".join(f"case {nt}: run<{nt}>(x, w, out, ntile); break;" for nt in NTS))
if os.environ.get("GEN"):
    open(os.environ["GEN"], "w").write(src); sys.exit(0)
lib = load_inline(f"ug_n7s21_{'_'.join(map(str, NTS))}_{NS}_{MC}_{VAR}_{os.environ.get('WG1', '1')}_{os.environ.get('ST', '1')}_{os.environ.get('DBG', '0')}_{os.environ.get('XT', '0')}", cpp_sources="void ugemm(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t nt, int64_t ntile);",
                  cuda_sources=src, functions=["ugemm"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo", "-Xptxas=-v"],
                  extra_ldflags=["-lcuda"], verbose=False)

torch.manual_seed(0)
NW = int(os.environ.get("NW", "8192")); KK = int(os.environ.get("KK", "4096")); WARM = int(os.environ.get("WARM", "0"))
W = (torch.randn(NW, KK, device="cuda") * 0.02).bfloat16()
x = torch.randn(256, KK, device="cuda").bfloat16()
ref = (x.float() @ W.float().t())
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
def tm(fn, reps=60):
    for _ in range(3): fn()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    ts = []
    with open("/tmp/okc-gpu.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for _ in range(reps):
            (None if WARM else flush.zero_()); a = torch.cuda.Event(True); b = torch.cuda.Event(True)
            a.record(); g.replay(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b) * 1e3)
        fcntl.flock(lk, fcntl.LOCK_UN)
    return statistics.median(ts)
o = torch.empty(256, NW, device="cuda", dtype=torch.bfloat16)
for nt in NTS:
    o.zero_(); lib.ugemm(x, W, o, nt, 2); torch.cuda.synchronize()
    rows = 2 * nt
    err = (o[:rows].float() - ref[:rows]).abs().max().item()
    print(f"NT{nt}: rows {rows} max err {err:.3e} (bf16 of ref ~{ref.abs().max().item():.2f})")
for rnd in range(2):
    s = f"cuBLAS256 {tm(lambda: torch.mm(x, W.t(), out=o)):.1f}"
    for nt in NTS:
        s += f" | ug NT{nt} {tm(lambda: lib.ugemm(x, W, o, nt, 2)):.1f}"
    print(s, flush=True)
