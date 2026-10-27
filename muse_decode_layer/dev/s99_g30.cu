// s99 "g30": T=128 GEMM out[T][N] = x[T,K] . W[N,K]^T over proj's lossless 12-bit BF16 weights (DEC3 dec8 units),
// W decoded straight into wgmma register A-fragments (swap-AB: A = W rows, B = x via TMA SW128, n = 128 tokens).
// CTA = (64-row group g, k-split s): 2 consumer WGs take alternate 64-k stages of the same 64 rows (smem-summed at the
// end) + 1 producer warp (1-D bulk copy of the 6 KB packed W stage + 2-D TMA of the 16 KB x box per stage).
// Escapes NOT handled (bound prototype). S == 1: BF16 out [T][N] = scale * acc; S > 1: FP32 slabs [S][T][N].
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaTypedefs.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>

#ifndef G30_NS
#define G30_NS 10
#endif
#ifndef G30_NOMMA
#define G30_NOMMA 0
#endif
#ifndef G30_NOW
#define G30_NOW 0
#endif
#ifndef G30_NOX
#define G30_NOX 0
#endif
#ifndef G30_EPI
#define G30_EPI 1
#endif
#ifndef G30_CL
#define G30_CL 1   // cluster size along N: x box multicast, 128/CL token rows per CTA
#endif
#ifndef G30_W0
#define G30_W0 0
#endif
#ifndef G30_MCOFF
#define G30_MCOFF 0
#endif
#ifndef G30_LOCALREL
#define G30_LOCALREL 0
#endif
#ifndef G30_NW
#define G30_NW 10   // W ring stages (own producer warp); G30_NS = x ring stages (s99 best NS10/NW10)
#endif
#ifndef G30_ROT
#define G30_ROT 0   // CTA g starts its k loop at stage g * KS / gridDim.x (spreads x L2 hot spots)
#endif
#ifndef G30_R2
#define G30_R2 0   // CTA = 128 W rows: WG w takes row group 2g + w over all k (x box shared; no WG reduce)
#endif
#define WSLOT (WST * (1 + G30_R2))
#define M_TOK 128
#define XBOX 16384
#define WST 6144

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint64_t* b, int c) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(c));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(tx) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t ph) {
  asm volatile(
      "{\n .reg .pred p;\n LAB_WAIT:\n"
      " mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      " @!p bra LAB_WAIT;\n}\n" :: "r"(smem_u32(b)), "r"(ph) : "memory");
}
__device__ __forceinline__ void tma2d(void* dst, const CUtensorMap* m, int c0, int c1, uint64_t* b) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
      :: "r"(smem_u32(dst)), "l"(m), "r"(c0), "r"(c1), "r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void bulk1d(void* dst, const void* src, uint32_t n, uint64_t* b) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(smem_u32(dst)), "l"(src), "r"(n), "r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void tma2d_mc(void* dst, const CUtensorMap* m, int c0, int c1, uint64_t* b, uint16_t mask) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, %3}], [%4], %5;"
      :: "r"(smem_u32(dst)), "l"(m), "r"(c0), "r"(c1), "r"(smem_u32(b)), "h"(mask) : "memory");
}
__device__ __forceinline__ void mbar_arrive_cl(uint64_t* b, uint32_t cta) {
  asm volatile("{\n .reg .b32 ra;\n mapa.shared::cluster.u32 ra, %0, %1;\n"
               " mbarrier.arrive.shared::cluster.b64 _, [ra];\n}\n" :: "r"(smem_u32(b)), "r"(cta) : "memory");
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ uint64_t dsw128(uint32_t a) {
  return (uint64_t)((a & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)64 << 32) | ((uint64_t)1 << 62);
}
__device__ __forceinline__ unsigned lop_aob(unsigned a, unsigned b, unsigned c) {
  unsigned r;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}
template <int P>
__device__ __forceinline__ unsigned decw(unsigned sw, unsigned nw, unsigned sel3) {
  const unsigned sh = P == 0 ? nw : P == 1 ? nw << 4 : P == 2 ? nw << 8 : __funnelshift_r(nw, nw, 4);
  const unsigned k38 = 0x38383838u >> (gridDim.y >> 16);
  unsigned p;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(p) : "r"(sw), "r"(k38), "r"(sel3));
  return lop_aob(sh, 0x87008700u, p);
}

#define D8(i) "+f"(d[i]), "+f"(d[i + 1]), "+f"(d[i + 2]), "+f"(d[i + 3]), "+f"(d[i + 4]), "+f"(d[i + 5]), "+f"(d[i + 6]), "+f"(d[i + 7])
__device__ __forceinline__ void wg_mma_rs(float* d, unsigned a0, unsigned a1, unsigned a2, unsigned a3, uint64_t db) {
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
      "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},"
      " {%64,%65,%66,%67}, %68, 1, 1, 1, 0;\n"
      : D8(0), D8(8), D8(16), D8(24), D8(32), D8(40), D8(48), D8(56)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
}

__global__ void __launch_bounds__(320, 1)
g30_kernel(const __grid_constant__ CUtensorMap tmX, const uint8_t* __restrict__ wp, void* __restrict__ out, int N,
           int KS, int S, float scale) {
  // KS = k-stages per CTA; wp = [G][S*KS][WST]
  extern __shared__ __align__(1024) uint8_t smraw[];
  uint8_t* sm = (uint8_t*)(((uintptr_t)smraw + 1023) & ~(uintptr_t)1023);
  uint8_t* xs = sm;                                  // NS x 16 KB
  uint8_t* ws = sm + G30_NS * XBOX;                  // NS x 6 KB
  uint64_t* full = (uint64_t*)(ws + G30_NW * WSLOT);   // x ring
  uint64_t* empty = full + G30_NS;
  uint64_t* wfull = empty + G30_NS;                  // W ring
  uint64_t* wempty = wfull + G30_NW;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = blockIdx.x, s = blockIdx.y;
  if (tid == 0) {
    for (int i = 0; i < G30_NS; i++) { mbar_init(&full[i], 1); mbar_init(&empty[i], G30_R2 ? 2 : G30_LOCALREL ? 1 : G30_CL); }
    for (int i = 0; i < G30_NW; i++) { mbar_init(&wfull[i], 1); mbar_init(&wempty[i], 1 + G30_R2); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
#if G30_CL > 1
  cluster_sync();
  const uint32_t crank = blockIdx.x % G30_CL;
#endif
  const int rot = G30_ROT ? (int)(((long)g * KS) / gridDim.x) : 0;
  if (warp == 8) {
    if (lane == 0 && !G30_NOW) {
      const uint8_t* src = wp + ((size_t)g * (1 + G30_R2) * S * KS + (size_t)s * KS) * WST;
      for (int i = 0; i < KS; i++) {
        const int st = i % G30_NW, ph = (i / G30_NW) & 1;
        mbar_wait(&wempty[st], ph ^ 1);
        mbar_expect_tx(&wfull[st], WSLOT);
        bulk1d(ws + st * WSLOT, src + (size_t)((i + rot) % KS) * WST, WST, &wfull[st]);
        if (G30_R2) bulk1d(ws + st * WSLOT + WST, src + ((size_t)S * KS + (i + rot) % KS) * WST, WST, &wfull[st]);
      }
    }
    goto done;
  }
  if (warp == 9) {
    if (lane == 0 && !G30_NOX) {
      for (int i = 0; i < KS; i++) {
        const int st = i % G30_NS, ph = (i / G30_NS) & 1;
        mbar_wait(&empty[st], ph ^ 1);
        mbar_expect_tx(&full[st], XBOX);
#if G30_CL > 1 && G30_MCOFF
        tma2d(xs + st * XBOX, &tmX, (s * KS + (i + rot) % KS) * 64, 0, &full[st]);
#elif G30_CL > 1
        tma2d_mc(xs + st * XBOX + crank * (XBOX / G30_CL), &tmX, (s * KS + (i + rot) % KS) * 64, crank * (M_TOK / G30_CL), &full[st],
                 (uint16_t)((1u << G30_CL) - 1));
#else
        tma2d(xs + st * XBOX, &tmX, (s * KS + (i + rot) % KS) * 64, 0, &full[st]);
#endif
      }
    }
    goto done;
  }
  {
  const int wg = warp >> 2, t = tid & 127;
  float d[64];
#pragma unroll
  for (int j = 0; j < 64; j++) d[j] = 0.f;
#if G30_CL > 1 && !G30_LOCALREL
#define RELEASE(st_) mbar_arrive_cl(&empty[st_], t)
#else
#define RELEASE(st_) mbar_arrive(&empty[st_])
#endif
  unsigned A0[16], A1[16];
  int prev = -1;
#define DEC(a, j, S0, S1, NW)                                \
    a[4 * j + 0] = decw<0>(S0, NW, 0x4140u);                 \
    a[4 * j + 1] = decw<1>(S0, NW, 0x4342u);                 \
    a[4 * j + 2] = decw<2>(S1, NW, 0x4140u);                 \
    a[4 * j + 3] = decw<3>(S1, NW, 0x4342u);
#define STEP(a, pa)                                                                                         \
  {                                                                                                         \
    const int st = i % G30_NS, ph = (i / G30_NS) & 1, sw = i % G30_NW, pw = (i / G30_NW) & 1;              \
    if (!G30_NOW) mbar_wait(&wfull[sw], pw);                                                                \
    if (!G30_NOX) mbar_wait(&full[st], ph);                                                                 \
    const uint8_t* wst = ws + sw * WSLOT + (G30_R2 ? wg * WST : 0);                                                                  \
    const uint4 s01 = *(const uint4*)(wst + t * 16), s23 = *(const uint4*)(wst + 2048 + t * 16);            \
    const uint4 nw = *(const uint4*)(wst + 4096 + t * 16);                                                  \
    DEC(a, 0, s01.x, s01.y, nw.x) DEC(a, 1, s01.z, s01.w, nw.y) DEC(a, 2, s23.x, s23.y, nw.z) DEC(a, 3, s23.z, s23.w, nw.w) \
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");                                                 \
    const uint32_t b = smem_u32(xs + st * XBOX);                                                            \
    if (!G30_NOMMA) {                                                                                       \
      _Pragma("unroll") for (int j = 0; j < 4; j++) wg_mma_rs(d, a[4 * j], a[4 * j + 1], a[4 * j + 2], a[4 * j + 3], dsw128(b + j * 32)); \
    }                                                                                                       \
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");                                          \
    if (G30_W0) {                                                                                           \
      asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");                                        \
      _Pragma("unroll") for (int j = 0; j < 16; j++) asm volatile("" : "+r"(a[j]));                         \
      if (t < (G30_LOCALREL ? 1 : G30_CL)) RELEASE(st);                                                     \
      if (t == 0) mbar_arrive(&wempty[sw]);                                                                          \
    } else {                                                                                                \
    asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");                                          \
    _Pragma("unroll") for (int j = 0; j < 16; j++) asm volatile("" : "+r"(pa[j]));                          \
    if (prev >= 0 && t < (G30_LOCALREL ? 1 : G30_CL)) RELEASE(prev % G30_NS);                               \
    if (prev >= 0 && t == 0) mbar_arrive(&wempty[prev % G30_NW]);                                                             \
    prev = i;                                                                                              \
    }                                                                                                       \
    i += 2 - G30_R2;                                                                                               \
  }
  int i = G30_R2 ? 0 : wg;
  while (i < KS) {
    STEP(A0, A1)
    if (i >= KS) break;
    STEP(A1, A0)
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
#pragma unroll
  for (int j = 0; j < 16; j++) asm volatile("" : "+r"(A0[j]), "+r"(A1[j]));
#pragma unroll
  for (int j = 0; j < 64; j++) asm volatile("" : "+f"(d[j]));
  if (prev >= 0 && t < (G30_LOCALREL ? 1 : G30_CL)) RELEASE(prev % G30_NS);
  if (prev >= 0 && t == 0) mbar_arrive(&wempty[prev % G30_NW]);
  // reduce WG1 into WG0 through smem (x ring is free once both WGs are done: barrier over 256 consumer threads)
  float* red = (float*)xs;
  asm volatile("bar.sync 1, 256;" ::: "memory");
  const int gq = G30_R2 ? 2 * g + wg : g;
  uint8_t* stgb = G30_R2 ? xs + wg * 40960 : xs + 32768;
  if (!G30_R2) {
  if (wg == 1) {
#pragma unroll
    for (int j = 0; j < 64; j++) red[j * 128 + t] = d[j];
  }
  asm volatile("bar.sync 1, 256;" ::: "memory");
  if (wg == 1) goto done;
#pragma unroll
  for (int j = 0; j < 64; j++) d[j] += red[j * 128 + t];
  }
  // acc layout (m64n128): d[4*jn + 2*h + e] = row (warp*16 + lane/4 + 8h), token 8*jn + 2*(lane&3) + e
  const int row = gq * 64 + (warp & 3) * 16 + (lane >> 2);
  const int tok = 2 * (lane & 3);
#if G30_EPI
  // stage the tile token-major in smem, then coalesced 16 B stores (128 B / 256 B per token row)
  const int r0 = (warp & 3) * 16 + (lane >> 2);
  if (S == 1) {
    __nv_bfloat16* stg = (__nv_bfloat16*)stgb;  // [128 tok][64 rows] (+8 pad)
#pragma unroll
    for (int jn = 0; jn < 16; jn++)
#pragma unroll
      for (int h = 0; h < 2; h++)
#pragma unroll
        for (int e = 0; e < 2; e++) stg[(8 * jn + tok + e) * 72 + r0 + 8 * h] = __float2bfloat16(d[4 * jn + 2 * h + e] * scale);
    asm volatile("bar.sync %0, 128;" :: "r"(2 + wg) : "memory");
    __nv_bfloat16* o = (__nv_bfloat16*)out;
#pragma unroll
    for (int q = 0; q < 8; q++) {
      const int idx = q * 128 + t, tk = idx >> 3, c = idx & 7;
      *(uint4*)(o + (size_t)tk * N + gq * 64 + c * 8) = *(const uint4*)(stg + tk * 72 + c * 8);
    }
  } else {
    float* stg = (float*)stgb;  // [128 tok][64 rows] (+4 pad)
#pragma unroll
    for (int jn = 0; jn < 16; jn++)
#pragma unroll
      for (int h = 0; h < 2; h++)
#pragma unroll
        for (int e = 0; e < 2; e++) stg[(8 * jn + tok + e) * 68 + r0 + 8 * h] = d[4 * jn + 2 * h + e] * scale;
    asm volatile("bar.sync %0, 128;" :: "r"(2 + wg) : "memory");
    float* o = (float*)out + (size_t)s * M_TOK * N;
#pragma unroll
    for (int q = 0; q < 16; q++) {
      const int idx = q * 128 + t, tk = idx >> 4, c = idx & 15;
      *(float4*)(o + (size_t)tk * N + gq * 64 + c * 4) = *(const float4*)(stg + tk * 68 + c * 4);
    }
  }
#else
  if (S == 1) {
    __nv_bfloat16* o = (__nv_bfloat16*)out;
#pragma unroll
    for (int jn = 0; jn < 16; jn++)
#pragma unroll
      for (int h = 0; h < 2; h++)
#pragma unroll
        for (int e = 0; e < 2; e++)
          o[(size_t)(8 * jn + tok + e) * N + row + 8 * h] = __float2bfloat16(d[4 * jn + 2 * h + e] * scale);
  } else {
    float* o = (float*)out + (size_t)s * M_TOK * N;
#pragma unroll
    for (int jn = 0; jn < 16; jn++)
#pragma unroll
      for (int h = 0; h < 2; h++)
#pragma unroll
        for (int e = 0; e < 2; e++)
          o[(size_t)(8 * jn + tok + e) * N + row + 8 * h] = d[4 * jn + 2 * h + e] * scale;
  }
#endif
  }
done:
#if G30_CL > 1
  __syncwarp();
  cluster_sync();
#endif
  return;
}

static PFN_cuTensorMapEncodeTiled_v12000 g_enc = nullptr;
static void mk(CUtensorMap* m, const void* p, uint64_t inner, uint64_t outer, uint32_t b0, uint32_t b1) {
  if (!g_enc) {
    cudaDriverEntryPointQueryResult q;
    cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", (void**)&g_enc, 12000, cudaEnableDefault, &q);
  }
  cuuint64_t dims[2] = {inner, outer}, strides[1] = {inner * 2};
  cuuint32_t box[2] = {b0, b1}, es[2] = {1, 1};
  CUresult r = g_enc(m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)p, dims, strides, box, es,
                     CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                     CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "encode failed ", (int)r);
}

void g30_run(torch::Tensor x, torch::Tensor wp, torch::Tensor out, int64_t N, int64_t S, double scale) {
  const int K = x.size(1);
  TORCH_CHECK(x.size(0) == M_TOK && K % (64 * S) == 0 && N % 64 == 0);
  const int KS = K / 64 / S;
  CUtensorMap tx;
  mk(&tx, x.data_ptr(), K, M_TOK, 64, G30_MCOFF ? 128 : 128 / G30_CL);
  const int smem = G30_NS * XBOX + G30_NW * WSLOT + 1024 + 512;
  static bool once = false;
  if (!once) { cudaFuncSetAttribute(g30_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); once = true; }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(N / 64 / (1 + G30_R2), S);
  cfg.blockDim = dim3(320);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = G30_CL; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  static bool pr = false;
  if (!pr) { int nc = -1; cudaOccupancyMaxActiveClusters(&nc, g30_kernel, &cfg); printf("g30 CL=%d max active clusters %d (grid %d)\n", G30_CL, nc, N / 64 * (int)S); pr = true; }
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, g30_kernel, tx, (const uint8_t*)wp.data_ptr(), out.data_ptr(), (int)N, KS, (int)S,
                                 (float)scale) == cudaSuccess, "g30 launch failed");
}
