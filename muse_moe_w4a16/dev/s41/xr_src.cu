// s41 "xr": T=128 BF16 GEMM out[N][M] (FP32, pre-zeroed) += W[N,K] . x[M,K]^T with a RESIDENT x K-slice per CTA.
// CTA (g, s): rows [g*NG, (g+1)*NG), k [s*KC, (s+1)*KC). x slice (M=128 tokens x KC) loaded once into smem (SW128 boxes
// 128 x 64), W streamed as 64x64 boxes through an NS-stage TMA ring, SS wgmma m64n64k16 (swap-AB: A = W, B = x; WG w
// takes tokens 64w..64w+63), after each 64-row tile the WG red.adds its 64x64 FP32 accumulator into out.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaTypedefs.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>

#ifndef XR_KB
#define XR_KB 8      // KC / 64
#endif
#ifndef XR_NS
#define XR_NS 10
#endif
#ifndef XR_WAIT1
#define XR_WAIT1 1   // keep one wgmma group in flight, release the previous stage
#endif
#ifndef XR_RED
#define XR_RED 2     // 0: no flush (diag), 2: red.v2
#endif

#ifndef XR_NOMMA
#define XR_NOMMA 0
#endif
#ifndef XR_LIN
#define XR_LIN 0     // W pre-packed [N/64][K/64][8 KB swizzled box]: one 1-D bulk copy per stage
#endif
#define M_TOK 128
#define XBOX 16384
#define WBOX 8192

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
__device__ __forceinline__ uint64_t dsw128(uint32_t a) {
  return (uint64_t)((a & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)64 << 32) | ((uint64_t)1 << 62);
}

#define WG_MMA(dA, dB, SCD)                                                                                         \
  asm volatile(                                                                                                    \
      "{\n .reg .pred p;\n setp.ne.b32 p, %34, 0;\n"                                                              \
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "                                                     \
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"\
      " %32, %33, p, 1, 1, 0, 0;\n}\n"                                                                            \
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),             \
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),       \
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),     \
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])      \
      : "l"(dA), "l"(dB), "r"(SCD))

__global__ void __launch_bounds__(288, 1)
xr_kernel(const __grid_constant__ CUtensorMap tmW, const __grid_constant__ CUtensorMap tmX, float* __restrict__ out,
          int NG, const uint8_t* __restrict__ wp, int KBT) {
  extern __shared__ __align__(1024) uint8_t smraw[];
  uint8_t* sm = (uint8_t*)(((uintptr_t)smraw + 1023) & ~(uintptr_t)1023);
  uint8_t* xs = sm;                                // XR_KB x 16 KB
  uint8_t* ws = sm + XR_KB * XBOX;                 // XR_NS x 8 KB
  uint64_t* bars = (uint64_t*)(ws + XR_NS * WBOX);
  uint64_t* xbar = bars;                           // XR_KB
  uint64_t* full = bars + XR_KB;                   // XR_NS
  uint64_t* empty = full + XR_NS;                  // XR_NS
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int row0 = blockIdx.x * NG, k0 = blockIdx.y * (XR_KB * 64);
  const int ntile = NG / 64, nbox = ntile * XR_KB;
  if (tid == 0) {
    for (int i = 0; i < XR_KB; i++) mbar_init(&xbar[i], 1);
    for (int i = 0; i < XR_NS; i++) { mbar_init(&full[i], 1); mbar_init(&empty[i], 2); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (warp == 8) {
    if (lane == 0) {
      // interleave: first W boxes and x boxes alternate so the first tile can start early
      for (int i = 0; i < nbox; i++) {
        if (i < XR_KB) { mbar_expect_tx(&xbar[i], XBOX); tma2d(xs + i * XBOX, &tmX, k0 + i * 64, 0, &xbar[i]); }
        const int st = i % XR_NS, ph = (i / XR_NS) & 1;
        mbar_wait(&empty[st], ph ^ 1);
        mbar_expect_tx(&full[st], WBOX);
#if XR_LIN
        bulk1d(ws + st * WBOX, wp + ((size_t)(row0 / 64 + i / XR_KB) * KBT + blockIdx.y * XR_KB + i % XR_KB) * WBOX,
               WBOX, &full[st]);
#else
        tma2d(ws + st * WBOX, &tmW, k0 + (i % XR_KB) * 64, row0 + (i / XR_KB) * 64, &full[st]);
#endif
      }
    }
    return;
  }
  const int wg = warp >> 2, wi = warp & 3;
  const uint32_t xb0 = smem_u32(xs) + wg * 8192, wb0 = smem_u32(ws);
  float d[32];
  int prev = -1;
  for (int t = 0; t < ntile; t++) {
#pragma unroll
    for (int j = 0; j < 32; j++) d[j] = 0.f;
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    for (int kb = 0; kb < XR_KB; kb++) {
      const int i = t * XR_KB + kb, st = i % XR_NS, ph = (i / XR_NS) & 1;
      if (t == 0) mbar_wait(&xbar[kb], 0);
      mbar_wait(&full[st], ph);
      const uint32_t a = wb0 + st * WBOX, b = xb0 + kb * XBOX;
#pragma unroll
      if (!XR_NOMMA) for (int k4 = 0; k4 < 4; k4++) { WG_MMA(dsw128(a + k4 * 32), dsw128(b + k4 * 32), 1); }
      asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
#if XR_WAIT1
      asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
      if (prev >= 0 && (tid & 127) == 0) mbar_arrive(&empty[prev]);
      prev = st;
#else
      asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
      if ((tid & 127) == 0) mbar_arrive(&empty[st]);
#endif
    }
#if XR_WAIT1
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    if ((tid & 127) == 0) mbar_arrive(&empty[prev]);
    prev = -1;
#endif
#pragma unroll
    for (int j = 0; j < 32; j++) asm volatile("" : "+f"(d[j]));
#if XR_RED == 2
    const int r = row0 + t * 64 + wi * 16 + (lane >> 2);
    float* o0 = out + (size_t)r * M_TOK + wg * 64 + 2 * (lane & 3);
    float* o1 = o0 + 8 * M_TOK;
#pragma unroll
    for (int j = 0; j < 8; j++) {
      asm volatile("red.global.add.v2.f32 [%0], {%1, %2};" :: "l"(o0 + 8 * j), "f"(d[4 * j]), "f"(d[4 * j + 1]) : "memory");
      asm volatile("red.global.add.v2.f32 [%0], {%1, %2};" :: "l"(o1 + 8 * j), "f"(d[4 * j + 2]), "f"(d[4 * j + 3]) : "memory");
    }
#elif XR_RED == 4
    {
      const int q = lane & 3, r = row0 + t * 64 + wi * 16 + (lane >> 2) + ((q & 1) ? 8 : 0);
      float* o = out + (size_t)r * M_TOK + wg * 64 + 2 * (q & 2);
#pragma unroll
      for (int j = 0; j < 8; j++) {
        const bool odd = q & 1;
        float s0 = odd ? d[4 * j] : d[4 * j + 2], s1 = odd ? d[4 * j + 1] : d[4 * j + 3];
        const float r0 = __shfl_xor_sync(0xffffffff, s0, 1), r1 = __shfl_xor_sync(0xffffffff, s1, 1);
        const float v0 = odd ? r0 : d[4 * j], v1 = odd ? r1 : d[4 * j + 1];
        const float v2 = odd ? d[4 * j + 2] : r0, v3 = odd ? d[4 * j + 3] : r1;
        asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(o + 8 * j), "f"(v0), "f"(v1), "f"(v2), "f"(v3)
                     : "memory");
      }
    }
#endif
  }
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

// out: float [N][128] (zeroed by the caller). G n-groups, S = K / (64*XR_KB) k-splits.
void xr_run(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t G, torch::Tensor wp) {
  const int N = w.size(0), K = w.size(1);
  TORCH_CHECK(x.size(0) == M_TOK && x.size(1) == K);
  const int S = K / (64 * XR_KB), NG = N / G;
  TORCH_CHECK(S * 64 * XR_KB == K && NG * G == N && NG % 64 == 0);
  CUtensorMap tw, tx;
  mk(&tw, w.data_ptr(), K, N, 64, 64);
  mk(&tx, x.data_ptr(), K, M_TOK, 64, 128);
  const int smem = XR_KB * XBOX + XR_NS * WBOX + 1024 + 512;
  static bool once = false;
  if (!once) { cudaFuncSetAttribute(xr_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); once = true; }
  xr_kernel<<<dim3(G, S), 288, smem, at::cuda::getCurrentCUDAStream()>>>(tw, tx, out.data_ptr<float>(), NG,
                                                                         (const uint8_t*)wp.data_ptr(), K / 64);
}
