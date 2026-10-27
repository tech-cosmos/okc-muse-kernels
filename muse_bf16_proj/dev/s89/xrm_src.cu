// s89 "xrm": moe s41 xr (resident x K-slice, W streamed, swap-AB SS wgmma, FP32 red.add into out[N][128]) with the
// tokens split over a CLUSTER of C CTAs: CTA rank r owns tokens [r*MT, (r+1)*MT), MT = 128/C, so its resident x slice
// covers C times more k (KC = 64*XR_KB) and the split-K count S (= red volume) drops C times. Every W box goes to all
// C CTAs by TMA multicast (rank r issues rows [r*64/C, (r+1)*64/C) of each box); stage reuse is cluster-wide (empty
// barriers count one arrive per CTA, arrived remotely). One consumer warpgroup (m64nMTk16) + one producer warp.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaTypedefs.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>

#ifndef XR_C
#define XR_C 2
#endif
#ifndef XR_KB
#define XR_KB (8 * XR_C)  // KC / 64
#endif
#ifndef XR_NS
#define XR_NS 8
#endif
#ifndef XR_RED
#define XR_RED 2          // 0: no flush (diag), 2: red.global.add.v2, 3: tile -> smem -> cp.reduce.async.bulk add.f32
#endif
#ifndef XR_OB
#define XR_OB 1           // RED 3: smem out buffers (1 or 2); RED 4: tile -> smem -> 4 red warps (red.global.add.v4), MMA WG continues
#endif
#ifndef XR_NOMMA
#define XR_NOMMA 0
#endif
#ifndef XR_NOMC
#define XR_NOMC 0         // diag: every CTA loads whole W boxes itself (no multicast; empty still cluster-wide)
#endif
#define MT (128 / XR_C)
#define XBOX (MT * 128)
#define WBOX 8192
#define WROWS (64 / XR_C)
#define NACC (MT / 2)
#define OROW (MT * 4 + 16)  // padded smem out row (bytes): conflict-free float2 stores, 16 B aligned bulk sources
#define OBUF (64 * OROW)

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint64_t* b, int c) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(c));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(tx) : "memory");
}
__device__ __forceinline__ void mbar_arrive_cl(uint64_t* b, uint32_t cta) {
  asm volatile("{\n .reg .b32 ra;\n mapa.shared::cluster.u32 ra, %0, %1;\n"
               " mbarrier.arrive.shared::cluster.b64 _, [ra];\n}" :: "r"(smem_u32(b)), "r"(cta) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t ph) {
  asm volatile(
      "{\n .reg .pred p;\n LAB_WAIT:\n"
      " mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      " @!p bra LAB_WAIT;\n}\n" :: "r"(smem_u32(b)), "r"(ph) : "memory");
}
__device__ __forceinline__ void mbar_wait_cl(uint64_t* b, uint32_t ph) {
  asm volatile(
      "{\n .reg .pred p;\n LAB_WAITC:\n"
      " mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\n"
      " @!p bra LAB_WAITC;\n}\n" :: "r"(smem_u32(b)), "r"(ph) : "memory");
}
__device__ __forceinline__ void tma2d(void* dst, const CUtensorMap* m, int c0, int c1, uint64_t* b) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
      :: "r"(smem_u32(dst)), "l"(m), "r"(c0), "r"(c1), "r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void tma2d_mc(void* dst, const CUtensorMap* m, int c0, int c1, uint64_t* b, uint16_t mask) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
      " [%0], [%1, {%2, %3}], [%4], %5;"
      :: "r"(smem_u32(dst)), "l"(m), "r"(c0), "r"(c1), "r"(smem_u32(b)), "h"(mask) : "memory");
}
__device__ __forceinline__ uint64_t dsw128(uint32_t a) {
  return (uint64_t)((a & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)64 << 32) | ((uint64_t)1 << 62);
}
__device__ __forceinline__ uint32_t cl_rank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
__device__ __forceinline__ void cl_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}

#if MT == 64
#define WG_MMA(dA, dB)                                                                                              \
  asm volatile(                                                                                                    \
      "{\n wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "                                                 \
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"\
      " %32, %33, 1, 1, 1, 0, 0;\n}\n"                                                                            \
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),             \
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),       \
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),     \
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])      \
      : "l"(dA), "l"(dB))
#elif MT == 32
#define WG_MMA(dA, dB)                                                                                              \
  asm volatile(                                                                                                    \
      "{\n wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "                                                 \
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, %16, %17, 1, 1, 1, 0, 0;\n}\n"                  \
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),             \
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])        \
      : "l"(dA), "l"(dB))
#else
#error "MT must be 64 or 32"
#endif

#define NTHR (XR_RED == 4 ? 288 : 160)
__global__ void __launch_bounds__(NTHR, 1)
xrm_kernel(const __grid_constant__ CUtensorMap tmW, const __grid_constant__ CUtensorMap tmX, float* __restrict__ out,
           int NG) {
  extern __shared__ __align__(1024) uint8_t smraw[];
  uint8_t* sm = (uint8_t*)(((uintptr_t)smraw + 1023) & ~(uintptr_t)1023);
  uint8_t* xs = sm;                                // XR_KB x XBOX
  uint8_t* ws = sm + XR_KB * XBOX;                 // XR_NS x 8 KB
  uint8_t* ob = ws + XR_NS * WBOX;                 // RED 3: XR_OB x OBUF
  uint64_t* bars = (uint64_t*)(ob + (XR_RED >= 3 ? XR_OB * OBUF : 0));
  uint64_t* xbar = bars;                           // XR_KB
  uint64_t* full = bars + XR_KB;                   // XR_NS
  uint64_t* empty = full + XR_NS;                  // XR_NS
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const uint32_t rank = cl_rank();
  const int row0 = blockIdx.x * NG, k0 = blockIdx.y * (XR_KB * 64);  // cluster dims (1, 1, C): z = token slice
  const int ntile = NG / 64, nbox = ntile * XR_KB;
  if (tid == 0) {
    for (int i = 0; i < XR_KB; i++) mbar_init(&xbar[i], 1);
    for (int i = 0; i < XR_NS; i++) { mbar_init(&full[i], 1); mbar_init(&empty[i], XR_C); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  cl_sync();
  if (warp >= 5) {  // RED 4 red warps
#if XR_RED == 4
    const int rt = tid - 160;
    for (int t = 0; t < ntile; t++) {
      asm volatile("bar.sync 2, 256;" ::: "memory");
      float4 v[MT / 8];
#pragma unroll
      for (int q = 0; q < MT / 8; q++) {
        const int idx = rt + 128 * q, r = idx / (MT / 4), c4 = idx % (MT / 4);
        v[q] = *(const float4*)(ob + r * OROW + c4 * 16);
      }
      asm volatile("bar.arrive 3, 256;" ::: "memory");
#pragma unroll
      for (int q = 0; q < MT / 8; q++) {
        const int idx = rt + 128 * q, r = idx / (MT / 4), c4 = idx % (MT / 4);
        float* o = out + (size_t)(row0 + t * 64 + r) * 128 + rank * MT + c4 * 4;
        asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(o), "f"(v[q].x), "f"(v[q].y), "f"(v[q].z),
                     "f"(v[q].w) : "memory");
      }
    }
#endif
  } else if (warp == 4) {
    if (lane == 0) {
      const uint16_t mask = (uint16_t)((1u << XR_C) - 1);
      for (int i = 0; i < nbox; i++) {
        if (i < XR_KB) { mbar_expect_tx(&xbar[i], XBOX); tma2d(xs + i * XBOX, &tmX, k0 + i * 64, rank * MT, &xbar[i]); }
        const int st = i % XR_NS, ph = (i / XR_NS) & 1;
        mbar_wait_cl(&empty[st], ph ^ 1);
        mbar_expect_tx(&full[st], WBOX);
#if XR_NOMC
        for (int q = 0; q < XR_C; q++)
          tma2d(ws + st * WBOX + q * (WROWS * 128), &tmW, k0 + (i % XR_KB) * 64, row0 + (i / XR_KB) * 64 + q * WROWS, &full[st]);
#else
        tma2d_mc(ws + st * WBOX + rank * (WROWS * 128), &tmW, k0 + (i % XR_KB) * 64,
                 row0 + (i / XR_KB) * 64 + rank * WROWS, &full[st], mask);
#endif
      }
    }
  } else {
    const int wi = warp;
    const uint32_t xb0 = smem_u32(xs), wb0 = smem_u32(ws);
    float d[NACC];
    int prev = -1;
    for (int t = 0; t < ntile; t++) {
#pragma unroll
      for (int j = 0; j < NACC; j++) d[j] = 0.f;
      asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
      for (int kb = 0; kb < XR_KB; kb++) {
        const int i = t * XR_KB + kb, st = i % XR_NS, ph = (i / XR_NS) & 1;
        if (t == 0) mbar_wait(&xbar[kb], 0);
        mbar_wait(&full[st], ph);
        const uint32_t a = wb0 + st * WBOX, b = xb0 + kb * XBOX;
#pragma unroll
        for (int k4 = 0; k4 < 4; k4++) { if (!XR_NOMMA) WG_MMA(dsw128(a + k4 * 32), dsw128(b + k4 * 32)); }
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
        if (prev >= 0 && lane < XR_C && wi == 0) mbar_arrive_cl(&empty[prev], lane);
        prev = st;
      }
      asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
      if (lane < XR_C && wi == 0) mbar_arrive_cl(&empty[prev], lane);
      prev = -1;
#pragma unroll
      for (int j = 0; j < NACC; j++) asm volatile("" : "+f"(d[j]));
#if XR_RED == 2
      const int r = row0 + t * 64 + wi * 16 + (lane >> 2);
      float* o0 = out + (size_t)r * 128 + rank * MT + 2 * (lane & 3);
      float* o1 = o0 + 8 * 128;
#pragma unroll
      for (int j = 0; j < MT / 8; j++) {
        asm volatile("red.global.add.v2.f32 [%0], {%1, %2};" :: "l"(o0 + 8 * j), "f"(d[4 * j]), "f"(d[4 * j + 1]) : "memory");
        asm volatile("red.global.add.v2.f32 [%0], {%1, %2};" :: "l"(o1 + 8 * j), "f"(d[4 * j + 2]), "f"(d[4 * j + 3]) : "memory");
      }
#elif XR_RED == 3
      {
        uint8_t* buf = ob + (t % XR_OB) * OBUF;
        if (t >= XR_OB && warp == 0) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
        asm volatile("bar.sync 1, 128;" ::: "memory");
        const int r = wi * 16 + (lane >> 2);
#pragma unroll
        for (int j = 0; j < MT / 8; j++) {
          *(float2*)(buf + r * OROW + (8 * j + 2 * (lane & 3)) * 4) = make_float2(d[4 * j], d[4 * j + 1]);
          *(float2*)(buf + (r + 8) * OROW + (8 * j + 2 * (lane & 3)) * 4) = make_float2(d[4 * j + 2], d[4 * j + 3]);
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        asm volatile("bar.sync 1, 128;" ::: "memory");
        if (warp == 0) {
#pragma unroll
          for (int h = 0; h < 2; h++) {
            const int rr = lane + 32 * h;
            float* dst = out + (size_t)(row0 + t * 64 + rr) * 128 + rank * MT;
            asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;"
                         :: "l"(dst), "r"(smem_u32(buf + rr * OROW)), "n"(MT * 4) : "memory");
          }
          asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        }
      }
#elif XR_RED == 4
      {
        if (t > 0) asm volatile("bar.sync 3, 256;" ::: "memory");
        const int r = wi * 16 + (lane >> 2);
#pragma unroll
        for (int j = 0; j < MT / 8; j++) {
          *(float2*)(ob + r * OROW + (8 * j + 2 * (lane & 3)) * 4) = make_float2(d[4 * j], d[4 * j + 1]);
          *(float2*)(ob + (r + 8) * OROW + (8 * j + 2 * (lane & 3)) * 4) = make_float2(d[4 * j + 2], d[4 * j + 3]);
        }
        asm volatile("bar.arrive 2, 256;" ::: "memory");
      }
#endif
    }
#if XR_RED == 4
    asm volatile("bar.sync 3, 256;" ::: "memory");
#endif
#if XR_RED == 3
    if (warp == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
#endif
  }
  __syncwarp();
  cl_sync();  // no CTA exits while a peer may still multicast into it or arrive on its barriers
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

// out: float [N][128] (zeroed by the caller). G n-groups, S = K / (64*XR_KB) k-splits, C token slices (cluster).
void xrm_run(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t G) {
  const int N = w.size(0), K = w.size(1);
  TORCH_CHECK(x.size(0) == 128 && x.size(1) == K);
  const int S = K / (64 * XR_KB), NG = N / G;
  TORCH_CHECK(S * 64 * XR_KB == K && NG * G == N && NG % 64 == 0);
  CUtensorMap tw, tx;
  mk(&tw, w.data_ptr(), K, N, 64, WROWS);
  mk(&tx, x.data_ptr(), K, 128, 64, MT);
  const int smem = XR_KB * XBOX + XR_NS * WBOX + (XR_RED >= 3 ? XR_OB * OBUF : 0) + 1024 + 512;
  static bool once = false;
  if (!once) {
    cudaFuncSetAttribute(xrm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    once = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(G, S, XR_C);
  cfg.blockDim = dim3(NTHR);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = 1; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = XR_C;
  cfg.attrs = at; cfg.numAttrs = 1;
  cudaError_t e = cudaLaunchKernelEx(&cfg, xrm_kernel, tw, tx, out.data_ptr<float>(), NG);
  TORCH_CHECK(e == cudaSuccess, "launch ", cudaGetErrorString(e));
}
