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
#include <cuda_bf16.h>

#ifndef XR_KB
#define XR_KB 8      // KC / 64
#endif
#ifndef XR_NS
#define XR_NS 8
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


// ---- s42: cluster of XR_C (=8) k-splits per n-group; per 64-row tile, reduce-scatter through DSMEM: CTA rank c owns
// rows 8c..8c+7 of the tile; every CTA st.async's its FP32 rows to the owners' recv buffers ([src][8 rows][RS floats]);
// a reducer warpgroup sums the 8 slices in fixed order and stores out (XR_OUT 0: BF16 [M][N] plain stores; 1: FP32
// red.add into [M][N], for S > C).
#ifndef XR_C
#define XR_C 8
#endif
#ifndef XR_OUT
#define XR_OUT 0
#endif
#ifndef XR_DIAG
#define XR_DIAG 0   // 1: no sends / no reduce (stream+MMA only), 2: sends but no freeb wait
#endif
#ifndef XR_TEST
#define XR_TEST 1   // cluster-completed barriers: spin with test_wait (try_wait wakes late on remote arrives?)
#endif
#if XR_TEST == 2
#define XR_WAITI " mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
#elif XR_TEST
#define XR_WAITI " mbarrier.test_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\n"
#else
#define XR_WAITI " mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\n"
#endif
#ifndef XR_REL
#define XR_REL 1
#endif
#if XR_REL == 2
#define XR_ARR "mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [%0];"
#elif XR_REL
#define XR_ARR "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];"
#else
#define XR_ARR "mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [%0];"
#endif
#ifndef XR_NB
#define XR_NB 2     // staging buffers
#endif
#define RS 128
__device__ long long g_tl[256][16][8];
__device__ __forceinline__ long long gtime() { long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define CTAID (blockIdx.y * gridDim.x + blockIdx.x)
#define STAMP(t, k) if (CTAID < 256 && (t) < 16) g_tl[CTAID][t][k] = gtime()                    // recv row stride (floats), 4-float pad
#define ROWS (64 / XR_C)           // rows per owner per tile
#define SLICE (ROWS * RS * 4)     // bytes per source slice
#define SBUF (XR_C * SLICE)
#define SWZ(r, c) ((c) ^ (((r) & 3) << 3))
__device__ __forceinline__ uint32_t mapa(uint32_t a, uint32_t r) {
  uint32_t o; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(o) : "r"(a), "r"(r)); return o;
}
__device__ __forceinline__ uint32_t ctarank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
__device__ __forceinline__ void cl_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
#ifndef XR_SLEEP
#define XR_SLEEP 0
#endif
__device__ __forceinline__ void mbar_wait_cl(uint64_t* b, uint32_t ph) {
#if XR_SLEEP
  uint32_t ok;
  for (;;) {
    asm volatile("{\n .reg .pred p;\n mbarrier.test_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(ok) : "r"(smem_u32(b)), "r"(ph) : "memory");
    if (ok) break;
    __nanosleep(XR_SLEEP);
  }
#else
  asm volatile(
      "{\n .reg .pred p;\n LAB_WAITC:\n"
      XR_WAITI
      " @!p bra LAB_WAITC;\n}\n" :: "r"(smem_u32(b)), "r"(ph) : "memory");
  if (XR_TEST == 2) asm volatile("fence.acq_rel.cluster;" ::: "memory");
#endif
}

#define WG_MMA128(dA, dB, SCD) asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %66, 0;\n" "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, p, 1, 1, 0, 0;\n}\n" : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]) : "l"(dA), "l"(dB), "r"(SCD))

__global__ void __launch_bounds__(416, 1)
xr2_kernel(const __grid_constant__ CUtensorMap tmW, const __grid_constant__ CUtensorMap tmX, void* __restrict__ out,
           int NG, int N) {
  extern __shared__ __align__(1024) uint8_t smraw[];
  uint8_t* sm = (uint8_t*)(((uintptr_t)smraw + 1023) & ~(uintptr_t)1023);
  uint8_t* xs = sm;                                // XR_KB x 16 KB
  uint8_t* ws = sm + XR_KB * XBOX;                 // XR_NS x 8 KB
  float* recv = (float*)(ws + XR_NS * WBOX);       // XR_C x SLICE
  uint64_t* bars = (uint64_t*)((uint8_t*)recv + XR_NB * SBUF);
  uint64_t* xbar = bars;                           // XR_KB
  uint64_t* full = bars + XR_KB;                   // XR_NS
  uint64_t* empty = full + XR_NS;                  // XR_NS
  uint64_t* fullr = empty + XR_NS;                 // all C srcs staged tile t (XR_C arrivals)
  uint64_t* freeb = fullr + XR_NB;                     // all owners consumed my previous tile (XR_C arrivals)
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  // NG = total 64-row tiles; n-group g takes tiles [g*NG/G, (g+1)*NG/G) (uneven when G does not divide)
  const int tb = blockIdx.x * NG / gridDim.x, te = (blockIdx.x + 1) * NG / gridDim.x;
  const int row0 = tb * 64, k0 = blockIdx.y * (XR_KB * 64);
  const int ntile = te - tb, nbox = ntile * XR_KB;
  const uint32_t rank = ctarank();
  if (tid == 0) {
    for (int i = 0; i < XR_KB; i++) mbar_init(&xbar[i], 1);
    for (int i = 0; i < XR_NS; i++) { mbar_init(&full[i], 1); mbar_init(&empty[i], 128); }
    for (int i = 0; i < XR_NB; i++) { mbar_init(&fullr[i], XR_C); mbar_init(&freeb[i], XR_C); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  cl_sync();
  if (warp == 8) {
    if (lane == 0) {
      for (int i = 0; i < nbox; i++) {
        if (i < XR_KB) { mbar_expect_tx(&xbar[i], XBOX); tma2d(xs + i * XBOX, &tmX, k0 + i * 64, 0, &xbar[i]); }
        const int st = i % XR_NS, ph = (i / XR_NS) & 1;
        mbar_wait(&empty[st], ph ^ 1);
        if (i % XR_KB == 0) STAMP(i / XR_KB, 0);
        mbar_expect_tx(&full[st], WBOX);
        tma2d(ws + st * WBOX, &tmW, k0 + (i % XR_KB) * 64, row0 + (i / XR_KB) * 64, &full[st]);
      }
    }
    __syncwarp();
    cl_sync();
    return;
  }
  if (warp > 8) {
    // reducer warpgroup: thread = token
    const int tok = tid - 288;
    for (int t = 0; t < ntile && XR_DIAG != 1 && XR_DIAG != 4; t++) {
      const int bf = t % XR_NB;
      if (lane == 0) mbar_wait_cl(&fullr[bf], (t / XR_NB) & 1);
      __syncwarp();
      if (tok == 0) STAMP(t, 4);
      float s[ROWS];
#pragma unroll
      for (int r = 0; r < ROWS; r++) s[r] = 0.f;
      const uint32_t mine = smem_u32(recv) + bf * SBUF + rank * SLICE;
      float v[XR_C][ROWS];
#pragma unroll
      for (int src = 0; src < XR_C; src++) {
        const uint32_t ra = mapa(mine, src);
#pragma unroll
        for (int r = 0; r < ROWS; r++) asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v[src][r]) : "r"(ra + r * RS * 4 + 4 * SWZ(r, tok)));
      }
#pragma unroll
      for (int src = 0; src < XR_C; src++)
#pragma unroll
        for (int r = 0; r < ROWS; r++) s[r] += v[src][r];
      const int grow = row0 + t * 64 + ROWS * rank;
#pragma unroll
      for (int r8 = 0; r8 < ROWS; r8 += 8) {
#if XR_OUT == 0
      uint4 v;
      __nv_bfloat162 p0 = __floats2bfloat162_rn(s[r8], s[r8+1]), p1 = __floats2bfloat162_rn(s[r8+2], s[r8+3]);
      __nv_bfloat162 p2 = __floats2bfloat162_rn(s[r8+4], s[r8+5]), p3 = __floats2bfloat162_rn(s[r8+6], s[r8+7]);
      v.x = *(uint32_t*)&p0; v.y = *(uint32_t*)&p1; v.z = *(uint32_t*)&p2; v.w = *(uint32_t*)&p3;
      *(uint4*)((__nv_bfloat16*)out + (size_t)tok * N + grow + r8) = v;
#else
      float* o = (float*)out + (size_t)tok * N + grow + r8;
      asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(o), "f"(s[r8]), "f"(s[r8+1]), "f"(s[r8+2]), "f"(s[r8+3]) : "memory");
      asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(o + 4), "f"(s[r8+4]), "f"(s[r8+5]), "f"(s[r8+6]), "f"(s[r8+7]) : "memory");
#endif
      }
      asm volatile("bar.sync 1, 128;" ::: "memory");
      if (tok == 0) STAMP(t, 5);
      if (XR_REL == 2 && tok < XR_C) asm volatile("fence.acq_rel.cluster;" ::: "memory");
      if (tok < XR_C)
        asm volatile(XR_ARR :: "r"(mapa(smem_u32(&freeb[bf]), tok)) : "memory");
    }
    cl_sync();
    return;
  }
  {
    // s42b: WG w takes the k-chunks kb = w, w+2, ... of every tile with m64n128 (all 128 tokens): W box read once per
    // wgmma (n64 x 2 WGs was smem-bandwidth-bound); the two K-half partials are summed in the staging buffer.
    const int wg = warp >> 2, wi = warp & 3, ctid = tid & 127;
    const uint32_t xb0 = smem_u32(xs), wb0 = smem_u32(ws);
    uint32_t ldst[2];
#pragma unroll
    for (int h = 0; h < 2; h++) {
      const uint32_t rt = wi * 16 + (lane >> 2) + 8 * h, own = rt / ROWS;
      ldst[h] = smem_u32(recv) + own * SLICE + (rt % ROWS) * RS * 4;
    }
    float d[64];
    int prev = -1;
    for (int t = 0; t < ntile; t++) {
      for (int kb = wg; kb < XR_KB; kb += 2) {
        const int i = t * XR_KB + kb, st = i % XR_NS, ph = (i / XR_NS) & 1;
        if (t == 0) mbar_wait(&xbar[kb], 0);
        mbar_wait(&full[st], ph);
        const uint32_t a = wb0 + st * WBOX, b = xb0 + kb * XBOX;
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#pragma unroll
        for (int k4 = 0; k4 < 4; k4++) { WG_MMA128(dsw128(a + k4 * 32), dsw128(b + k4 * 32), (int)((kb >= 2) | (k4 != 0))); }
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
        if (prev >= 0) mbar_arrive(&empty[prev]);
        prev = st;
      }
      asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
      mbar_arrive(&empty[prev]);
      prev = -1;
      if (tid == 0) STAMP(t, 1);
      if (tid == 128) STAMP(t, 6);
      if (XR_DIAG == 1) {
        float z = 0.f;
#pragma unroll
        for (int j = 0; j < 64; j++) z += d[j];
        if (z == 1.2345f) recv[tid] = z;   // keep the MMAs alive
        continue;
      }
      const int bf = t % XR_NB;
      const uint32_t sb = bf * SBUF;
      if (t >= XR_NB && XR_DIAG != 4) { if (lane == 0) mbar_wait_cl(&freeb[bf], ((t - XR_NB) / XR_NB) & 1); __syncwarp(); }
      if (tid == 128) STAMP(t, 7);
      if (wg == 1) {
#pragma unroll
        for (int j = 0; j < 16; j++)
#pragma unroll
          for (int h = 0; h < 2; h++)
            asm volatile("st.shared.v2.f32 [%0], {%1, %2};" :: "r"(ldst[h] + sb + 4 * SWZ(lane >> 2, 8 * j + 2 * (lane & 3))), "f"(d[4 * j + 2 * h]), "f"(d[4 * j + 2 * h + 1]) : "memory");
        asm volatile("bar.arrive 2, 256;" ::: "memory");
      } else {
        asm volatile("bar.sync 2, 256;" ::: "memory");
        float u[64];
#pragma unroll
        for (int j = 0; j < 16; j++)
#pragma unroll
          for (int h = 0; h < 2; h++)
            asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];" : "=f"(u[4 * j + 2 * h]), "=f"(u[4 * j + 2 * h + 1])
                         : "r"(ldst[h] + sb + 4 * SWZ(lane >> 2, 8 * j + 2 * (lane & 3))));
#pragma unroll
        for (int j = 0; j < 16; j++)
#pragma unroll
          for (int h = 0; h < 2; h++)
            asm volatile("st.shared.v2.f32 [%0], {%1, %2};" :: "r"(ldst[h] + sb + 4 * SWZ(lane >> 2, 8 * j + 2 * (lane & 3))),
                         "f"(d[4 * j + 2 * h] + u[4 * j + 2 * h]), "f"(d[4 * j + 2 * h + 1] + u[4 * j + 2 * h + 1]));
        asm volatile("" ::: "memory");
        if (tid == 0) STAMP(t, 2);
        asm volatile("bar.sync 3, 128;" ::: "memory");
        if (tid == 0) STAMP(t, 3);
        if (XR_REL == 2 && ctid < XR_C) asm volatile("fence.acq_rel.cluster;" ::: "memory");
        if (ctid < XR_C)
          asm volatile(XR_ARR :: "r"(mapa(smem_u32(&fullr[bf]), ctid)) : "memory");
      }
    }
  }
  cl_sync();
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


// out: BF16 [128][N] (XR_OUT 0) or FP32 [128][N] zeroed (XR_OUT 1). G n-groups, S = K / (64*XR_KB) k-splits, cluster XR_C along S.
void xr_run(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t G, torch::Tensor wp) {
  const int N = w.size(0), K = w.size(1);
  TORCH_CHECK(x.size(0) == M_TOK && x.size(1) == K);
  const int S = K / (64 * XR_KB), NG = N / 64;
  TORCH_CHECK(S * 64 * XR_KB == K && N % 64 == 0 && S % XR_C == 0 && G <= NG);
  TORCH_CHECK(XR_OUT == 1 || S == XR_C);
  CUtensorMap tw, tx;
  mk(&tw, w.data_ptr(), K, N, 64, 64);
  mk(&tx, x.data_ptr(), K, M_TOK, 64, 128);
  const int smem = XR_KB * XBOX + XR_NS * WBOX + XR_NB * SBUF + 1024 + 256;
  static bool once = false;
  if (!once) {
    TORCH_CHECK(cudaFuncSetAttribute(xr2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess, "smem ", smem);
    if (XR_C > 8) cudaFuncSetAttribute(xr2_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    once = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(G, S); cfg.blockDim = dim3(416); cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = 1; at[0].val.clusterDim.y = XR_C; at[0].val.clusterDim.z = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  static int ncl = -1;
  if (ncl < 0) { cudaOccupancyMaxActiveClusters(&ncl, xr2_kernel, &cfg); printf("max active clusters %d\n", ncl); }
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, xr2_kernel, tw, tx, out.data_ptr(), NG, N) == cudaSuccess, "launch");
}

torch::Tensor xr_tl() {
  auto t = torch::empty({256, 16, 8}, torch::dtype(torch::kInt64));
  cudaMemcpyFromSymbol(t.data_ptr(), g_tl, sizeof(long long) * 256 * 16 * 8);
  return t;
}
