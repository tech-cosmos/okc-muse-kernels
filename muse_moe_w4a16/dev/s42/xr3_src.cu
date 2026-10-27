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
#define XR_NS 16
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
#define RS 8
#ifndef XR_PUSH
#define XR_PUSH 0   // 1: st.async partials into the owners' recv slots (complete_tx), reducer reads locally
#endif
#ifndef XR_XF
#define XR_XF 0
#endif
#ifndef XR_FREL
#define XR_FREL 0
#endif
#if XR_FREL
#define XR_FREEARR "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];"
#else
#define XR_FREEARR "mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [%0];"
#endif
__device__ long long g_tl[256][16][8];
__device__ __forceinline__ long long gtime() { long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define CTAID (blockIdx.y * gridDim.x + blockIdx.x)
#ifdef XR_TLON
#define STAMP(t, k) if (CTAID < 256 && (t) < 16) g_tl[CTAID][t][k] = gtime()
#else
#define STAMP(t, k)
#endif                    // recv row stride (floats), 4-float pad
#define ROWS (64 / XR_C)           // rows per owner per tile
#define SLICE (128 * 8 * 4)       // bytes per source slice: [tok 128][8 n] FP32
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

#define WG_RS(A, dB, SCD) asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %37, 0;\n" "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32, %33, %34, %35}, %36, p, 1, 1, 0;\n}\n" : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]) : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "l"(dB), "r"(SCD))

// ---- s42 xr3: x K-slice RESIDENT IN REGISTERS as wgmma RS A fragments (M = tokens: WG w = tokens 64w..64w+63,
// 32 k16 chunks x 4 regs = 128 regs), W streamed as 64-row x 64-k SW128 boxes = the smem B operand (N = 64 W rows).
// D[tok][n]: thread column group j (n = 8j..8j+7) belongs to owner CTA rank j of the cluster (C = 8 k-splits);
// partials go to a local double-buffered staging [owner][tok][8] FP32, owners pull their slices over DSMEM.
// 512 threads: WG0/1 consumers (setmaxnreg 208), WG2 warp 8 = TMA producer (40), WG3 = reducer (56).
__global__ void __launch_bounds__(512, 1)
xr3_kernel(const __grid_constant__ CUtensorMap tmW, const __nv_bfloat16* __restrict__ X, void* __restrict__ out, int NG,
           int N, int K) {
  extern __shared__ __align__(1024) uint8_t smraw[];
  uint8_t* sm = (uint8_t*)(((uintptr_t)smraw + 1023) & ~(uintptr_t)1023);
  uint8_t* ws = sm;                                  // XR_NS x 8 KB
  float* stg = (float*)(ws + XR_NS * WBOX);          // XR_NB x XR_C x 4 KB
  uint64_t* bars = (uint64_t*)((uint8_t*)stg + XR_NB * SBUF);
  uint64_t* full = bars;                             // XR_NS
  uint64_t* empty = full + XR_NS;                    // XR_NS
  uint64_t* fullr = empty + XR_NS;                   // XR_NB: all C srcs staged (C arrivals)
  uint64_t* freeb = fullr + XR_NB;                   // XR_NB: all C owners consumed (C arrivals)
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int tb = blockIdx.x * NG / gridDim.x, te = (blockIdx.x + 1) * NG / gridDim.x;
  const int row0 = tb * 64, k0 = blockIdx.y * (XR_KB * 64);
  const int ntile = te - tb, nbox = ntile * XR_KB;
  const uint32_t rank = ctarank();
  if (tid == 0) {
    for (int i = 0; i < XR_NS; i++) { mbar_init(&full[i], 1); mbar_init(&empty[i], 256); }
    for (int i = 0; i < XR_NB; i++) { mbar_init(&fullr[i], XR_PUSH ? 1 : XR_C); mbar_init(&freeb[i], XR_C); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  cl_sync();
  if (warp >= 8) {
    if (warp < 12) {
      asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
      if (warp == 8 && lane == 0) {
        for (int i = 0; i < nbox; i++) {
          const int st = i % XR_NS, ph = (i / XR_NS) & 1;
          mbar_wait(&empty[st], ph ^ 1);
          if (i % XR_KB == 0) STAMP(i / XR_KB, 0);
          mbar_expect_tx(&full[st], WBOX);
          tma2d(ws + st * WBOX, &tmW, k0 + (i % XR_KB) * 64, row0 + (i / XR_KB) * 64, &full[st]);
        }
      }
      __syncwarp();
    } else {
      asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
      const int tok = tid - 384;
      for (int t = 0; t < ntile && XR_DIAG != 1 && XR_DIAG != 4; t++) {
        const int bf = t % XR_NB;
        if (XR_PUSH && tok == 0) mbar_expect_tx(&fullr[bf], XR_C * SLICE);
        if (lane == 0) mbar_wait_cl(&fullr[bf], (t / XR_NB) & 1);
        __syncwarp();
        if (tok == 0) STAMP(t, 4);
        float s[8];
#pragma unroll
        for (int r = 0; r < 8; r++) s[r] = 0.f;
        const uint32_t mine = smem_u32(stg) + bf * SBUF + rank * SLICE + tok * 32;
#pragma unroll
        for (int s0 = 0; s0 < XR_C; s0 += 2) {
          float v[2][8];
#pragma unroll
          for (int u = 0; u < 2; u++) {
            const uint32_t ra = XR_PUSH ? smem_u32(stg) + bf * SBUF + (s0 + u) * SLICE + tok * 32 : mapa(mine, s0 + u);
            asm volatile("ld.shared::cluster.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(v[u][0]), "=f"(v[u][1]), "=f"(v[u][2]), "=f"(v[u][3]) : "r"(ra));
            asm volatile("ld.shared::cluster.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(v[u][4]), "=f"(v[u][5]), "=f"(v[u][6]), "=f"(v[u][7]) : "r"(ra + 16));
          }
#pragma unroll
          for (int u = 0; u < 2; u++)
#pragma unroll
            for (int r = 0; r < 8; r++) s[r] += v[u][r];
        }
        const int grow = row0 + t * 64 + 8 * rank;
#if XR_OUT == 0
        uint4 o;
        __nv_bfloat162 p0 = __floats2bfloat162_rn(s[0], s[1]), p1 = __floats2bfloat162_rn(s[2], s[3]);
        __nv_bfloat162 p2 = __floats2bfloat162_rn(s[4], s[5]), p3 = __floats2bfloat162_rn(s[6], s[7]);
        o.x = *(uint32_t*)&p0; o.y = *(uint32_t*)&p1; o.z = *(uint32_t*)&p2; o.w = *(uint32_t*)&p3;
        *(uint4*)((__nv_bfloat16*)out + (size_t)tok * N + grow) = o;
#else
        float* o = (float*)out + (size_t)tok * N + grow;
        asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(o), "f"(s[0]), "f"(s[1]), "f"(s[2]), "f"(s[3]) : "memory");
        asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(o + 4), "f"(s[4]), "f"(s[5]), "f"(s[6]), "f"(s[7]) : "memory");
#endif
        asm volatile("bar.sync 1, 128;" ::: "memory");
        if (tok == 0) STAMP(t, 5);
        // free: relaxed (the DSMEM loads above were consumed by the sums before bar.sync; a release here would also
        // wait for this thread's global out stores)
        if (tok < XR_C) asm volatile(XR_FREEARR :: "r"(mapa(smem_u32(&freeb[bf]), tok)) : "memory");
      }
    }
    cl_sync();
    return;
  }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 208;");
  const int wg = warp >> 2, wi = warp & 3, ctid = tid & 127, g = lane >> 2, q = lane & 3;
  // A fragments: rows (tokens) r0 = 64wg + 16wi + g, r1 = r0 + 8; chunk c: k = k0 + 16c + 2q (+8)
  uint32_t xa[XR_KB * 4][4];
#if XR_XF
  // fragment-major x (built by the caller / the producing kernel): xf[s][wg][wi][c][lane] = uint4 {a0, a1, a2, a3}
  {
    const uint4* xf = (const uint4*)X + ((size_t)((blockIdx.y * 2 + wg) * 4 + wi) * (XR_KB * 4)) * 32 + lane;
#pragma unroll
    for (int c = 0; c < XR_KB * 4; c++) {
      const uint4 v = __ldg(xf + 32 * c);
      xa[c][0] = v.x; xa[c][1] = v.y; xa[c][2] = v.z; xa[c][3] = v.w;
    }
  }
#else
  {
    const uint32_t* x0 = (const uint32_t*)(X + (size_t)(64 * wg + 16 * wi + g) * K + k0 + 2 * q);
    const uint32_t* x1 = x0 + 4 * K;   // + 8 rows (uint32 = 2 bf16)
#pragma unroll
    for (int c = 0; c < XR_KB * 4; c++) {
      xa[c][0] = __ldg(x0 + 8 * c); xa[c][1] = __ldg(x1 + 8 * c);
      xa[c][2] = __ldg(x0 + 8 * c + 4); xa[c][3] = __ldg(x1 + 8 * c + 4);
    }
  }
#endif
  // staging: owner j, token row tok = 64wg + 16wi + g (+8), n = 2q
  const uint32_t sbase = smem_u32(stg) + (64 * wg + 16 * wi + g) * 32 + 8 * q;
  const uint32_t wb0 = smem_u32(ws);
  float d[32];
  for (int t = 0; t < ntile; t++) {
#pragma unroll
    for (int kb = 0; kb < XR_KB; kb++) {
      const int i = t * XR_KB + kb, st = i % XR_NS, ph = (i / XR_NS) & 1;
      mbar_wait(&full[st], ph);
      const uint32_t b = wb0 + st * WBOX;
      asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#pragma unroll
      for (int k4 = 0; k4 < 4; k4++) {
        // xa index must be compile-time: switch over kb via the unrolled outer loop below
        WG_RS(xa[kb * 4 + k4], dsw128(b + k4 * 32), (int)((kb | k4) != 0));
      }
      asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
      asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
      mbar_arrive(&empty[st]);
    }
    if (tid == 0) STAMP(t, 1);
    if (XR_DIAG == 1) {
      float z = 0.f;
#pragma unroll
      for (int j = 0; j < 32; j++) z += d[j];
      if (z == 1.2345f) stg[tid] = z;
      continue;
    }
    const int bf = t % XR_NB;
    if (t >= XR_NB && XR_DIAG != 4) { if (lane == 0) mbar_wait_cl(&freeb[bf], ((t - XR_NB) / XR_NB) & 1); __syncwarp(); }
    const uint32_t sa = sbase + bf * SBUF;
#if XR_PUSH
    {
      // my slice in owner j's recv: [bf][src = rank][tok][8]
      const uint32_t la = smem_u32(stg) + bf * SBUF + rank * SLICE + (64 * wg + 16 * wi + g) * 32 + 8 * q;
#pragma unroll
      for (int j = 0; j < 8; j++) {
        const uint32_t ra = mapa(la, j), rb = mapa(smem_u32(&fullr[bf]), j);
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [%0], {%1, %2}, [%3];"
                     :: "r"(ra), "f"(d[4 * j]), "f"(d[4 * j + 1]), "r"(rb) : "memory");
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [%0], {%1, %2}, [%3];"
                     :: "r"(ra + 8 * 32), "f"(d[4 * j + 2]), "f"(d[4 * j + 3]), "r"(rb) : "memory");
      }
    }
    continue;
#endif
#pragma unroll
    for (int j = 0; j < 8; j++) {
      asm volatile("st.shared.v2.f32 [%0], {%1, %2};" :: "r"(sa + j * SLICE), "f"(d[4 * j]), "f"(d[4 * j + 1]) : "memory");
      asm volatile("st.shared.v2.f32 [%0], {%1, %2};" :: "r"(sa + j * SLICE + 8 * 32), "f"(d[4 * j + 2]), "f"(d[4 * j + 3]) : "memory");
    }
    asm volatile("bar.sync 2, 256;" ::: "memory");
    if (tid == 0) STAMP(t, 2);
    if (XR_REL == 2 && tid < XR_C) asm volatile("fence.acq_rel.cluster;" ::: "memory");
    if (tid < XR_C) asm volatile(XR_ARR :: "r"(mapa(smem_u32(&fullr[bf]), tid)) : "memory");
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



// x [128][K] BF16 row-major, w [N][K]; out BF16 [128][N] (XR_OUT 0, needs S == XR_C) or FP32 [128][N] zeroed (XR_OUT 1).
void xr_run(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t G, torch::Tensor wp) {
  const int N = w.size(0), K = w.size(1);
  TORCH_CHECK(x.size(0) == M_TOK && x.size(1) == K);
  const int S = K / (64 * XR_KB), NG = N / 64;
  TORCH_CHECK(S * 64 * XR_KB == K && N % 64 == 0 && S % XR_C == 0 && G <= NG);
  TORCH_CHECK(XR_OUT == 1 || S == XR_C);
  CUtensorMap tw;
  mk(&tw, w.data_ptr(), K, N, 64, 64);
  const int smem = XR_NS * WBOX + XR_NB * SBUF + 1024 + 512;
  static bool once = false;
  if (!once) {
    TORCH_CHECK(cudaFuncSetAttribute(xr3_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess, "smem ", smem);
    once = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(G, S); cfg.blockDim = dim3(512); cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = 1; at[0].val.clusterDim.y = XR_C; at[0].val.clusterDim.z = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  static int ncl = -1;
  if (ncl < 0) { cudaOccupancyMaxActiveClusters(&ncl, xr3_kernel, &cfg); printf("max active clusters %d\n", ncl); }
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, xr3_kernel, tw, (const __nv_bfloat16*)x.data_ptr(), out.data_ptr(), NG, N, K) == cudaSuccess, "launch");
}

torch::Tensor xr_tl() {
  auto t = torch::empty({256, 16, 8}, torch::dtype(torch::kInt64));
  cudaMemcpyFromSymbol(t.data_ptr(), g_tl, sizeof(long long) * 256 * 16 * 8);
  return t;
}
