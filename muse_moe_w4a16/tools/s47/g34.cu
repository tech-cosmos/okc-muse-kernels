// moe s47 "g34": g30 (layer g30_src.cu: 12-bit W decoded into wgmma A registers, swap-AB) at m = MT = 16/32 tokens for
// the d16/d32 pooler + unpool. CTA = 64 W rows x k-stage range [k0, k1) (uneven split S = gridDim.y), B = x TMA box
// {64 k, MT rows} SW128, wgmma m64nMTk16; out FP32 [MT][N].
// NWG consumer WGs take stages i = wg mod NWG (4 = s47 best: m32 consumer-only 12.2 vs 14.2 us at 2), summed in smem.
// S == 1 plain stores, S > 1 red.global.add (out zeroed before; S == 2 is order-independent = deterministic).
// Escapes as g30 (A-half patch), candidates filtered to the CTA's k range.
#ifndef G30_NW
#define G30_NW 20  // s47 H100 best with NWG 4 (16: +0.8 us full)
#endif
#ifndef G30_NS
#define G30_NS 20
#endif
#include "helpers.inc"
#ifndef MT
#define MT 32
#endif
#ifndef G34_PRED
#define G34_PRED 0
#endif
#define XBOX (MT * 128)
#ifndef NWG
#define NWG 4  // consumer warpgroups (alternate stages i = wg mod NWG)
#endif
#define PW (4 * NWG)  // W producer warp; PW + 1 = x producer warp
namespace g34 {
#if MT == 32
#define NACC 16
__device__ __forceinline__ void wg_mma_rs(float* d, unsigned a0, unsigned a1, unsigned a2, unsigned a3, uint64_t db) {
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %20, 1, 1, 1, 0;\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
}
#else
#define NACC 8
__device__ __forceinline__ void wg_mma_rs(float* d, unsigned a0, unsigned a1, unsigned a2, unsigned a3, uint64_t db) {
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, %12, 1, 1, 1, 0;\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
}
#endif

__global__ void __launch_bounds__(128 * NWG + 64, 1)
g34_kernel(const __grid_constant__ CUtensorMap tmX, const uint8_t* __restrict__ wp, float* __restrict__ out,
           const float2* __restrict__ hdr, int N, int KT, float scale) {
  extern __shared__ __align__(1024) uint8_t smraw[];
  uint8_t* sm = (uint8_t*)(((uintptr_t)smraw + 1023) & ~(uintptr_t)1023);
  uint8_t* xs = sm;
  uint8_t* ws = sm + G30_NS * XBOX;
  uint64_t* full = (uint64_t*)(ws + G30_NW * G30_WST);
  uint64_t* empty = full + G30_NS;
  uint64_t* wfull = empty + G30_NS;
  uint64_t* wempty = wfull + G30_NW;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = blockIdx.x, s = blockIdx.y, S = gridDim.y;
  const int k0 = (int)(((long)s * KT) / S), k1 = (int)(((long)(s + 1) * KT) / S), KS = k1 - k0;
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (tid == 0) {
    for (int i = 0; i < G30_NS; i++) { mbar_init(&full[i], 1); mbar_init(&empty[i], 1); }
    for (int i = 0; i < G30_NW; i++) { mbar_init(&wfull[i], 1); mbar_init(&wempty[i], 1); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (warp == PW) {
    if (lane == 0) {
      const uint8_t* src = wp + ((size_t)g * KT + k0) * G30_WST;
      for (int i = 0; i < KS; i++) {
        const int st = i % G30_NW, ph = (i / G30_NW) & 1;
        mbar_wait(&wempty[st], ph ^ 1);
#ifdef G34_NOW  // consumer-only probe: no W loads (decodes whatever is in smem)
        mbar_arrive(&wfull[st]);
#else
        mbar_expect_tx(&wfull[st], G30_WST);
        bulk1d(ws + st * G30_WST, src + (size_t)i * G30_WST, G30_WST, &wfull[st]);
#endif
      }
    }
    return;
  }
  if (warp == PW + 1) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (lane == 0) {
      for (int i = 0; i < KS; i++) {
        const int st = i % G30_NS, ph = (i / G30_NS) & 1;
        mbar_wait(&empty[st], ph ^ 1);
        mbar_expect_tx(&full[st], XBOX);
        tma2d(xs + st * XBOX, &tmX, (k0 + i) * 64, 0, &full[st]);
      }
    }
    return;
  }
  const int wg = warp >> 2, t = tid & 127;
  float d[NACC];
#pragma unroll
  for (int j = 0; j < NACC; j++) d[j] = 0.f;
#if G30_ESC
  unsigned cand[2 * G30_ESC], nxt;
#define G30_POP { nxt = cand[0]; _Pragma("unroll") for (int e = 0; e < 2 * G30_ESC - 1; e++) cand[e] = cand[e + 1]; cand[2 * G30_ESC - 1] = ~0u; }
  {
    const int q = t & 3;
#pragma unroll
    for (int rr = 0; rr < 2; rr++)
#pragma unroll
      for (int e = 0; e < G30_ESC; e++) {
        const float2 hv = __ldg(hdr + (size_t)(g * 64 + (t >> 5) * 16 + 8 * rr + ((t & 31) >> 2)) * G30_ESC + e);
        const int col = __float_as_int(hv.x);
        const int stg = col >> 6;  // k-stage of the escape: its owner WG is (stage - k0) & 1
        const bool mine = hv.y != 0x1p-15f && ((col >> 1) & 3) == q && stg >= k0 && stg < k1 && ((stg - k0) % NWG) == wg;
        cand[rr * G30_ESC + e] = mine ? ((unsigned)col << 18) | ((unsigned)rr << 16) | (__float_as_uint(hv.y) >> 16) : ~0u;
      }
#pragma unroll
    for (int a_ = 0; a_ < 2 * G30_ESC; a_++)
#pragma unroll
      for (int b_ = 2 * G30_ESC - 1; b_ > a_; b_--) {
        const unsigned lo = min(cand[b_ - 1], cand[b_]), hi = max(cand[b_ - 1], cand[b_]);
        cand[b_ - 1] = lo; cand[b_] = hi;
      }
    G30_POP;
  }
  asm volatile("griddepcontrol.wait;" ::: "memory");
#define G30_FIX(a) { const unsigned kend = (unsigned)(k0 + i + 1) * 64u; while ((nxt >> 18) < kend) { esc_patch(a, nxt); G30_POP } }
#else
  asm volatile("griddepcontrol.wait;" ::: "memory");
#define G30_FIX(a)
#endif
  unsigned A0[16], A1[16];
  int prev = -1;
#define G30_DEC(a, j, S0, S1, NW)                            \
    a[4 * j + 0] = decw<0>(S0, NW, 0x4140u);                 \
    a[4 * j + 1] = decw<1>(S0, NW, 0x4342u);                 \
    a[4 * j + 2] = decw<2>(S1, NW, 0x4140u);                 \
    a[4 * j + 3] = decw<3>(S1, NW, 0x4342u);
#define G30_STEP(a, pa)                                                                                     \
  {                                                                                                         \
    const int st = i % G30_NS, ph = (i / G30_NS) & 1, sw = i % G30_NW, pw = (i / G30_NW) & 1;              \
    mbar_wait(&wfull[sw], pw);                                                                              \
    const uint8_t* wst = ws + sw * G30_WST;                                                                 \
    const uint4 s01 = *(const uint4*)(wst + t * 16), s23 = *(const uint4*)(wst + 2048 + t * 16);            \
    const uint4 nw = *(const uint4*)(wst + 4096 + t * 16);                                                  \
    G30_DEC(a, 0, s01.x, s01.y, nw.x) G30_DEC(a, 1, s01.z, s01.w, nw.y) G30_DEC(a, 2, s23.x, s23.y, nw.z)   \
    G30_DEC(a, 3, s23.z, s23.w, nw.w) G30_FIX(a)                                                            \
    mbar_wait(&full[st], ph);                                                                               \
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");                                                 \
    const uint32_t b = smem_u32(xs + st * XBOX);                                                            \
    _Pragma("unroll") for (int j = 0; j < 4; j++) wg_mma_rs(d, a[4 * j], a[4 * j + 1], a[4 * j + 2], a[4 * j + 3], dsw128(b + j * 32)); \
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");                                          \
    asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");                                          \
    _Pragma("unroll") for (int j = 0; j < 16; j++) asm volatile("" : "+r"(pa[j]));                          \
    if (prev >= 0 && t == 0) { mbar_arrive(&empty[prev % G30_NS]); mbar_arrive(&wempty[prev % G30_NW]); }   \
    prev = i;                                                                                               \
    i += NWG;                                                                                               \
  }
  int i = wg;
  while (i < KS) {
    G30_STEP(A0, A1)
    if (i >= KS) break;
    G30_STEP(A1, A0)
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
#pragma unroll
  for (int j = 0; j < 16; j++) asm volatile("" : "+r"(A0[j]), "+r"(A1[j]));
#pragma unroll
  for (int j = 0; j < NACC; j++) asm volatile("" : "+f"(d[j]));
  if (prev >= 0 && t == 0) { mbar_arrive(&empty[prev % G30_NS]); mbar_arrive(&wempty[prev % G30_NW]); }
  // WG1 -> WG0 through a dedicated smem area after the barriers (the rings may still be in flight for nobody: all done)
  float* red = (float*)(wempty + G30_NW + 8);
  asm volatile("bar.sync 1, %0;" :: "n"(128 * NWG) : "memory");
  if (wg > 0) {
#pragma unroll
    for (int j = 0; j < NACC; j++) red[((wg - 1) * NACC + j) * 128 + t] = d[j];
  }
  asm volatile("bar.sync 1, %0;" :: "n"(128 * NWG) : "memory");
  if (wg > 0) return;
#pragma unroll
  for (int w_ = 0; w_ < NWG - 1; w_++)
#pragma unroll
    for (int j = 0; j < NACC; j++) d[j] += red[(w_ * NACC + j) * 128 + t];
  // acc (m64nMT): d[4 jn + 2 h + e] = row r0 + 8 h, token 8 jn + tok + e
  const int r0 = (warp & 3) * 16 + (lane >> 2), tok = 2 * (lane & 3);
#pragma unroll
  for (int jn = 0; jn < MT / 8; jn++)
#pragma unroll
    for (int h = 0; h < 2; h++)
#pragma unroll
      for (int e = 0; e < 2; e++) {
        float* p = out + (size_t)(8 * jn + tok + e) * N + g * 64 + r0 + 8 * h;
        const float v = d[4 * jn + 2 * h + e] * scale;
        if (S == 1) *p = v; else atomicAdd(p, v);
      }
}

__global__ void pred_kernel(long ns) {  // PDL predecessor: triggers at start, spins ns
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  unsigned long long t0, t1;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1)); } while ((long)(t1 - t0) < ns);
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
  TORCH_CHECK(r == CUDA_SUCCESS, "g34 tensor map encode failed ", (int)r);
}
}  // namespace g34

void g34_pred(int64_t ns) { g34::pred_kernel<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>(ns); }

void g34_gemm(torch::Tensor x, torch::Tensor wp, torch::Tensor hdr, torch::Tensor out, double scale, int64_t S, int64_t pdl) {
  const int K = x.size(1), N = out.size(1);
  TORCH_CHECK(x.size(0) == MT && x.is_contiguous() && K % 64 == 0 && N % 64 == 0 && out.size(0) == MT);
  CUtensorMap tx;
  g34::mk(&tx, x.data_ptr(), K, MT, 64, MT);
  const int smem = G30_NS * XBOX + G30_NW * G30_WST + 1024 + 512 + (NWG - 1) * NACC * 128 * 4;
  static bool once = false;
  if (!once) { cudaFuncSetAttribute(g34::g34_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); once = true; }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(N / 64, S);
  cfg.blockDim = dim3(128 * NWG + 64);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  cfg.attrs = at; cfg.numAttrs = 1;
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, g34::g34_kernel, tx, (const uint8_t*)wp.data_ptr(), (float*)out.data_ptr(),
                                 (const float2*)hdr.data_ptr(), N, K / 64, (float)scale) == cudaSuccess, "g34 launch failed");
}
