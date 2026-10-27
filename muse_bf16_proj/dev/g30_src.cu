// s100 "g30" (layer s99 dev/s99_g30.cu, layer form): T=128 GEMM out[128][N] = x[128,K] . W[N,K]^T in BF16 over proj's
// lossless 12-bit BF16 weights (DEC3 dec8 units, pack30 in layer.py), W decoded straight into wgmma register
// A-fragments (swap-AB: A = W rows, B = x via TMA SW128, n = 128 tokens). CTA = 64 W rows x all K: producer warp 8
// streams the 6 KB packed W stages (G30_NW ring), warp 9 the 16 KB x boxes (G30_NS ring); 2 consumer WGs take
// alternate 64-k stages, summed through smem; epilogue staged token-major, 16 B stores. Escapes (proj s91, r-proj
// dev/s91/g30e.cu): an escape decodes to 2^-15; the thread owning (row, k) in its A fragment overwrites that BF16 half
// with v = true / scale from the header [N][ESC] float2 (col bits, v; empty slot (0, 2^-15)): exact like a plain BF16
// GEMM (prepare() checks v is exact in BF16). Candidates live in registers as keys col<<18|rr<<16|bits, sorted once (proj s92); one compare per stage. PDL: W streams before griddepcontrol.wait (weights only), x after it; dependents triggered at start.
// H100 cold-L2 unp 8192x4096 21.6 vs cuBLAS 25.7 us (layer s99, NS10/NW10).
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>

#ifndef G30_NS
#define G30_NS 10
#endif
#ifndef G30_NW
#define G30_NW 10
#endif
#ifndef G30_PRE
#define G30_PRE 1  // proj s94: consumers read + sort the escape header before griddepcontrol.wait
#endif
#ifndef G30_ESC
#define G30_ESC 8
#endif
#ifndef G30_PL  // proj n5s5 (H200 check b3ff96fb m32 -0.2..-0.5 us): sorted escape keys in a per-thread smem list (pop = 1 LDS),
#define G30_PL (G30_MTOK <= 32)  // fragment index precomputed; +17 KB smem (does not fit the T=128 rings)
#endif
#ifndef G30_MTOK
#define G30_MTOK 128
#endif
#define G30_ND (G30_MTOK / 2)
#ifndef G30_S
#define G30_S 1  // proj s95: K split over a (1, S, 1) cluster, rank 1 sums into rank 0 via DSMEM (S <= 2)
#endif
#define G30_XBOX (G30_MTOK * 128)
#define G30_WST 6144

namespace g30 {
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
      "{\n .reg .pred p;\n G30_WAIT:\n"
      " mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      " @!p bra G30_WAIT;\n}\n" :: "r"(smem_u32(b)), "r"(ph) : "memory");
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
__device__ __forceinline__ unsigned lop_aob(unsigned a, unsigned b, unsigned c) {
  unsigned r;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}
// patch the half (kh, rr, h) of j's A fragment with BF16 bits vb: a[4 j + 2 kh + rr] (m16n8k16 A layout), h = high half
__device__ __forceinline__ void esc_patch(unsigned (&a)[16], unsigned key) {
  const int kk = (key >> 18) & 63, idx = 4 * (kk >> 4) + 2 * ((kk >> 3) & 1) + ((key >> 16) & 1);
  const unsigned vb = key & 0xFFFFu, sel = (kk & 1) ? 0x5410u : 0x3254u;
#pragma unroll
  for (int u = 0; u < 16; u++)
    if (u == idx) a[u] = __byte_perm(a[u], vb, sel);
}
// key = stage << 21 | h << 20 | idx << 16 | vb (G30_PL)
__device__ __forceinline__ void esc_patch2(unsigned (&a)[16], unsigned key) {
  const int idx = (key >> 16) & 15;
  const unsigned vb = key & 0xFFFFu, sel = (key & (1u << 20)) ? 0x5410u : 0x3254u;
#pragma unroll
  for (int u = 0; u < 16; u++)
    if (u == idx) a[u] = __byte_perm(a[u], vb, sel);
}
template <int P>
__device__ __forceinline__ unsigned decw(unsigned sw, unsigned nw, unsigned sel3) {
  const unsigned sh = P == 0 ? nw : P == 1 ? nw << 4 : P == 2 ? nw << 8 : __funnelshift_r(nw, nw, 4);
  const unsigned k38 = 0x38383838u >> (gridDim.y >> 16);
  unsigned p;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(p) : "r"(sw), "r"(k38), "r"(sel3));
  return lop_aob(sh, 0x87008700u, p);
}
#define G30_D8(i) "+f"(d[i]), "+f"(d[i + 1]), "+f"(d[i + 2]), "+f"(d[i + 3]), "+f"(d[i + 4]), "+f"(d[i + 5]), "+f"(d[i + 6]), "+f"(d[i + 7])
__device__ __forceinline__ void wg_mma_rs(float* d, unsigned a0, unsigned a1, unsigned a2, unsigned a3, uint64_t db) {
#if G30_MTOK == 128
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
      "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},"
      " {%64,%65,%66,%67}, %68, 1, 1, 1, 0;\n"
      : G30_D8(0), G30_D8(8), G30_D8(16), G30_D8(24), G30_D8(32), G30_D8(40), G30_D8(48), G30_D8(56)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
#elif G30_MTOK == 32
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},"
      " {%16,%17,%18,%19}, %20, 1, 1, 1, 0;\n"
      : G30_D8(0), G30_D8(8)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
#elif G30_MTOK == 16
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7},"
      " {%8,%9,%10,%11}, %12, 1, 1, 1, 0;\n"
      : G30_D8(0)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
#else
#error G30_MTOK
#endif
}

__global__ void __launch_bounds__(320, 1)
g30_kernel(const __grid_constant__ CUtensorMap tmX, const uint8_t* __restrict__ wp, __nv_bfloat16* __restrict__ out,
           const float2* __restrict__ hdr, int N, int KS, float scale) {
  // KS = k-stages; wp = [G][KS][WST]
  extern __shared__ __align__(1024) uint8_t smraw[];
  uint8_t* sm = (uint8_t*)(((uintptr_t)smraw + 1023) & ~(uintptr_t)1023);
  uint8_t* xs = sm;                                  // NS x 16 KB
  uint8_t* ws = sm + G30_NS * G30_XBOX;              // NW x 6 KB
  uint64_t* full = (uint64_t*)(ws + G30_NW * G30_WST);  // x ring
  uint64_t* empty = full + G30_NS;
  uint64_t* wfull = empty + G30_NS;                  // W ring
  uint64_t* wempty = wfull + G30_NW;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = blockIdx.x;
  const int i0 = 2 * ((int)blockIdx.y * (KS / 2) / G30_S), i1 = 2 * (((int)blockIdx.y + 1) * (KS / 2) / G30_S);  // even: WG parity
#if G30_S > 1
  uint64_t* rbar = wempty + G30_NW;
  float* rbuf = (float*)(ws + G30_NW * G30_WST + 1024);  // [S - 1][ND][128] partials from ranks 1..S-1
#endif
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (tid == 0) {
    for (int i = 0; i < G30_NS; i++) { mbar_init(&full[i], 1); mbar_init(&empty[i], 1); }
    for (int i = 0; i < G30_NW; i++) { mbar_init(&wfull[i], 1); mbar_init(&wempty[i], 1); }
#if G30_S > 1
    mbar_init(rbar, G30_S - 1);
#endif
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
#if G30_S > 1
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
#else
  __syncthreads();
#endif
  if (warp == 8) {  // W: weights only, may run before the producer kernel is done
    if (lane == 0) {
      const uint8_t* src = wp + ((size_t)g * KS + i0) * G30_WST;
      for (int i = 0; i < i1 - i0; i++) {
        const int st = i % G30_NW, ph = (i / G30_NW) & 1;
        mbar_wait(&wempty[st], ph ^ 1);
        mbar_expect_tx(&wfull[st], G30_WST);
        bulk1d(ws + st * G30_WST, src + (size_t)i * G30_WST, G30_WST, &wfull[st]);
      }
    }
    return;
  }
#if !G30_PRE
  asm volatile("griddepcontrol.wait;" ::: "memory");  // x (and out's memory) belong to the previous kernels
#endif
  if (warp == 9) {
#if G30_PRE
    asm volatile("griddepcontrol.wait;" ::: "memory");  // x belongs to the previous kernels
#endif
    if (lane == 0) {
      for (int i = 0; i < i1 - i0; i++) {
        const int st = i % G30_NS, ph = (i / G30_NS) & 1;
        mbar_wait(&empty[st], ph ^ 1);
        mbar_expect_tx(&full[st], G30_XBOX);
        tma2d(xs + st * G30_XBOX, &tmX, (i0 + i) * 64, 0, &full[st]);
      }
    }
    return;
  }
  const int wg = warp >> 2, t = tid & 127;
  float d[G30_ND];
#pragma unroll
  for (int j = 0; j < G30_ND; j++) d[j] = 0.f;
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
        const bool mine = hv.y != 0x1p-15f && ((col >> 1) & 3) == q && ((col >> 6) & 1) == wg;
        cand[rr * G30_ESC + e] = mine ? ((unsigned)col << 18) | ((unsigned)rr << 16) | (__float_as_uint(hv.y) >> 16) : ~0u;
      }
    // sort the candidates once (ascending key = ascending col): the per-stage pop is then a register shift, not a
    // 16-way min + compare chain on the warpgroup's critical path (proj s92: unp T=128 -0.4 us, adversarial -1.5 us)
#pragma unroll
    for (int a_ = 0; a_ < 2 * G30_ESC; a_++)
#pragma unroll
      for (int b_ = 2 * G30_ESC - 1; b_ > a_; b_--) {
        const unsigned lo = min(cand[b_ - 1], cand[b_]), hi = max(cand[b_ - 1], cand[b_]);
        cand[b_ - 1] = lo; cand[b_] = hi;
      }
    G30_POP;
    while ((nxt >> 18) < (unsigned)i0 * 64u) G30_POP  // escapes of the other K split
  }
#if G30_PL
  unsigned* plist = (unsigned*)(ws + G30_NW * G30_WST + 1024 + (G30_S > 1 ? (G30_S - 1) * G30_ND * 128 * 4 : 0));  // [2 ESC + 1][256]
  {
    const int tc = tid & 255;
    int e = 0;
    while (nxt != ~0u) {
      const int kk = (nxt >> 18) & 63;
      plist[e * 256 + tc] = ((nxt >> 24) << 21) | ((unsigned)(kk & 1) << 20) |
                            ((unsigned)(4 * (kk >> 4) + 2 * ((kk >> 3) & 1) + ((nxt >> 16) & 1)) << 16) | (nxt & 0xFFFFu);
      e++;
      G30_POP
    }
    plist[e * 256 + tc] = ~0u;
    nxt = plist[tc];
  }
  int pp = 256 + (tid & 255);
#endif
#if G30_PRE
  asm volatile("griddepcontrol.wait;" ::: "memory");  // consumers: escape header (weights) read before, out written after
#endif
#if G30_PL
#define G30_FIX(a) { const unsigned kend = (unsigned)(i + 1) << 21; while (nxt < kend) { esc_patch2(a, nxt); nxt = plist[pp]; pp += 256; } }
#else
#define G30_FIX(a) { const unsigned kend = (unsigned)(i + 1) * 64u; while ((nxt >> 18) < kend) { esc_patch(a, nxt); G30_POP } }
#endif
#else
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
    const int ir = i - i0, st = ir % G30_NS, ph = (ir / G30_NS) & 1, sw = ir % G30_NW, pw = (ir / G30_NW) & 1;              \
    mbar_wait(&wfull[sw], pw);                                                                              \
    mbar_wait(&full[st], ph);                                                                               \
    const uint8_t* wst = ws + sw * G30_WST;                                                                 \
    const uint4 s01 = *(const uint4*)(wst + t * 16), s23 = *(const uint4*)(wst + 2048 + t * 16);            \
    const uint4 nw = *(const uint4*)(wst + 4096 + t * 16);                                                  \
    G30_DEC(a, 0, s01.x, s01.y, nw.x) G30_DEC(a, 1, s01.z, s01.w, nw.y) G30_DEC(a, 2, s23.x, s23.y, nw.z)   \
    G30_DEC(a, 3, s23.z, s23.w, nw.w) G30_FIX(a)                                                            \
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");                                                 \
    const uint32_t b = smem_u32(xs + st * G30_XBOX);                                                        \
    _Pragma("unroll") for (int j = 0; j < 4; j++) wg_mma_rs(d, a[4 * j], a[4 * j + 1], a[4 * j + 2], a[4 * j + 3], dsw128(b + j * 32)); \
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");                                          \
    asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");                                          \
    _Pragma("unroll") for (int j = 0; j < 16; j++) asm volatile("" : "+r"(pa[j]));                          \
    if (prev >= 0 && t == 0) { mbar_arrive(&empty[prev % G30_NS]); mbar_arrive(&wempty[prev % G30_NW]); }   \
    prev = ir;                                                                                               \
    i += 2;                                                                                                 \
  }
  int i = i0 + wg;
  while (i < i1) {
    G30_STEP(A0, A1)
    if (i >= i1) break;
    G30_STEP(A1, A0)
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
#pragma unroll
  for (int j = 0; j < 16; j++) asm volatile("" : "+r"(A0[j]), "+r"(A1[j]));
#pragma unroll
  for (int j = 0; j < G30_ND; j++) asm volatile("" : "+f"(d[j]));
  if (prev >= 0 && t == 0) { mbar_arrive(&empty[prev % G30_NS]); mbar_arrive(&wempty[prev % G30_NW]); }
  // reduce WG1 into WG0 through smem (the x ring is free once both WGs are done: barrier over the 256 consumer threads)
  float* red = (float*)xs;
  asm volatile("bar.sync 1, 256;" ::: "memory");
  if (wg == 1) {
#pragma unroll
    for (int j = 0; j < G30_ND; j++) red[j * 128 + t] = d[j];
  }
  asm volatile("bar.sync 1, 256;" ::: "memory");
  if (wg == 1) return;
#pragma unroll
  for (int j = 0; j < G30_ND; j++) d[j] += red[j * 128 + t];
#if G30_S > 1
  if (blockIdx.y > 0) {  // push the partial into rank 0's rbuf slot y - 1, one remote arrive
    uint32_t rb, rm;
    asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(rb) : "r"(smem_u32(rbuf + (blockIdx.y - 1) * G30_ND * 128)));
    asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(rm) : "r"(smem_u32(rbar)));
#pragma unroll
    for (int j = 0; j < G30_ND; j++) asm volatile("st.shared::cluster.f32 [%0], %1;" :: "r"(rb + (j * 128 + t) * 4), "f"(d[j]) : "memory");
    asm volatile("bar.sync 2, 128;" ::: "memory");
    if (t == 0) asm volatile("fence.acq_rel.cluster;\nmbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" :: "r"(rm) : "memory");
    return;
  }
  {
    asm volatile(
        "{\n .reg .pred p;\n G30_RW:\n"
        " mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], 0;\n"
        " @!p bra G30_RW;\n}\n" :: "r"(smem_u32(rbar)) : "memory");
#pragma unroll
    for (int y = 0; y < G30_S - 1; y++)
#pragma unroll
      for (int j = 0; j < G30_ND; j++) d[j] += rbuf[(y * G30_ND + j) * 128 + t];
  }
#endif
  // acc layout (m64n128): d[4*jn + 2*h + e] = row r0 + 8h, token 8*jn + tok + e
  const int r0 = (warp & 3) * 16 + (lane >> 2);
  const int tok = 2 * (lane & 3);
  // stage the tile token-major in smem, then coalesced 16 B stores (128 B per token row)
  __nv_bfloat16* stg = (__nv_bfloat16*)(xs + G30_MTOK * 256);  // [128 tok][64 rows] (+8 pad)
#pragma unroll
  for (int jn = 0; jn < G30_MTOK / 8; jn++)
#pragma unroll
    for (int h = 0; h < 2; h++)
#pragma unroll
      for (int q = 0; q < 2; q++) stg[(8 * jn + tok + q) * 72 + r0 + 8 * h] = __float2bfloat16(d[4 * jn + 2 * h + q] * scale);
  asm volatile("bar.sync 2, 128;" ::: "memory");
#pragma unroll
  for (int q = 0; q < G30_MTOK / 16; q++) {
    const int idx = q * 128 + t, tk = idx >> 3, c = idx & 7;
    *(uint4*)(out + (size_t)tk * N + g * 64 + c * 8) = *(const uint4*)(stg + tk * 72 + c * 8);
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
  TORCH_CHECK(r == CUDA_SUCCESS, "g30 tensor map encode failed ", (int)r);
}
}  // namespace g30

// out [128][N] BF16 = x [128][K] @ W^T; wp = pack30 stages, hdr = [N][G30_ESC] float2; pdl: launch as a PDL secondary
void g30_gemm(torch::Tensor x, torch::Tensor wp, torch::Tensor hdr, torch::Tensor out, double scale, int64_t pdl) {
  const int K = x.size(1), N = out.size(1);
  TORCH_CHECK(x.size(0) == G30_MTOK && x.is_contiguous() && K % 64 == 0 && N % 64 == 0 && out.size(0) == G30_MTOK);
  TORCH_CHECK((K / 64) % 2 == 0 && K / 64 >= 4 * G30_S, "g30: K stages");
  CUtensorMap tx;
  g30::mk(&tx, x.data_ptr(), K, G30_MTOK, 64, G30_MTOK);
  const int smem = G30_NS * G30_XBOX + G30_NW * G30_WST + 1024 + 512 + (G30_S > 1 ? (G30_S - 1) * G30_ND * 128 * 4 + 1024 : 0)
                   + (G30_PL ? (2 * G30_ESC + 1) * 256 * 4 : 0);
  static bool once = false;
  if (!once) { cudaFuncSetAttribute(g30::g30_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); once = true; }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(N / 64, G30_S);
  cfg.blockDim = dim3(320);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[2];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  at[1].id = cudaLaunchAttributeClusterDimension;
  at[1].val.clusterDim.x = 1; at[1].val.clusterDim.y = G30_S; at[1].val.clusterDim.z = 1;
  cfg.attrs = at; cfg.numAttrs = G30_S > 1 ? 2 : 1;
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, g30::g30_kernel, tx, (const uint8_t*)wp.data_ptr(), (__nv_bfloat16*)out.data_ptr(),
                                 (const float2*)hdr.data_ptr(), N, K / 64,
                                 (float)scale) == cudaSuccess, "g30 launch failed");
}

#ifdef G30K_ON
#if G30_PL && G30_ESC  // g30k keeps the register-shift escape pop (G30_PL is g30's only)
#undef G30_FIX
#define G30_FIX(a) { const unsigned kend = (unsigned)(i + 1) * 64u; while ((nxt >> 18) < kend) { esc_patch(a, nxt); G30_POP } }
#endif
// proj s99 "g30k": g30 (MTOK 16/32) as a stream-K GEMM on a runtime grid G (e.g. 104 CTAs, leaving SMs to the side-stream
// router). The (64-row tile, 128-k stage pair) units are split evenly over the G CTAs in tile-major order; a CTA's range
// covers parts of <= 2-3 tiles (segments). W/x rings and the escape path are g30's; ring slots run on one counter across
// segments. Two output forms:
// G30K_F32 1 (the layer's): every segment writes its FP32 partial into slab[piece][MTOK][N] (piece = the CTA's index among
//   the tile's CTAs) of a persistent zeroed buffer; (tile, slab) pairs nobody owns stay zero (static: shapes + G), so the
//   consumer sums the slabs (the pooler rmsnorm does, like the bmm path's). No reduce, deterministic. Optional ZN: WG1
//   zeroes a slice of the next g24's FP32 out after its work. H100 pooler m32 G108 ~20.0 us (g30n S2 19.2, cuBLAS 28.2).
// G30K_F32 0: BF16 out; a tile owned by one CTA is written directly, else pieces go to slab[tile][piece] + a ticket, the
//   last one sums them in piece order and resets the ticket. Measured +3.5..5.5 us over F32 (end-of-kernel chain): unused.
#ifndef G30K_PF
#define G30K_PF 1  // next segment's escape header loaded during the current one
#endif
#ifndef G30K_F32
#define G30K_F32 0  // 1: out = FP32 slabs [piece][MTOK][N] (piece = CTA index within the tile), no reduce: the consumer sums them
#endif
#ifndef G30K_MAXP
#define G30K_MAXP 4
#endif
namespace g30 {
__device__ __forceinline__ int sk_cta(long long u, long long P, int G) {  // CTA whose unit range holds unit u
  int c = (int)(u * G / P);
  while ((long long)(c + 1) * P / G <= u) c++;
  while ((long long)c * P / G > u) c--;
  return c;
}

__global__ void __launch_bounds__(320, 1)
g30k_kernel(const __grid_constant__ CUtensorMap tmX, const uint8_t* __restrict__ wp, __nv_bfloat16* __restrict__ out,
            const float2* __restrict__ hdr, int N, int KS, float scale, float* __restrict__ slab, int* __restrict__ ticket, int zn) {
  extern __shared__ __align__(1024) uint8_t smraw[];
  uint8_t* sm = (uint8_t*)(((uintptr_t)smraw + 1023) & ~(uintptr_t)1023);
  uint8_t* xs = sm;
  uint8_t* ws = sm + G30_NS * G30_XBOX;
  uint64_t* full = (uint64_t*)(ws + G30_NW * G30_WST);
  uint64_t* empty = full + G30_NS;
  uint64_t* wfull = empty + G30_NS;
  uint64_t* wempty = wfull + G30_NW;
  int* tk_s = (int*)(wempty + G30_NW);
  float* red = (float*)(ws + G30_NW * G30_WST + 1024);            // [ND][128] WG1 -> WG0
  __nv_bfloat16* stg = (__nv_bfloat16*)(red + G30_ND * 128);      // [MTOK][72]
  float* stg32 = (float*)red;  // F32 mode: [MTOK][68] (overlaps red + stg; used after WG0 has read red)
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int G = gridDim.x, KP = KS / 2;
  const long long P = (long long)(N / 64) * KP;
  const long long u0 = (long long)blockIdx.x * P / G, u1 = (long long)(blockIdx.x + 1) * P / G;
  const int gA = (int)(u0 / KP), gB = (int)((u1 - 1) / KP);
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (tid == 0) {
    for (int i = 0; i < G30_NS; i++) { mbar_init(&full[i], 1); mbar_init(&empty[i], 1); }
    for (int i = 0; i < G30_NW; i++) { mbar_init(&wfull[i], 1); mbar_init(&wempty[i], 1); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
#define G30K_SEG(gg, i0, i1)                                                   \
  const long long ta = (long long)(gg) * KP;                                   \
  const int i0 = 2 * (int)((u0 > ta ? u0 : ta) - ta);                          \
  const int i1 = 2 * (int)((u1 < ta + KP ? u1 : ta + KP) - ta);
  if (warp == 8) {
    if (lane == 0) {
      int r = 0;
      for (int g = gA; g <= gB; g++) {
        G30K_SEG(g, i0, i1)
        const uint8_t* src = wp + ((size_t)g * KS + i0) * G30_WST;
        for (int i = 0; i < i1 - i0; i++, r++) {
          const int st = r % G30_NW, ph = (r / G30_NW) & 1;
          mbar_wait(&wempty[st], ph ^ 1);
          mbar_expect_tx(&wfull[st], G30_WST);
          bulk1d(ws + st * G30_WST, src + (size_t)i * G30_WST, G30_WST, &wfull[st]);
        }
      }
    }
    return;
  }
  if (warp == 9) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (lane == 0) {
      int r = 0;
      for (int g = gA; g <= gB; g++) {
        G30K_SEG(g, i0, i1)
        for (int i = i0; i < i1; i++, r++) {
          const int st = r % G30_NS, ph = (r / G30_NS) & 1;
          mbar_wait(&empty[st], ph ^ 1);
          mbar_expect_tx(&full[st], G30_XBOX);
          tma2d(xs + st * G30_XBOX, &tmX, i * 64, 0, &full[st]);
        }
      }
    }
    return;
  }
  const int wg = warp >> 2, t = tid & 127;
  const int r0 = (warp & 3) * 16 + (lane >> 2);
  const int tok = 2 * (lane & 3);
#define G30K_OUT(gg)                                                                                                    \
  {                                                                                                                       \
    _Pragma("unroll") for (int jn = 0; jn < G30_MTOK / 8; jn++) _Pragma("unroll") for (int h = 0; h < 2; h++)            \
      _Pragma("unroll") for (int q = 0; q < 2; q++) stg[(8 * jn + tok + q) * 72 + r0 + 8 * h] = __float2bfloat16(d[4 * jn + 2 * h + q] * scale); \
    asm volatile("bar.sync 2, 128;" ::: "memory");                                                                        \
    _Pragma("unroll") for (int q = 0; q < G30_MTOK / 16; q++) {                                                           \
      const int idx = q * 128 + t, tk = idx >> 3, c = idx & 7;                                                            \
      *(uint4*)(out + (size_t)tk * N + (gg) * 64 + c * 8) = *(const uint4*)(stg + tk * 72 + c * 8);                       \
    }                                                                                                                     \
    asm volatile("bar.sync 2, 128;" ::: "memory");                                                                        \
  }
#define G30K_OUT32(gg, slot)                                                                                            \
  {                                                                                                                       \
    asm volatile("bar.sync 2, 128;" ::: "memory");  /* every WG0 thread has read red (stg32 overlaps it) */            \
    _Pragma("unroll") for (int jn = 0; jn < G30_MTOK / 8; jn++) _Pragma("unroll") for (int h = 0; h < 2; h++)            \
      _Pragma("unroll") for (int q = 0; q < 2; q++) stg32[(8 * jn + tok + q) * 68 + r0 + 8 * h] = d[4 * jn + 2 * h + q] * scale; \
    asm volatile("bar.sync 2, 128;" ::: "memory");                                                                        \
    float* o32 = (float*)out + (size_t)(slot) * G30_MTOK * N;                                                             \
    _Pragma("unroll") for (int q = 0; q < G30_MTOK / 8; q++) {                                                            \
      const int idx = q * 128 + t, tk = idx >> 4, c = idx & 15;                                                           \
      *(float4*)(o32 + (size_t)tk * N + (gg) * 64 + c * 4) = *(const float4*)(stg32 + tk * 68 + c * 4);                   \
    }                                                                                                                     \
    asm volatile("bar.sync 2, 128;" ::: "memory");                                                                        \
  }
  int rbase = 0;
#if G30_ESC
  float2 hcur[2 * G30_ESC];
#define G30K_HDR(gg)                                                                                          \
  _Pragma("unroll") for (int rr = 0; rr < 2; rr++) _Pragma("unroll") for (int e = 0; e < G30_ESC; e++)       \
    hcur[rr * G30_ESC + e] = __ldg(hdr + (size_t)((gg) * 64 + (t >> 5) * 16 + 8 * rr + ((t & 31) >> 2)) * G30_ESC + e);
  G30K_HDR(gA)
#endif
  int nsh = 0, shg[G30K_MAXP];  // shared tiles of this CTA (partials in the slab), tickets taken at the end
  for (int g = gA; g <= gB; g++) {
    G30K_SEG(g, i0, i1)
    float d[G30_ND];
#pragma unroll
    for (int j = 0; j < G30_ND; j++) d[j] = 0.f;
#if G30_ESC
    unsigned cand[2 * G30_ESC], nxt;
#if !G30K_PF
    if (g > gA) G30K_HDR(g)
#endif
    {
      const int q = t & 3;
#pragma unroll
      for (int rr = 0; rr < 2; rr++)
#pragma unroll
        for (int e = 0; e < G30_ESC; e++) {
          const float2 hv = hcur[rr * G30_ESC + e];
          const int col = __float_as_int(hv.x);
          const bool mine = hv.y != 0x1p-15f && ((col >> 1) & 3) == q && ((col >> 6) & 1) == wg;
          cand[rr * G30_ESC + e] = mine ? ((unsigned)col << 18) | ((unsigned)rr << 16) | (__float_as_uint(hv.y) >> 16) : ~0u;
        }
#if G30K_PF
      if (g < gB) G30K_HDR(g + 1)  // next segment's header: in flight during this segment
#endif
#pragma unroll
      for (int a_ = 0; a_ < 2 * G30_ESC; a_++)
#pragma unroll
        for (int b_ = 2 * G30_ESC - 1; b_ > a_; b_--) {
          const unsigned lo = min(cand[b_ - 1], cand[b_]), hi = max(cand[b_ - 1], cand[b_]);
          cand[b_ - 1] = lo; cand[b_] = hi;
        }
      G30_POP;
      while ((nxt >> 18) < (unsigned)i0 * 64u) G30_POP
    }
#endif
    if (g == gA) asm volatile("griddepcontrol.wait;" ::: "memory");  // first segment: header read before, x/out after
    unsigned A0[16], A1[16];
    int prev = -1;
#define G30K_STEP(a, pa)                                                                                    \
  {                                                                                                         \
    const int ir = rbase + i - i0, st = ir % G30_NS, ph = (ir / G30_NS) & 1, sw = ir % G30_NW, pw = (ir / G30_NW) & 1; \
    mbar_wait(&wfull[sw], pw);                                                                              \
    mbar_wait(&full[st], ph);                                                                               \
    const uint8_t* wst = ws + sw * G30_WST;                                                                 \
    const uint4 s01 = *(const uint4*)(wst + t * 16), s23 = *(const uint4*)(wst + 2048 + t * 16);            \
    const uint4 nw = *(const uint4*)(wst + 4096 + t * 16);                                                  \
    G30_DEC(a, 0, s01.x, s01.y, nw.x) G30_DEC(a, 1, s01.z, s01.w, nw.y) G30_DEC(a, 2, s23.x, s23.y, nw.z)   \
    G30_DEC(a, 3, s23.z, s23.w, nw.w) G30_FIX(a)                                                            \
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");                                                 \
    const uint32_t b = smem_u32(xs + st * G30_XBOX);                                                        \
    _Pragma("unroll") for (int j = 0; j < 4; j++) wg_mma_rs(d, a[4 * j], a[4 * j + 1], a[4 * j + 2], a[4 * j + 3], dsw128(b + j * 32)); \
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");                                          \
    asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");                                          \
    _Pragma("unroll") for (int j = 0; j < 16; j++) asm volatile("" : "+r"(pa[j]));                          \
    if (prev >= 0 && t == 0) { mbar_arrive(&empty[prev % G30_NS]); mbar_arrive(&wempty[prev % G30_NW]); }   \
    prev = ir;                                                                                              \
    i += 2;                                                                                                 \
  }
    int i = i0 + wg;
    while (i < i1) {
      G30K_STEP(A0, A1)
      if (i >= i1) break;
      G30K_STEP(A1, A0)
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
#pragma unroll
    for (int j = 0; j < 16; j++) asm volatile("" : "+r"(A0[j]), "+r"(A1[j]));
#pragma unroll
    for (int j = 0; j < G30_ND; j++) asm volatile("" : "+f"(d[j]));
    if (prev >= 0 && t == 0) { mbar_arrive(&empty[prev % G30_NS]); mbar_arrive(&wempty[prev % G30_NW]); }
    rbase += i1 - i0;
    asm volatile("bar.sync 1, 256;" ::: "memory");
    if (wg == 1) {
#pragma unroll
      for (int j = 0; j < G30_ND; j++) red[j * 128 + t] = d[j];
    }
    asm volatile("bar.sync 1, 256;" ::: "memory");
    if (wg == 1) continue;
#pragma unroll
    for (int j = 0; j < G30_ND; j++) d[j] += red[j * 128 + t];
    const int cf = sk_cta((long long)g * KP, P, G), cl = sk_cta((long long)g * KP + KP - 1, P, G);
#if G30K_F32
    G30K_OUT32(g, blockIdx.x - cf)
    continue;
#endif
    if (cl > cf) {  // shared tile: partial into slab[g][piece]; ticket at the end
      float* sb = slab + ((size_t)g * G30K_MAXP + (blockIdx.x - cf)) * G30_ND * 128;
#pragma unroll
      for (int j = 0; j < G30_ND; j++) __stcg(sb + j * 128 + t, d[j]);
      shg[nsh++] = g;
      continue;
    }
    G30K_OUT(g)
  }
#if G30K_F32
  if (wg == 1 && zn > 0) {  // ZN: zero this CTA's slice of the next g24's FP32 out (slab = that buffer in F32 mode)
    const int z0 = (int)((long long)blockIdx.x * (zn >> 2) / G) * 4, z1 = (int)((long long)(blockIdx.x + 1) * (zn >> 2) / G) * 4;
    for (int z = z0 + 4 * t; z < z1; z += 512) *(float4*)(slab + z) = make_float4(0.f, 0.f, 0.f, 0.f);
    if (blockIdx.x == G - 1)
      for (int z = (zn & ~3) + t; z < zn; z += 128) slab[z] = 0.f;
  }
#endif
  if (wg == 1 || nsh == 0) return;
  asm volatile("bar.sync 2, 128;" ::: "memory");  // all partial stores of this CTA before the release below (CUTLASS pattern)
  if (t < nsh) {
    const int g = shg[t];
    const int np = sk_cta((long long)g * KP + KP - 1, P, G) - sk_cta((long long)g * KP, P, G) + 1;
    int old;
    asm volatile("atom.add.acq_rel.gpu.global.s32 %0, [%1], 1;" : "=r"(old) : "l"(ticket + g) : "memory");
    tk_s[t] = old == np - 1;
  }
  asm volatile("bar.sync 2, 128;" ::: "memory");
  for (int s_ = 0; s_ < nsh; s_++) {
    if (!tk_s[s_]) continue;
    const int g = shg[s_];
    const int cf = sk_cta((long long)g * KP, P, G), np = sk_cta((long long)g * KP + KP - 1, P, G) - cf + 1;
    if (t == 0) ticket[g] = 0;
    const float* sb = slab + (size_t)g * G30K_MAXP * G30_ND * 128;
    float v[G30K_MAXP][G30_ND];
#pragma unroll
    for (int p = 0; p < G30K_MAXP; p++)
#pragma unroll
      for (int j = 0; j < G30_ND; j++) v[p][j] = p < np ? __ldcg(sb + (p * G30_ND + j) * 128 + t) : 0.f;
    float d[G30_ND];
#pragma unroll
    for (int j = 0; j < G30_ND; j++) {
      d[j] = v[0][j];
#pragma unroll
      for (int p = 1; p < G30K_MAXP; p++) if (p < np) d[j] += v[p][j];
    }
    G30K_OUT(g)
  }
}
}  // namespace g30

// stream-K g30 on G CTAs: out [MTOK][N] BF16; slab [N/64][G30K_MAXP][ND][128] FP32 scratch, ticket [N/64] int32 zeros.
// G30K_F32: out = FP32 slabs [>= pieces][MTOK][N] (zeros where no CTA writes); slab = a buffer zeroed after the work (ZN; may be empty)
void g30k_gemm(torch::Tensor x, torch::Tensor wp, torch::Tensor hdr, torch::Tensor out, double scale, int64_t pdl,
               torch::Tensor slab, torch::Tensor ticket, int64_t G) {
  const int K = x.size(1), N = out.size(-1);
  TORCH_CHECK(x.size(0) == G30_MTOK && x.is_contiguous() && K % 128 == 0 && N % 64 == 0 && out.size(-2) == G30_MTOK && out.is_contiguous());
  const long long KP = K / 128, P = (long long)(N / 64) * KP;
  TORCH_CHECK(G >= 1 && P / G >= 2, "g30k: grid");
  TORCH_CHECK(KP / (P / G) + 2 <= G30K_MAXP, "g30k: too many pieces per tile");
#if G30K_F32
  TORCH_CHECK(out.scalar_type() == torch::kFloat32 && out.numel() >= (KP / (P / G) + 2) * G30_MTOK * N, "g30k: FP32 slabs");
#else
  TORCH_CHECK(out.scalar_type() == torch::kBFloat16 && out.dim() == 2);
  TORCH_CHECK(slab.numel() >= (N / 64) * G30K_MAXP * G30_ND * 128 && ticket.numel() >= N / 64);
#endif
  CUtensorMap tx;
  g30::mk(&tx, x.data_ptr(), K, G30_MTOK, 64, G30_MTOK);
  const int smem = G30_NS * G30_XBOX + G30_NW * G30_WST + 1024 + 1024 + G30_ND * 128 * 4 + G30_MTOK * 72 * 2;
  static bool once = false;
  if (!once) { cudaFuncSetAttribute(g30::g30k_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); once = true; }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)G);
  cfg.blockDim = dim3(320);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  cfg.attrs = at; cfg.numAttrs = 1;
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, g30::g30k_kernel, tx, (const uint8_t*)wp.data_ptr(), (__nv_bfloat16*)out.data_ptr(),
                                 (const float2*)hdr.data_ptr(), N, K / 64, (float)scale, (float*)slab.data_ptr(),
                                 (int*)ticket.data_ptr(), G30K_F32 ? (int)slab.numel() : 0) == cudaSuccess, "g30k launch failed");
}
#endif  // G30K_ON
