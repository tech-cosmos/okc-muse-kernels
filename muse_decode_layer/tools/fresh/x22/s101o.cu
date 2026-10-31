// s100 "g30" (layer s99 dev/s99_g30.cu, layer form): T=128 GEMM out[128][N] = x[128,K] . W[N,K]^T in BF16 over proj's
// lossless 12-bit BF16 weights (DEC3 dec8 units, pack30 in layer.py), W decoded straight into wgmma register
// A-fragments (swap-AB: A = W rows, B = x via TMA SW128, n = 128 tokens). CTA = 64 W rows x all K: producer warp 8
// streams the 6 KB packed W stages (G30_NW ring), warp 9 the 16 KB x boxes (G30_NS ring); 2 consumer WGs take
// alternate 64-k stages, summed through smem; epilogue staged token-major, 16 B stores. Escapes (proj s91, r-proj
// dev/s91/g30e.cu): an escape decodes to 2^-15; the thread owning (row, k) in its A fragment overwrites that BF16 half
// with v = true / scale from the header [N][ESC] float2 (col bits, v; empty slot (0, 2^-15)): exact like a plain BF16
// GEMM (prepare() checks v is exact in BF16). Candidates live in registers as keys col<<18|rr<<16|bits; one compare per stage. PDL: W streams before griddepcontrol.wait (weights only), x after it; dependents triggered at start.
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
#ifndef G30_ESC
#define G30_ESC 8
#endif
#ifndef O_NOX
#define O_NOX 0
#endif
#ifndef O_NOWL
#define O_NOWL 0
#endif
#ifndef O_NOFIX
#define O_NOFIX 0
#endif
#ifndef O_NOMMA
#define O_NOMMA 0
#endif
#ifndef O_NOLOOP
#define O_NOLOOP 0
#endif
#ifndef O_NOEPI
#define O_NOEPI 0
#endif
#ifndef O_NOWAIT
#define O_NOWAIT 0
#endif
#ifndef O_DSM
#define O_DSM 0
#endif
#ifndef O_E
#define O_E 0
#endif
#ifndef O_NODEC
#define O_NODEC 0
#endif
#define G30_MTOK 128
#define G30_XBOX 16384
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
__device__ __forceinline__ uint32_t mapa(uint32_t a, int r) {
  uint32_t o; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(o) : "r"(a), "r"(r)); return o;
}
__device__ __forceinline__ void mbar_arrive_remote(uint64_t* b, int r) {
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" :: "r"(mapa(smem_u32(b), r)) : "memory");
}
__device__ __forceinline__ void mbar_wait_cl(uint64_t* b, uint32_t ph) {
  asm volatile(
      "{\n .reg .pred p;\n G30_WAITC:\n"
      " mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\n"
      " @!p bra G30_WAITC;\n}\n" :: "r"(smem_u32(b)), "r"(ph) : "memory");
}
__device__ __forceinline__ float4 ldcl(uint32_t a) {
  float4 v; asm("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(a)); return v;
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
template <int NC>
__device__ __forceinline__ unsigned esc_pop(unsigned (&c)[NC]) {  // smallest key (= smallest col), removed; ~0u: none
  unsigned m = c[0];
#pragma unroll
  for (int u = 1; u < NC; u++) m = min(m, c[u]);
#pragma unroll
  for (int u = 0; u < NC; u++) c[u] = c[u] == m ? ~0u : c[u];
  return m;
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
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
      "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},"
      " {%64,%65,%66,%67}, %68, 1, 1, 1, 0;\n"
      : G30_D8(0), G30_D8(8), G30_D8(16), G30_D8(24), G30_D8(32), G30_D8(40), G30_D8(48), G30_D8(56)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
}

__global__ void __launch_bounds__(320, 1)
g30_kernel(const __grid_constant__ CUtensorMap tmX, const uint8_t* __restrict__ wp, __nv_bfloat16* __restrict__ out,
           const float2* __restrict__ hdr, int N, int KT, float scale, float* __restrict__ wsl, int* __restrict__ cnt) {
  // KT = k-stages of K; wp = [G][KT][WST]. s101 split-K: gridDim.y = S CTAs per row group, CTA s takes stages
  // [k0, k1) (uneven: S need not divide KT), writes its FP32 tile to slab s of wsl [S][128][N]; the last CTA of the
  // row group (ticket cnt[g], self-resetting) sums the S slabs in slab order and stores BF16.
  extern __shared__ __align__(1024) uint8_t smraw[];
  uint8_t* sm = (uint8_t*)(((uintptr_t)smraw + 1023) & ~(uintptr_t)1023);
  uint8_t* xs = sm;                                  // NS x 16 KB
  uint8_t* ws = sm + G30_NS * G30_XBOX;              // NW x 6 KB
  uint64_t* full = (uint64_t*)(ws + G30_NW * G30_WST);  // x ring
  uint64_t* empty = full + G30_NS;
  uint64_t* wfull = empty + G30_NS;                  // W ring
  uint64_t* wempty = wfull + G30_NW;
#if O_DSM
  __shared__ uint64_t rdy, dne;
#endif
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = blockIdx.x, s = blockIdx.y, S = gridDim.y;
  const int k0 = (int)(((long)s * KT) / S), KS = O_NOLOOP ? 0 : (int)(((long)(s + 1) * KT) / S) - k0;
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (tid == 0) {
    for (int i = 0; i < G30_NS; i++) { mbar_init(&full[i], 1); mbar_init(&empty[i], 1); }
    for (int i = 0; i < G30_NW; i++) { mbar_init(&wfull[i], 1); mbar_init(&wempty[i], 1); }
#if O_DSM
    mbar_init(&rdy, S - 1); mbar_init(&dne, S - 1);
#endif
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
#if O_DSM
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
#endif
  if (warp == 8) {  // W: weights only, may run before the producer kernel is done
    if (lane == 0) {
      const uint8_t* src = wp + ((size_t)g * KT + k0) * G30_WST;
      for (int i = 0; i < KS; i++) {
        const int st = i % G30_NW, ph = (i / G30_NW) & 1;
        mbar_wait(&wempty[st], ph ^ 1);
        if (O_NOWL) { mbar_arrive(&wfull[st]); continue; }
        mbar_expect_tx(&wfull[st], G30_WST);
        bulk1d(ws + st * G30_WST, src + (size_t)i * G30_WST, G30_WST, &wfull[st]);
      }
    }
    return;
  }
  asm volatile("griddepcontrol.wait;" ::: "memory");  // x (and out's memory) belong to the previous kernels
  if (warp == 9) {
    if (lane == 0) {
      for (int i = 0; i < KS; i++) {
        const int st = i % G30_NS, ph = (i / G30_NS) & 1;
        mbar_wait(&empty[st], ph ^ 1);
        if (O_NOX) { mbar_arrive(&full[st]); continue; }
        mbar_expect_tx(&full[st], G30_XBOX);
        tma2d(xs + st * G30_XBOX, &tmX, (k0 + i) * 64, 0, &full[st]);
      }
    }
    return;
  }
  const int wg = warp >> 2, t = tid & 127;
  float d[64];
#pragma unroll
  for (int j = 0; j < 64; j++) d[j] = 0.f;
#if G30_ESC
  unsigned cand[2 * G30_ESC], nxt;
  {
    const int q = t & 3;
#pragma unroll
    for (int rr = 0; rr < 2; rr++)
#pragma unroll
      for (int e = 0; e < G30_ESC; e++) {
        const float2 hv = __ldg(hdr + (size_t)(g * 64 + (t >> 5) * 16 + 8 * rr + ((t & 31) >> 2)) * G30_ESC + e);
        const int col = __float_as_int(hv.x);
        const bool mine = hv.y != 0x1p-15f && ((col >> 1) & 3) == q && (((col >> 6) - k0) & 1) == wg && (col >> 6) >= k0
                          && (col >> 6) < k0 + KS;
        cand[rr * G30_ESC + e] = mine ? ((unsigned)col << 18) | ((unsigned)rr << 16) | (__float_as_uint(hv.y) >> 16) : ~0u;
      }
    nxt = esc_pop(cand);
  }
#define G30_FIX(a) { const unsigned kend = (unsigned)(k0 + i + 1) * 64u; while ((nxt >> 18) < kend) { esc_patch(a, nxt); nxt = esc_pop(cand); } }
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
    const int st = i % G30_NS, ph = (i / G30_NS) & 1, sw = i % G30_NW, pw = (i / G30_NW) & 1;              \
    if (!O_NOWAIT) { mbar_wait(&wfull[sw], pw);                                                                              \
    mbar_wait(&full[st], ph); }                                                                               \
    const uint8_t* wst = ws + sw * G30_WST;                                                                 \
    const uint4 s01 = *(const uint4*)(wst + t * 16), s23 = *(const uint4*)(wst + 2048 + t * 16);            \
    const uint4 nw = *(const uint4*)(wst + 4096 + t * 16);                                                  \
    if (O_NODEC) { a[0]=s01.x;a[1]=s01.y;a[2]=s01.z;a[3]=s01.w;a[4]=s23.x;a[5]=s23.y;a[6]=s23.z;a[7]=s23.w;a[8]=nw.x;a[9]=nw.y;a[10]=nw.z;a[11]=nw.w;a[12]=s01.x^nw.x;a[13]=s01.y^nw.y;a[14]=s23.x^nw.z;a[15]=s23.y^nw.w; } else { \
    G30_DEC(a, 0, s01.x, s01.y, nw.x) G30_DEC(a, 1, s01.z, s01.w, nw.y) G30_DEC(a, 2, s23.x, s23.y, nw.z)   \
    G30_DEC(a, 3, s23.z, s23.w, nw.w) if (!O_NOFIX) G30_FIX(a) }                                                            \
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");                                                 \
    const uint32_t b = smem_u32(xs + st * G30_XBOX);                                                        \
    if (!O_NOMMA) _Pragma("unroll") for (int j = 0; j < 4; j++) wg_mma_rs(d, a[4 * j], a[4 * j + 1], a[4 * j + 2], a[4 * j + 3], dsw128(b + j * 32)); \
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");                                          \
    asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");                                          \
    _Pragma("unroll") for (int j = 0; j < 16; j++) asm volatile("" : "+r"(pa[j]));                          \
    if (prev >= 0 && t == 0) { mbar_arrive(&empty[prev % G30_NS]); mbar_arrive(&wempty[prev % G30_NW]); }   \
    prev = i;                                                                                               \
    i += 2;                                                                                                 \
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
  for (int j = 0; j < 64; j++) asm volatile("" : "+f"(d[j]));
  if (prev >= 0 && t == 0) { mbar_arrive(&empty[prev % G30_NS]); mbar_arrive(&wempty[prev % G30_NW]); }
  // reduce WG1 into WG0 through smem (the x ring is free once both WGs are done: barrier over the 256 consumer threads)
  float* red = (float*)xs;
  asm volatile("bar.sync 1, 256;" ::: "memory");
  if (wg == 1) {
#pragma unroll
    for (int j = 0; j < 64; j++) red[j * 128 + t] = d[j];
  }
  asm volatile("bar.sync 1, 256;" ::: "memory");
  if (wg == 1) return;
#pragma unroll
  for (int j = 0; j < 64; j++) d[j] += red[j * 128 + t];
  // acc layout (m64n128): d[4*jn + 2*h + e] = row r0 + 8h, token 8*jn + tok + e
  const int r0 = (warp & 3) * 16 + (lane >> 2);
  const int tok = 2 * (lane & 3);
  if (O_NOEPI) { if (d[0] == 12345.f) out[t] = __float2bfloat16(d[1]); return; }
  if (S > 1) {  // split-K: FP32 tile -> slab s, ticket; the last CTA sums the slabs (slab order) to BF16
    float* stf = (float*)(xs + 32768);  // [128 tok][64 rows] (+4 pad)
#pragma unroll
    for (int jn = 0; jn < 16; jn++)
#pragma unroll
      for (int h = 0; h < 2; h++)
#pragma unroll
        for (int q = 0; q < 2; q++) stf[(8 * jn + tok + q) * 68 + r0 + 8 * h] = d[4 * jn + 2 * h + q] * scale;
    asm volatile("bar.sync 2, 128;" ::: "memory");
#if O_DSM
    if (O_E == 1) return;
    if (t == 0) { asm volatile("fence.acq_rel.cluster;" ::: "memory"); for (int r = 0; r < S; r++) if (r != s) mbar_arrive_remote(&rdy, r); }
    mbar_wait_cl(&rdy, 0);
    if (O_E == 2) { if (t == 0) { for (int r = 0; r < S; r++) if (r != s) mbar_arrive_remote(&dne, r); mbar_wait_cl(&dne, 0); } return; }
    {
      const uint32_t sb = smem_u32(stf);
      for (int idx = (s * 1024) / S + t; idx < ((s + 1) * 1024) / S; idx += 128) {
        const int tk = idx >> 3, c = idx & 7;
        float a[8];
#pragma unroll
        for (int e = 0; e < 8; e++) a[e] = 0.f;
        float4 V[8];
#pragma unroll
        for (int u = 0; u < 4; u++) if (u < S) { const uint32_t ad = mapa(sb + (uint32_t)(tk * 68 + c * 8) * 4u, u); V[2 * u] = ldcl(ad); V[2 * u + 1] = ldcl(ad + 16); }
#pragma unroll
        for (int u = 0; u < 4; u++) if (u < S) {
          const float4 v0 = V[2 * u], v1 = V[2 * u + 1];
          a[0] += v0.x; a[1] += v0.y; a[2] += v0.z; a[3] += v0.w; a[4] += v1.x; a[5] += v1.y; a[6] += v1.z; a[7] += v1.w;
        }
        __nv_bfloat162 r[4] = {__floats2bfloat162_rn(a[0], a[1]), __floats2bfloat162_rn(a[2], a[3]),
                               __floats2bfloat162_rn(a[4], a[5]), __floats2bfloat162_rn(a[6], a[7])};
        *(uint4*)(out + (size_t)tk * N + g * 64 + c * 8) = *(const uint4*)r;
      }
    }
    asm volatile("bar.sync 2, 128;" ::: "memory");
    if (O_E == 3) return;
    if (t == 0) { for (int r = 0; r < S; r++) if (r != s) mbar_arrive_remote(&dne, r); mbar_wait_cl(&dne, 0); }
    return;
#endif
    float* o = wsl + (size_t)s * G30_MTOK * N;
#pragma unroll
    for (int q = 0; q < 16; q++) {
      const int idx = q * 128 + t, tk = idx >> 4, c = idx & 15;
      __stcg((float4*)(o + (size_t)tk * N + g * 64 + c * 4), *(const float4*)(stf + tk * 68 + c * 4));
    }
    __threadfence();
    asm volatile("bar.sync 2, 128;" ::: "memory");
    __shared__ int last;
    if (t == 0) {
      const int old = atomicAdd(cnt + g, 1);
      last = old == S - 1;
      if (last) cnt[g] = 0;  // self-reset for the next call (every CTA of this row group has arrived)
    }
    asm volatile("bar.sync 2, 128;" ::: "memory");
    if (!last) return;
    __threadfence();
    // the other slabs' tiles -> smem (all cp.async in flight at once; the x ring is free): other slab j at
    // xs + (j ? 69632 + (j - 1) * 32768 : 0), [128 tok][64 rows] FP32; own tile stays in stf
    const uint32_t sb = smem_u32(xs);
    for (int u = 0, j = 0; u < S; u++) {
      if (u == s) continue;
      const uint32_t bj = sb + (j ? 69632u + (uint32_t)(j - 1) * 32768u : 0u);
      const float* src = wsl + (size_t)u * G30_MTOK * N + g * 64;
#pragma unroll
      for (int q = 0; q < 16; q++) {
        const int idx = q * 128 + t, tk = idx >> 4, c = idx & 15;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(bj + (uint32_t)(tk * 256 + c * 16)),
                     "l"(src + (size_t)tk * N + c * 4) : "memory");
      }
      j++;
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");
    asm volatile("bar.sync 2, 128;" ::: "memory");
#pragma unroll
    for (int q = 0; q < 8; q++) {
      const int idx = q * 128 + t, tk = idx >> 3, c = idx & 7;
      float a[8];
#pragma unroll
      for (int e = 0; e < 8; e++) a[e] = 0.f;
      for (int u = 0, j = 0; u < S; u++) {  // slab order: deterministic whichever CTA is last
        const float* v = u == s ? stf + tk * 68 + c * 8
                                : (const float*)(xs + (j ? 69632 + (j - 1) * 32768 : 0)) + tk * 64 + c * 8;
        j += u != s;
        const float4 v0 = *(const float4*)v, v1 = *(const float4*)(v + 4);
        a[0] += v0.x; a[1] += v0.y; a[2] += v0.z; a[3] += v0.w; a[4] += v1.x; a[5] += v1.y; a[6] += v1.z; a[7] += v1.w;
      }
      __nv_bfloat162 r[4] = {__floats2bfloat162_rn(a[0], a[1]), __floats2bfloat162_rn(a[2], a[3]),
                             __floats2bfloat162_rn(a[4], a[5]), __floats2bfloat162_rn(a[6], a[7])};
      *(uint4*)(out + (size_t)tk * N + g * 64 + c * 8) = *(const uint4*)r;
    }
    return;
  }
  // stage the tile token-major in smem, then coalesced 16 B stores (128 B per token row)
  __nv_bfloat16* stg = (__nv_bfloat16*)(xs + 32768);  // [128 tok][64 rows] (+8 pad)
#pragma unroll
  for (int jn = 0; jn < 16; jn++)
#pragma unroll
    for (int h = 0; h < 2; h++)
#pragma unroll
      for (int q = 0; q < 2; q++) stg[(8 * jn + tok + q) * 72 + r0 + 8 * h] = __float2bfloat16(d[4 * jn + 2 * h + q] * scale);
  asm volatile("bar.sync 2, 128;" ::: "memory");
#pragma unroll
  for (int q = 0; q < 8; q++) {
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
// split > 1: ws = FP32 [>= split * 128 * N], cnt = int32 [N / 64] zeroed once (self-resetting tickets)
void g30_gemm(torch::Tensor x, torch::Tensor wp, torch::Tensor hdr, torch::Tensor out, double scale, int64_t pdl,
              torch::Tensor ws, torch::Tensor cnt, int64_t split) {
  const int K = x.size(1), N = out.size(1);
  TORCH_CHECK(x.size(0) == G30_MTOK && x.is_contiguous() && K % 64 == 0 && N % 64 == 0 && out.size(0) == G30_MTOK);
  CUtensorMap tx;
  g30::mk(&tx, x.data_ptr(), K, G30_MTOK, 64, G30_MTOK);
  const int smem = G30_NS * G30_XBOX + G30_NW * G30_WST + 1024 + 512;
  static bool once = false;
  if (!once) { cudaFuncSetAttribute(g30::g30_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); once = true; }
  cudaLaunchConfig_t cfg = {};
  TORCH_CHECK(split >= 1 && split <= 4 && split <= K / 64 && (split == 1 || (ws.numel() >= split * G30_MTOK * N && cnt.numel() >= N / 64)));
  cfg.gridDim = dim3(N / 64, split);
  cfg.blockDim = dim3(320);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[2];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  at[1].id = cudaLaunchAttributeClusterDimension;
  at[1].val.clusterDim.x = 1; at[1].val.clusterDim.y = (unsigned)split; at[1].val.clusterDim.z = 1;
  cfg.attrs = at; cfg.numAttrs = O_DSM ? 2 : 1;
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, g30::g30_kernel, tx, (const uint8_t*)wp.data_ptr(), (__nv_bfloat16*)out.data_ptr(),
                                 (const float2*)hdr.data_ptr(), N, K / 64,
                                 (float)scale, split > 1 ? ws.data_ptr<float>() : nullptr, split > 1 ? cnt.data_ptr<int>() : nullptr) == cudaSuccess, "g30 launch failed");
}
