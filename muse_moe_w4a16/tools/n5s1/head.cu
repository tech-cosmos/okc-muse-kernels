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
#ifndef G30_MTOK
#define G30_MTOK 128
#endif
#define G30_ND (G30_MTOK / 2)
#ifndef G30_S
#define G30_S 1  // proj s95: K split over a (1, S, 1) cluster, rank 1 sums into rank 0 via DSMEM (S <= 2)
#endif
#define G30_XBOX (G30_MTOK * 128)
#define G30_WST 6144

namespace pr32 {
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
      "{\n .reg .pred p;\n PR32_WAIT:\n"
      " mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      " @!p bra PR32_WAIT;\n}\n" :: "r"(smem_u32(b)), "r"(ph) : "memory");
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

