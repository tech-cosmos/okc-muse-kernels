
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>
#define NS_ 6
#define DBG_ 0
#define XT_ 0
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
#define MC_ 1
#define VAR_ 0
#define WG1_ 1
#define ST_ 0
__device__ __forceinline__ uint64_t dsw(uint32_t a) {
  return (uint64_t)((a & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)64 << 32) | ((uint64_t)1 << 62); }

__device__ __forceinline__ void mma128(float* d, uint64_t a, uint64_t b, int sc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, p, 1, 1, 0, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]) : "l"(a), "l"(b), "r"(sc));
}
__device__ __forceinline__ void stage128(float* d, uint64_t a0, uint64_t b0, uint64_t a1, uint64_t b1, uint64_t a2, uint64_t b2, uint64_t a3, uint64_t b3, int sc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %72, 0;\nwgmma.fence.sync.aligned;\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, p, 1, 1, 0, 0;\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %66, %67, 1, 1, 1, 0, 0;\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %68, %69, 1, 1, 1, 0, 0;\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %70, %71, 1, 1, 1, 0, 0;\nwgmma.commit_group.sync.aligned;\n}\n" : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]) : "l"(a0), "l"(b0), "l"(a1), "l"(b1), "l"(a2), "l"(b2), "l"(a3), "l"(b3), "r"(sc) : "memory");
}

__device__ __forceinline__ void mma112(float* d, uint64_t a, uint64_t b, int sc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %58, 0;\nwgmma.mma_async.sync.aligned.m64n112k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55}, %56, %57, p, 1, 1, 0, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]) : "l"(a), "l"(b), "r"(sc));
}
__device__ __forceinline__ void stage112(float* d, uint64_t a0, uint64_t b0, uint64_t a1, uint64_t b1, uint64_t a2, uint64_t b2, uint64_t a3, uint64_t b3, int sc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %64, 0;\nwgmma.fence.sync.aligned;\nwgmma.mma_async.sync.aligned.m64n112k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55}, %56, %57, p, 1, 1, 0, 0;\nwgmma.mma_async.sync.aligned.m64n112k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55}, %58, %59, 1, 1, 1, 0, 0;\nwgmma.mma_async.sync.aligned.m64n112k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55}, %60, %61, 1, 1, 1, 0, 0;\nwgmma.mma_async.sync.aligned.m64n112k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55}, %62, %63, 1, 1, 1, 0, 0;\nwgmma.commit_group.sync.aligned;\n}\n" : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]) : "l"(a0), "l"(b0), "l"(a1), "l"(b1), "l"(a2), "l"(b2), "l"(a3), "l"(b3), "r"(sc) : "memory");
}

__device__ __forceinline__ void mma104(float* d, uint64_t a, uint64_t b, int sc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %54, 0;\nwgmma.mma_async.sync.aligned.m64n104k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51}, %52, %53, p, 1, 1, 0, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]) : "l"(a), "l"(b), "r"(sc));
}
__device__ __forceinline__ void stage104(float* d, uint64_t a0, uint64_t b0, uint64_t a1, uint64_t b1, uint64_t a2, uint64_t b2, uint64_t a3, uint64_t b3, int sc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %60, 0;\nwgmma.fence.sync.aligned;\nwgmma.mma_async.sync.aligned.m64n104k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51}, %52, %53, p, 1, 1, 0, 0;\nwgmma.mma_async.sync.aligned.m64n104k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51}, %54, %55, 1, 1, 1, 0, 0;\nwgmma.mma_async.sync.aligned.m64n104k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51}, %56, %57, 1, 1, 1, 0, 0;\nwgmma.mma_async.sync.aligned.m64n104k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51}, %58, %59, 1, 1, 1, 0, 0;\nwgmma.commit_group.sync.aligned;\n}\n" : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]) : "l"(a0), "l"(b0), "l"(a1), "l"(b1), "l"(a2), "l"(b2), "l"(a3), "l"(b3), "r"(sc) : "memory");
}

__device__ __forceinline__ void mma96(float* d, uint64_t a, uint64_t b, int sc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %50, 0;\nwgmma.mma_async.sync.aligned.m64n96k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47}, %48, %49, p, 1, 1, 0, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]) : "l"(a), "l"(b), "r"(sc));
}
__device__ __forceinline__ void stage96(float* d, uint64_t a0, uint64_t b0, uint64_t a1, uint64_t b1, uint64_t a2, uint64_t b2, uint64_t a3, uint64_t b3, int sc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %56, 0;\nwgmma.fence.sync.aligned;\nwgmma.mma_async.sync.aligned.m64n96k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47}, %48, %49, p, 1, 1, 0, 0;\nwgmma.mma_async.sync.aligned.m64n96k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47}, %50, %51, 1, 1, 1, 0, 0;\nwgmma.mma_async.sync.aligned.m64n96k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47}, %52, %53, 1, 1, 1, 0, 0;\nwgmma.mma_async.sync.aligned.m64n96k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47}, %54, %55, 1, 1, 1, 0, 0;\nwgmma.commit_group.sync.aligned;\n}\n" : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]) : "l"(a0), "l"(b0), "l"(a1), "l"(b1), "l"(a2), "l"(b2), "l"(a3), "l"(b3), "r"(sc) : "memory");
}template <int NT> __device__ __forceinline__ void mma_dispatch(float* d, uint64_t a, uint64_t b, int sc) {
  if constexpr (NT == 128) mma128(d, a, b, sc);
  if constexpr (NT == 112) mma112(d, a, b, sc);
  if constexpr (NT == 104) mma104(d, a, b, sc);
  if constexpr (NT == 96) mma96(d, a, b, sc);
}
template <int NT> __device__ __forceinline__ void stage_dispatch(float* d, uint32_t a, uint32_t b, int sc) {
  if constexpr (NT == 128) stage128(d, dsw(a), dsw(b), dsw(a + 32), dsw(b + 32), dsw(a + 64), dsw(b + 64), dsw(a + 96), dsw(b + 96), sc);
  if constexpr (NT == 112) stage112(d, dsw(a), dsw(b), dsw(a + 32), dsw(b + 32), dsw(a + 64), dsw(b + 64), dsw(a + 96), dsw(b + 96), sc);
  if constexpr (NT == 104) stage104(d, dsw(a), dsw(b), dsw(a + 32), dsw(b + 32), dsw(a + 64), dsw(b + 64), dsw(a + 96), dsw(b + 96), sc);
  if constexpr (NT == 96) stage96(d, dsw(a), dsw(b), dsw(a + 32), dsw(b + 32), dsw(a + 64), dsw(b + 64), dsw(a + 96), dsw(b + 96), sc);
}

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
    for (int j = 0; j < 4; j++) mma_dispatch<NT>(d, dsw(a + 32 * j), dsw(b + 32 * j), (int)(i > 0 || j > 0));
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
  switch (nt) { case 128: run<128>(x, w, out, ntile); break; case 112: run<112>(x, w, out, ntile); break; case 104: run<104>(x, w, out, ntile); break; case 96: run<96>(x, w, out, ntile); break; default: TORCH_CHECK(false, "nt"); }
}
