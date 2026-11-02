# o45 item (3): CROSS-SM DEQUANT feasibility = DSMEM bandwidth. 2-CTA clusters (1 CTA/SM), CTA 0 = producer writes
# BF16 "panels" into a ring in CTA 1's smem, CTA 1 = consumer (waits full, optionally LDS-reads the stage, frees it).
# MODE 0: producer st.shared::cluster.v4 from registers (256 thr), per-warp syncwarp + 1 remote release arrive.
# MODE 1: same, every thread arrives (count 256).
# MODE 2: producer cp.async.bulk smem -> peer smem (TMA engine, complete_tx on the peer's full barrier), 1 thread.
# MODE 3: MODE 2 with 4 producer threads each copying a quarter of the stage.
# RD 1: consumer reads every byte with LDS.128 before freeing (RD 0: frees at once). Reports B/clk per consumer SM.
import torch, sys, fcntl
from torch.utils.cpp_extension import load_inline
SRC = r"""
#include <cstdint>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
__device__ __forceinline__ uint32_t su32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ uint32_t mapa(uint32_t a, uint32_t r) { uint32_t o; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(o) : "r"(a), "r"(r)); return o; }
__device__ __forceinline__ void minit(uint32_t b, int c) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(b), "r"(c)); }
__device__ __forceinline__ void rarrive(uint32_t rb) { asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::"r"(rb) : "memory"); }
__device__ __forceinline__ void larrive_tx(uint32_t b, uint32_t tx) { asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(b), "r"(tx) : "memory"); }
__device__ __forceinline__ void mwait(uint32_t b, uint32_t ph) {
  asm volatile("{\n.reg .pred p;\nW: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}\n" ::"r"(b), "r"(ph) : "memory"); }
__device__ __forceinline__ void st_cl(uint32_t a, uint4 v) { asm volatile("st.shared::cluster.v4.b32 [%0], {%1, %2, %3, %4};" ::"r"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory"); }
__device__ __forceinline__ void bulk_cl(uint32_t dst, uint32_t src, uint32_t n, uint32_t mb) {
  asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst), "r"(src), "r"(n), "r"(mb) : "memory"); }
template <int MODE, int RD, int BUF, int NS>
__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(256, 1) mbd(unsigned long long* out, uint32_t* sink, int iters) {
  extern __shared__ __align__(1024) char sm[];
  __shared__ __align__(8) uint64_t full[NS], empty[NS];
  cg::cluster_group cl = cg::this_cluster();
  const int rank = cl.block_rank(), tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  char* ring = sm;                       // consumer: NS x BUF landing ring; producer: BUF source panel
  if (tid == 0) for (int s = 0; s < NS; ++s) {
    minit(su32(&full[s]), MODE == 0 ? 8 : MODE == 1 ? 256 : 1);
    minit(su32(&empty[s]), 8);
  }
  for (int i = tid; i < NS * BUF / 4; i += 256) reinterpret_cast<uint32_t*>(sm)[i] = i * 2654435761u;
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  __syncthreads();
  if (MODE >= 2 && rank == 1 && tid == 0) for (int s = 0; s < NS; ++s) larrive_tx(su32(&full[s]), BUF);  // arm the first fills
  cl.sync();
  uint32_t xs = 0;
  const unsigned long long t0 = clock64();
  if (rank == 0) {  // producer
    const uint32_t rring = mapa(su32(ring), 1);
#pragma unroll 1
    for (int it = 0; it < iters; ++it) {
      const int s = it % NS; const uint32_t ph = (it / NS) & 1;
      if (it >= NS) mwait(su32(&empty[s]), ph ^ 1);
      if (MODE <= 1) {
        const uint32_t base = rring + s * BUF;
#pragma unroll 4
        for (int i = tid; i < BUF / 16; i += 256) st_cl(base + i * 16, make_uint4(it, i, it ^ i, tid));
        if (MODE == 0) { __syncwarp(); if (lane == 0) rarrive(mapa(su32(&full[s]), 1)); }
        else rarrive(mapa(su32(&full[s]), 1));
      } else {
        constexpr int NP = MODE == 2 ? 1 : 4;
        if (tid < NP) bulk_cl(rring + s * BUF + tid * (BUF / NP), su32(ring) + tid * (BUF / NP), BUF / NP, mapa(su32(&full[s]), 1));
      }
    }
  } else {  // consumer
#pragma unroll 1
    for (int it = 0; it < iters; ++it) {
      const int s = it % NS; const uint32_t ph = (it / NS) & 1;
      mwait(su32(&full[s]), ph);
      if (RD) {
#pragma unroll 4
        for (int i = tid; i < BUF / 16; i += 256) { const uint4 v = reinterpret_cast<const uint4*>(ring + s * BUF)[i]; xs ^= v.x ^ v.y ^ v.z ^ v.w; }
      }
      __syncwarp();
      if (MODE >= 2 && tid == 0 && it + NS < iters) larrive_tx(su32(&full[s]), BUF);  // re-arm for the next fill
      if (lane == 0) rarrive(mapa(su32(&empty[s]), 0));
    }
  }
  const unsigned long long t1 = clock64();
  sink[blockIdx.x * 256 + tid] = xs;
  if (tid == 0) out[blockIdx.x] = t1 - t0;
  cl.sync();
}
template <int MODE, int RD, int BUF, int NS> __global__ void arm(int) {}
int64_t run(int64_t mode, int64_t rd, int64_t buf, torch::Tensor out, torch::Tensor sink, int64_t iters, int64_t grid) {
  const int SM = 200 * 1024;
#define GO(M, R, B, N) if (mode == M && rd == R && buf == B) { auto k = mbd<M, R, B, N>; \
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SM); \
    k<<<grid, 256, SM>>>((unsigned long long*)out.data_ptr(), (uint32_t*)sink.data_ptr(), (int)iters); return (int64_t)cudaGetLastError(); }
#define GM(M) GO(M, 0, 8192, 8) GO(M, 1, 8192, 8) GO(M, 0, 16384, 8) GO(M, 1, 16384, 8) GO(M, 0, 32768, 4) GO(M, 1, 32768, 4) GO(M, 0, 49152, 4) GO(M, 1, 49152, 4)
  GM(0) GM(1) GM(2) GM(3)
  return -1;
}
"""
lib = load_inline("o45_dsmem5", cpp_sources="int64_t run(int64_t, int64_t, int64_t, torch::Tensor, torch::Tensor, int64_t, int64_t);",
                  cuda_sources="#include <torch/extension.h>\n" + SRC, functions=["run"],
                  extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"], verbose=False)
out = torch.zeros(132, dtype=torch.int64, device="cuda"); sink = torch.zeros(132 * 256, dtype=torch.int32, device="cuda")
lk = open("/tmp/okc-gpu.lock", "w"); IT = 4000
names = {0: "st.cluster.v4, warp arrive", 1: "st.cluster.v4, thread arrive", 2: "bulk copy x1", 3: "bulk copy x4"}
BUFS = [int(b) for b in sys.argv[2].split(',')] if len(sys.argv) > 2 else (8192, 16384)
for G in ([int(sys.argv[1])] if len(sys.argv) > 1 else (2, 132)):
    for mode in (0, 1, 2, 3):
        for buf in BUFS:
            row = []
            for rd in (0, 1):
                assert lib.run(mode, rd, buf, out, sink, 16, G) == 0
                res = []
                for rep in range(3):
                    fcntl.flock(lk, fcntl.LOCK_EX); lib.run(mode, rd, buf, out, sink, IT, G); torch.cuda.synchronize(); fcntl.flock(lk, fcntl.LOCK_UN)
                    res.append(IT * buf / out[1:G:2].float().mean().item())
                row.append(f"rd{rd} {sorted(res)[1]:5.1f}")
            print(f"grid {G:3d} {names[mode]:28s} stage {buf // 1024:2d} KB: B/clk/consumer SM " + " | ".join(row), flush=True)
