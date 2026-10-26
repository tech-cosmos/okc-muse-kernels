# s28 microbenchmark: hot-expert single pass (RT2, N up to 128 tokens) with the real streams: per stage a producer lane
# bulk-copies 8 KB of weights + 1 KB of scales from HBM (distinct per CTA) and an N x 64 BF16 x tile (SW128, 128 B per
# row) from a 2 MB L2-resident buffer; with MC = 2 the x tile is TMA-multicast over a 2-CTA cluster (each CTA copies
# half, both receive all). 2 consumer WGs do twm's dq + RS wgmma (MODE 3) or nothing (MODE 0: ring only).
# Prints cycles per stage (consumer view, mean over CTAs) and wall ns per stage. Usage: python mb_mc.py [G]
import torch, sys, fcntl
from torch.utils.cpp_extension import load_inline
def gen_rs(n):
    na = n // 2; dl = ", ".join(f"%{i}" for i in range(na)); al = ", ".join(f"%{na + i}" for i in range(4))
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(na))
    return (f"template <> __device__ __forceinline__ void wg_rs<{n}>(float* d, const uint32_t* a, uint64_t desc) {{\n"
            f'  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, %{na + 5}, 0;\\n"\n'
            f'    "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{dl}}}, {{{al}}}, %{na + 4}, p, 1, 1, 0;\\n}}\\n"\n'
            f'    : {outs} : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));\n}}\n')
NS = (64, 72, 96, 104, 128)
SRC = r"""
#include <cstdint>
#include <cuda_runtime.h>
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ uint32_t bmul(uint32_t a, uint32_t b) {
  uint32_t r; asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
__device__ __forceinline__ void dq(uint32_t q, uint32_t s0, uint32_t s1, uint32_t* a) {
  const uint32_t M = 0x81C081C0u;
  const uint32_t o0 = q & M, o1 = __funnelshift_l(q, q, 10) & M, o2 = __funnelshift_l(q, q, 13) & M;
  const uint32_t o3 = ((q & 0x001B001Bu) * 16648u) & M;
  a[0] = bmul(o0, s0); a[1] = bmul(o1, s1); a[2] = bmul(o2, s0); a[3] = bmul(o3, s1);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int K> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(K) : "memory"); }
__device__ __forceinline__ void fence_r(uint32_t& r) { asm volatile("" : "+r"(r)::"memory"); }
__device__ __forceinline__ uint64_t sw128_desc(const void* p) {
  return (uint64_t)((smem_u32(p) & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62); }
template <int N> __device__ __forceinline__ void wg_rs(float* d, const uint32_t* a, uint64_t desc);
@@GEN@@
__device__ __forceinline__ uint32_t cl_rank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
__device__ __forceinline__ void mbar_init(uint64_t* b, int count) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(b)), "r"(count)); }
__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, int bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(b)), "r"(bytes) : "memory"); }
__device__ __forceinline__ void mbar_arrive_cl(uint64_t* b, int rank) {   // arrive on CTA `rank`'s barrier at b's offset
  uint32_t ra; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(smem_u32(b)), "r"(rank));
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::"r"(ra) : "memory"); }
__device__ __forceinline__ void mbar_arrive(uint64_t* b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(b)) : "memory"); }
template <int MC> __device__ __forceinline__ void mbar_rel(uint64_t* b, int lane) {
  if (MC == 1 || lane == (int)cl_rank()) mbar_arrive(b); else mbar_arrive_cl(b, lane); }
__device__ __forceinline__ void mbar_wait(uint64_t* b, int parity) {
  asm volatile("{\n.reg .pred p;\nWAIT_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra WAIT_%=;\n}\n"
               ::"r"(smem_u32(b)), "r"(parity) : "memory"); }
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, int bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory"); }
__device__ __forceinline__ void bulk_g2s_mc(void* dst, const void* src, int bytes, uint64_t* bar, uint16_t mask) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1], %2, [%3], %4;"
               ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)), "h"(mask) : "memory"); }
__device__ __forceinline__ void cl_sync() { asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory"); }

constexpr int NSTG = @@NSTG@@;
template <int N> struct L { static constexpr int XB = N * 128, WB = 9216, SB = ((WB + XB + 1023) / 1024) * 1024; };
template <int N, int NA> __device__ __forceinline__ void cons_stage(const char* sb, int it, int warp, int lane, float (&acc)[2][NA], uint32_t (&afA)[2][4], uint32_t (&afB)[2][4], uint64_t* empty, uint32_t s0, uint32_t s1) {
  constexpr int XB = L<N>::XB; constexpr int MC = 1;
  {
        uint4 wv[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) wv[r] = reinterpret_cast<const uint4*>(sb + XB)[(warp * 2 + r) * 32 + lane];
        const uint64_t d0 = sw128_desc(sb);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const bool odd = j & 1;
#pragma unroll
          for (int r = 0; r < 2; ++r) {
            const uint32_t q = (j == 0 ? wv[r].x : j == 1 ? wv[r].y : j == 2 ? wv[r].z : wv[r].w) & 0x3F3F3F3Fu;
            if (odd) dq(q, s0, s1, afB[r]); else dq(q, s0, s1, afA[r]);
          }
          wg_fence();
#pragma unroll
          for (int r = 0; r < 2; ++r) wg_rs<N>((float*)acc[r], odd ? afB[r] : afA[r], d0 + j * 2);
          wg_commit();
          wg_wait<1>();
          if (j == 0 && it > 0 && lane < MC) mbar_rel<MC>(&empty[(it + NSTG - 1) % NSTG], lane);  // previous stage's wgmmas done
#pragma unroll
          for (int r = 0; r < 2; ++r)
#pragma unroll
            for (int i = 0; i < 4; ++i) { if (odd) fence_r(afA[r][i]); else fence_r(afB[r][i]); }
        }
  }
}
template <int MODE, int N, int MC>
__global__ void __launch_bounds__(384, 1) mb(const char* __restrict__ wg, long wbytes, const char* __restrict__ xg, long xbytes,
                                            unsigned long long* out, float* sinkf, int iters) {
  extern __shared__ __align__(1024) char sm_raw[];
  char* sm = (char*)(((uintptr_t)sm_raw + 1023) & ~(uintptr_t)1023);
  __shared__ uint64_t full[NSTG], empty[NSTG];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const uint32_t rank = MC > 1 ? cl_rank() : 0;
  if (tid == 0) {
    for (int s = 0; s < NSTG; ++s) { mbar_init(&full[s], 1); mbar_init(&empty[s], 8 * MC); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (MC > 1) cl_sync();
  constexpr int SB = L<N>::SB, XB = L<N>::XB, WB = L<N>::WB;
  if (warp >= 8) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n" ::: "memory");
    if (warp == 8 && lane == 0) {
      long wo = (long)blockIdx.x * (wbytes / gridDim.x / 1024) * 1024;
      long xo = (long)(blockIdx.x / MC) * 4096 % xbytes;
      for (int it = 0; it < iters; ++it) {
        const int st = it % NSTG;
        if (it >= NSTG) mbar_wait(&empty[st], ((it / NSTG) & 1) ^ 1);
        char* sb = sm + st * SB;
        mbar_expect_tx(&full[st], MODE == 1 ? WB : WB + XB);
        if (wo + WB > wbytes) wo = 0;
        bulk_g2s(sb + XB, wg + wo, WB, &full[st]); wo += WB;
        if (xo + XB > xbytes) xo = 0;
        if (MODE == 1) {}
        else if (MC == 1) bulk_g2s(sb, xg + xo, XB, &full[st]);
        else bulk_g2s_mc(sb + rank * (XB / MC), xg + xo + rank * (XB / MC), XB / MC, &full[st], (uint16_t)((1 << MC) - 1));
        xo += XB;
      }
    }
  } else {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n" ::: "memory");
    constexpr int NA = N / 2;
    float acc[2][NA];
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
      for (int i = 0; i < NA; ++i) acc[r][i] = 0.f;
    uint32_t afA[2][4], afB[2][4];
    const uint32_t s0 = 0x3F803F80u ^ (lane & 1), s1 = 0x3F803F80u;
    const unsigned long long t0 = clock64();
#pragma unroll 1
    for (int it = 0; it < iters; ++it) {
      const int st = it % NSTG;
      mbar_wait(&full[st], (it / NSTG) & 1);
      const char* sb = sm + st * SB;
      if (MODE == 5) {
        if ((it >> 4) & 1) cons_stage<72, N / 2>(sb, it, warp, lane, acc, afA, afB, empty, s0, s1); else cons_stage<128, N / 2>(sb, it, warp, lane, acc, afA, afB, empty, s0, s1);
      } else if (MODE == 3) {
        uint4 wv[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) wv[r] = reinterpret_cast<const uint4*>(sb + XB)[(warp * 2 + r) * 32 + lane];
        const uint64_t d0 = sw128_desc(sb);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const bool odd = j & 1;
#pragma unroll
          for (int r = 0; r < 2; ++r) {
            const uint32_t q = (j == 0 ? wv[r].x : j == 1 ? wv[r].y : j == 2 ? wv[r].z : wv[r].w) & 0x3F3F3F3Fu;
            if (odd) dq(q, s0, s1, afB[r]); else dq(q, s0, s1, afA[r]);
          }
          wg_fence();
#pragma unroll
          for (int r = 0; r < 2; ++r) wg_rs<N>(acc[r], odd ? afB[r] : afA[r], d0 + j * 2);
          wg_commit();
          wg_wait<1>();
          if (j == 0 && it > 0 && lane < MC) mbar_rel<MC>(&empty[(it + NSTG - 1) % NSTG], lane);  // previous stage's wgmmas done
#pragma unroll
          for (int r = 0; r < 2; ++r)
#pragma unroll
            for (int i = 0; i < 4; ++i) { if (odd) fence_r(afA[r][i]); else fence_r(afB[r][i]); }
        }
      } else {
        __syncwarp();
        if (lane < MC) mbar_rel<MC>(&empty[st], lane);
      }
    }
    wg_wait<0>();
    const unsigned long long t1 = clock64();
    float s = 0.f;
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
      for (int i = 0; i < NA; ++i) s += acc[r][i];
    sinkf[blockIdx.x * 256 + tid] = s;
    if (tid == 0) out[blockIdx.x] = t1 - t0;
  }
  __syncthreads();
  if (MC > 1) cl_sync();
}
int64_t run(int64_t mode, int64_t n, int64_t mc, torch::Tensor w, torch::Tensor x, torch::Tensor out, torch::Tensor sf, int64_t iters, int64_t grid) {
  #define GO(M, NN, C) if (mode == M && n == NN && mc == C) { auto k = mb<M, NN, C>; const int SM = NSTG * L<NN>::SB + 1024; \
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SM); \
    cudaLaunchConfig_t cfg = {}; cfg.gridDim = dim3(grid); cfg.blockDim = dim3(384); cfg.dynamicSmemBytes = SM; \
    cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = C; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1; \
    cfg.attrs = at; cfg.numAttrs = 1; \
    cudaError_t e = cudaLaunchKernelEx(&cfg, k, (const char*)w.data_ptr(), (long)w.numel(), (const char*)x.data_ptr(), (long)x.numel(), \
      (unsigned long long*)out.data_ptr(), (float*)sf.data_ptr(), (int)iters); return e == cudaSuccess ? 0 : 1000 + (int)e; }
  #define GON(NN) GO(5, NN, 1) GO(1, NN, 1) GO(0, NN, 1) GO(0, NN, 2) GO(3, NN, 1) GO(3, NN, 2)
  GON(64) GON(72) GON(96) GON(104) GON(128)
  return -1;
}
"""
import os
NSTG = int(os.environ.get("NSTG", "6"))
SRC = SRC.replace("@@GEN@@", "".join(gen_rs(n) for n in NS)).replace("@@NSTG@@", str(NSTG))
lib = load_inline(f"mb28_mcv{NSTG}", cpp_sources="int64_t run(int64_t, int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t);",
                  cuda_sources="#include <torch/extension.h>\n" + SRC, functions=["run"],
                  extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"], verbose=False)
G = int(sys.argv[1]) if len(sys.argv) > 1 else 132
w = torch.randint(0, 127, (512 << 20,), dtype=torch.uint8, device="cuda")      # 512 MB weight stream (HBM)
x = torch.full((1 << 20,), 0x3F80, dtype=torch.int16, device="cuda").view(torch.uint8)  # 2 MB x (L2)
out = torch.zeros(G, dtype=torch.int64, device="cuda"); sf = torch.zeros(G * 256, device="cuda")
IT = 3000
lk = open("/tmp/okc-gpu.lock", "w")
for n in [int(v) for v in os.environ.get('NL', '64,96,128').split(',')]:
    row = []
    for mode, mc in [tuple(int(c) for c in v) for v in os.environ.get('MM', '11,01,02,31,32').split(',')]:
        if True:
            assert lib.run(mode, n, mc, w, x, out, sf, 50, G) == 0
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            fcntl.flock(lk, fcntl.LOCK_EX)
            best = (1e9, 0)
            for rep in range(5):
                e0.record(); r = lib.run(mode, n, mc, w, x, out, sf, IT, G); e1.record(); torch.cuda.synchronize()
                assert r == 0, r
                best = min(best, (e0.elapsed_time(e1) * 1e6 / IT, out.float().mean().item() / IT))
            fcntl.flock(lk, fcntl.LOCK_UN)
            ns, cyc = best
            row.append(f"{['ringWX','ringW','','full','','mix72/128'][mode]} mc{mc} {cyc:5.0f}c {ns:5.0f}ns")
    print(f"G{G} N{n}: " + " | ".join(row), flush=True)
