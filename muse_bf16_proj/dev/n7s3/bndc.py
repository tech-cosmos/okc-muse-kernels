# n7s3 bndc: the T>=128 boundary as a CUDA kernel, one row split over a cluster of CS CTAs (N / CS columns each, 128
# threads, 8 columns per thread); the two row sums of squares (branch, then new stream) go through DSMEM: each CTA writes its
# partial into slot [rank] of every CTA of the cluster, one cluster barrier, every CTA sums the CS slots in rank order
# (same value in all CTAs). Same math as layer.py _boundary (NB = N, SPL 1, no PDL/RB/H32/PF).
import torch
from torch.utils.cpp_extension import load_inline
SRC = r"""
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
__device__ __forceinline__ float warp_sum(float v) {
  #pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ void cl_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ unsigned cl_rank() { unsigned r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
// partial -> slot[rank] of every CTA in the cluster, barrier, sum of the CS slots (fixed order)
template <int CS, int NW>
__device__ __forceinline__ float cl_allsum(float v, float* wred, float* slot, unsigned rank) {
  v = warp_sum(v);
  if ((threadIdx.x & 31) == 0) wred[threadIdx.x >> 5] = v;
  __syncthreads();
  if (threadIdx.x < CS) {
    float t = 0.f;  // CS == 1: plain block reduce
    #pragma unroll
    for (int w = 0; w < NW; ++w) t += wred[w];
    unsigned la = (unsigned)__cvta_generic_to_shared(slot + rank), ra;
#ifdef BNDC_NOBAR
    ra = la;
#else
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(la), "r"(threadIdx.x));
#endif
    asm volatile("st.shared::cluster.f32 [%0], %1;" :: "r"(ra), "f"(t) : "memory");
  }
#ifdef BNDC_NOBAR  // timing probe only (wrong sums): no cluster barrier
  __syncthreads();
#else
  cl_sync();
#endif
  float t = 0.f;
  #pragma unroll
  for (int r = 0; r < CS; ++r) t += slot[r];
  return t;
}
template <int CS, bool BGAIN, int TPB>
__global__ void __launch_bounds__(TPB) bndc_kernel(const __nv_bfloat16* __restrict__ br, const float* __restrict__ s,
    const float* __restrict__ alpha, const float* __restrict__ beta, const float* __restrict__ bgain,
    const float* __restrict__ gain, float* __restrict__ s_out, __nv_bfloat16* __restrict__ h, int N, float eps_post, float eps) {
  constexpr int NW = TPB / 32;
  __shared__ float wred[2][NW];
  __shared__ float slot[2][CS];
  unsigned rank = cl_rank();
  long row = blockIdx.x / CS;
  int c = (int)rank * (N / CS) + threadIdx.x * 8;
  const float4* sp = reinterpret_cast<const float4*>(s + row * N + c);
  float4 s0 = __ldcs(sp), s1 = __ldcs(sp + 1);
  uint4 bq = __ldcg(reinterpret_cast<const uint4*>(br + row * N + c));
  float4 a0 = __ldg(reinterpret_cast<const float4*>(alpha + c)), a1 = __ldg(reinterpret_cast<const float4*>(alpha + c) + 1);
  float4 e0 = __ldg(reinterpret_cast<const float4*>(beta + c)), e1 = __ldg(reinterpret_cast<const float4*>(beta + c) + 1);
  float4 g0 = __ldg(reinterpret_cast<const float4*>(gain + c)), g1 = __ldg(reinterpret_cast<const float4*>(gain + c) + 1);
  float b[8], sv[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
  const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&bq);
  #pragma unroll
  for (int i = 0; i < 4; ++i) { float2 f = __bfloat1622float2(b2[i]); b[2 * i] = f.x; b[2 * i + 1] = f.y; }
  if (BGAIN) {
    float4 q0 = __ldg(reinterpret_cast<const float4*>(bgain + c)), q1 = __ldg(reinterpret_cast<const float4*>(bgain + c) + 1);
    float q[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
    #pragma unroll
    for (int i = 0; i < 8; ++i) b[i] *= q[i];
  }
  float p = 0.f;
  #pragma unroll
  for (int i = 0; i < 8; ++i) p += b[i] * b[i];
  float rb = rsqrtf(cl_allsum<CS, NW>(p, wred[0], slot[0], rank) / N + eps_post);
  float av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w}, ev[8] = {e0.x, e0.y, e0.z, e0.w, e1.x, e1.y, e1.z, e1.w};
  p = 0.f;
  #pragma unroll
  for (int i = 0; i < 8; ++i) { sv[i] = av[i] * sv[i] + ev[i] * (b[i] * rb); p += sv[i] * sv[i]; }
  float4* op = reinterpret_cast<float4*>(s_out + row * N + c);
#ifndef BNDC_LATE
  op[0] = make_float4(sv[0], sv[1], sv[2], sv[3]);
  op[1] = make_float4(sv[4], sv[5], sv[6], sv[7]);
#endif
  float rs = rsqrtf(cl_allsum<CS, NW>(p, wred[1], slot[1], rank) / N + eps);
#ifdef BNDC_LATE  // the s' stores after the 2nd cluster barrier (its .release would wait for them)
  op[0] = make_float4(sv[0], sv[1], sv[2], sv[3]);
  op[1] = make_float4(sv[4], sv[5], sv[6], sv[7]);
#endif
  float gv[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
  uint4 hq; __nv_bfloat162* h2 = reinterpret_cast<__nv_bfloat162*>(&hq);
  #pragma unroll
  for (int i = 0; i < 4; ++i) h2[i] = __floats2bfloat162_rn(sv[2 * i] * rs * gv[2 * i], sv[2 * i + 1] * rs * gv[2 * i + 1]);
  *reinterpret_cast<uint4*>(h + row * N + c) = hq;
}
template <int CS, bool BG, int TPB>
static void launch(torch::Tensor br, torch::Tensor s, torch::Tensor alpha, torch::Tensor beta, torch::Tensor bgain,
                   torch::Tensor gain, torch::Tensor s_out, torch::Tensor h, double eps_post, double eps) {
  int T = s.size(0), N = s.size(1);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(T * CS); cfg.blockDim = dim3(TPB); cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = CS;
  at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1; cfg.attrs = at; cfg.numAttrs = CS > 1;
  cudaLaunchKernelEx(&cfg, bndc_kernel<CS, BG, TPB>, (const __nv_bfloat16*)br.data_ptr(), (const float*)s.data_ptr(),
      (const float*)alpha.data_ptr(), (const float*)beta.data_ptr(), (const float*)bgain.data_ptr(), (const float*)gain.data_ptr(),
      (float*)s_out.data_ptr(), (__nv_bfloat16*)h.data_ptr(), N, (float)eps_post, (float)eps);
}

// bndp: no cluster; 1024 threads own 8 columns each of a whole row; the CTA runs RPC consecutive rows, loading the
// per-column vectors (alpha, beta, gain, bgain) once, and issues row i+1's loads before row i's reductions.
__device__ __forceinline__ float blk_sum(float v, float* wred) {  // 32 warps
  v = warp_sum(v);
  if ((threadIdx.x & 31) == 0) wred[threadIdx.x >> 5] = v;
  __syncthreads();
  float t = wred[threadIdx.x & 31];
  t = warp_sum(t);
  return t;
}
__device__ __forceinline__ void ld8(const float* p, float* v) {
  float4 a = __ldg(reinterpret_cast<const float4*>(p)), b = __ldg(reinterpret_cast<const float4*>(p) + 1);
  v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}
template <bool BGAIN, int RPC>
__global__ void __launch_bounds__(1024) bndp_kernel(const __nv_bfloat16* __restrict__ br, const float* __restrict__ s,
    const float* __restrict__ alpha, const float* __restrict__ beta, const float* __restrict__ bgain,
    const float* __restrict__ gain, float* __restrict__ s_out, __nv_bfloat16* __restrict__ h, int T, float eps_post, float eps) {
  constexpr int N = 8192;
  __shared__ float wred[2 * RPC][32];
  const int c = threadIdx.x * 8;
  const long r0 = (long)blockIdx.x * RPC;
  float4 sq[RPC][2]; uint4 bq[RPC];
  #pragma unroll
  for (int i = 0; i < RPC; ++i) if (r0 + i < T) {
    const float4* sp = reinterpret_cast<const float4*>(s + (r0 + i) * N + c);
    sq[i][0] = __ldcs(sp); sq[i][1] = __ldcs(sp + 1);
    bq[i] = __ldcg(reinterpret_cast<const uint4*>(br + (r0 + i) * N + c));
  }
  float av[8], ev[8], gv[8], qv[8];
  ld8(alpha + c, av); ld8(beta + c, ev); ld8(gain + c, gv);
  if (BGAIN) ld8(bgain + c, qv);
  #pragma unroll
  for (int i = 0; i < RPC; ++i) {
    if (r0 + i >= T) break;
    const long row = r0 + i;
    float b[8], sv[8] = {sq[i][0].x, sq[i][0].y, sq[i][0].z, sq[i][0].w, sq[i][1].x, sq[i][1].y, sq[i][1].z, sq[i][1].w};
    const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&bq[i]);
    #pragma unroll
    for (int j = 0; j < 4; ++j) { float2 f = __bfloat1622float2(b2[j]); b[2 * j] = f.x; b[2 * j + 1] = f.y; }
    if (BGAIN) {
      #pragma unroll
      for (int j = 0; j < 8; ++j) b[j] *= qv[j];
    }
    float p = 0.f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) p += b[j] * b[j];
    float rb = rsqrtf(blk_sum(p, wred[2 * i]) / N + eps_post);
    p = 0.f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) { sv[j] = av[j] * sv[j] + ev[j] * (b[j] * rb); p += sv[j] * sv[j]; }
    float4* op = reinterpret_cast<float4*>(s_out + row * N + c);
    op[0] = make_float4(sv[0], sv[1], sv[2], sv[3]);
    op[1] = make_float4(sv[4], sv[5], sv[6], sv[7]);
    float rs = rsqrtf(blk_sum(p, wred[2 * i + 1]) / N + eps);
    uint4 hq; __nv_bfloat162* h2 = reinterpret_cast<__nv_bfloat162*>(&hq);
    #pragma unroll
    for (int j = 0; j < 4; ++j) h2[j] = __floats2bfloat162_rn(sv[2 * j] * rs * gv[2 * j], sv[2 * j + 1] * rs * gv[2 * j + 1]);
    *reinterpret_cast<uint4*>(h + row * N + c) = hq;
  }
}
void bndp(torch::Tensor br, torch::Tensor s, torch::Tensor alpha, torch::Tensor beta, torch::Tensor bgain,
          torch::Tensor gain, torch::Tensor s_out, torch::Tensor h, double eps_post, double eps, int64_t rpc, bool bg) {
  int T = s.size(0);
  TORCH_CHECK(s.size(1) == 8192, "N must be 8192");
  auto st = c10::cuda::getCurrentCUDAStream();
#define P_(R) if (rpc == R) { int g = (T + R - 1) / R; \
    if (bg) bndp_kernel<true, R><<<g, 1024, 0, st>>>((const __nv_bfloat16*)br.data_ptr(), (const float*)s.data_ptr(), \
      (const float*)alpha.data_ptr(), (const float*)beta.data_ptr(), (const float*)bgain.data_ptr(), (const float*)gain.data_ptr(), \
      (float*)s_out.data_ptr(), (__nv_bfloat16*)h.data_ptr(), T, (float)eps_post, (float)eps); \
    else bndp_kernel<false, R><<<g, 1024, 0, st>>>((const __nv_bfloat16*)br.data_ptr(), (const float*)s.data_ptr(), \
      (const float*)alpha.data_ptr(), (const float*)beta.data_ptr(), (const float*)bgain.data_ptr(), (const float*)gain.data_ptr(), \
      (float*)s_out.data_ptr(), (__nv_bfloat16*)h.data_ptr(), T, (float)eps_post, (float)eps); return; }
  P_(1) P_(2) P_(3) P_(4)
  TORCH_CHECK(false, "rpc");
}
void bndc(torch::Tensor br, torch::Tensor s, torch::Tensor alpha, torch::Tensor beta, torch::Tensor bgain,
          torch::Tensor gain, torch::Tensor s_out, torch::Tensor h, double eps_post, double eps, int64_t cs, bool bg) {
  TORCH_CHECK(s.size(1) == 8192, "N must be 8192");
#define L_(C) if (cs == C) { if (bg) launch<C, true, 1024 / C>(br, s, alpha, beta, bgain, gain, s_out, h, eps_post, eps); \
                 else launch<C, false, 1024 / C>(br, s, alpha, beta, bgain, gain, s_out, h, eps_post, eps); return; }
  L_(1) L_(2) L_(4) L_(8)
  TORCH_CHECK(false, "cs");
}
"""
CPP = "void bndp(torch::Tensor br, torch::Tensor s, torch::Tensor alpha, torch::Tensor beta, torch::Tensor bgain, torch::Tensor gain, torch::Tensor s_out, torch::Tensor h, double eps_post, double eps, int64_t rpc, bool bg); void bndc(torch::Tensor br, torch::Tensor s, torch::Tensor alpha, torch::Tensor beta, torch::Tensor bgain, torch::Tensor gain, torch::Tensor s_out, torch::Tensor h, double eps_post, double eps, int64_t cs, bool bg);"
_lib = None
import os
DEFS = os.environ.get("BNDC_DEFS", "").split()
def lib():
    global _lib
    if _lib is None:
        _lib = load_inline("pj_bndc" + "".join("_" + d.lower() for d in DEFS), CPP, cuda_sources="".join(f"#define {d}\n" for d in DEFS) + SRC, functions=["bndc", "bndp"], extra_cuda_cflags=["-O3", "-arch=sm_90a"],
                           build_directory=None, verbose=False)
    return _lib
