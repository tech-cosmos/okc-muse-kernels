import torch, fcntl, statistics
from torch.utils.cpp_extension import load_inline
src = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
// grid G*CL CTAs, cluster CL along K. Each CTA: NT tiles of TF floats (64 rows x M tokens); per tile: write own partial
// to smem buf[t&1], cluster.sync, CTA r sums slice r (TF/CL floats) over the CL peers via DSMEM, red.v4 to out.
template<int CL>
__global__ void __launch_bounds__(256) dsmk(float* out, int NT, int TF, int G, int nored) {
  extern __shared__ float4 sm4[];
  float* sm = (float*)sm4;
  cg::cluster_group cl = cg::this_cluster();
  int r = cl.block_rank(), g = blockIdx.x / CL;
  int sl = TF / CL;
  for (int t = 0; t < NT; ++t) {
    float* buf = sm + (t & 1) * TF;
    for (int i = threadIdx.x * 4; i < TF; i += 1024)
      *(float4*)(buf + i) = make_float4(r + t, 1.f, 2.f, 3.f);
    cl.sync();
    float* o = out + ((size_t)g * NT + t) * TF + r * sl;
    for (int i = threadIdx.x * 4; i < sl; i += 1024) {
      float4 a = make_float4(0, 0, 0, 0);
#pragma unroll
      for (int p = 0; p < CL; ++p) {
        float4* q = (float4*)cl.map_shared_rank(buf + r * sl + i, p);
        float4 b = *q; a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
      }
      if (!nored) asm volatile("red.global.add.v4.f32 [%0], {%1,%2,%3,%4};" :: "l"(o + i), "f"(a.x), "f"(a.y), "f"(a.z), "f"(a.w) : "memory");
      else if (a.x == -1.f) o[i] = a.y;
    }
  }
  cl.sync();
}
void run(torch::Tensor out, int64_t G, int64_t CLs, int64_t NT, int64_t TF, int64_t nored) {
  auto st = at::cuda::getCurrentCUDAStream();
  int smem = 2 * TF * 4;
  cudaLaunchConfig_t c = {}; c.gridDim = dim3(G * CLs); c.blockDim = dim3(256); c.dynamicSmemBytes = smem; c.stream = st;
  cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = CLs;
  at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1; c.attrs = at; c.numAttrs = 1;
  if (CLs == 8) { cudaFuncSetAttribute(dsmk<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaFuncSetAttribute(dsmk<8>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaLaunchKernelEx(&c, dsmk<8>, out.data_ptr<float>(), (int)NT, (int)TF, (int)G, (int)nored); }
  else { cudaFuncSetAttribute(dsmk<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaLaunchKernelEx(&c, dsmk<4>, out.data_ptr<float>(), (int)NT, (int)TF, (int)G, (int)nored); }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}
'''
m = load_inline("dsmb2", cpp_sources="void run(torch::Tensor out, int64_t G, int64_t CLs, int64_t NT, int64_t TF, int64_t nored);",
                cuda_sources=src, functions=["run"], extra_cuda_cflags=["-O3", "-gencode", "arch=compute_90a,code=sm_90a"])
def t(fn, reps=50):
    g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn(); torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=s): fn()
    torch.cuda.synchronize(); ts = []
    with open("/tmp/okc-gpu.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for _ in range(reps):
            a, b = torch.cuda.Event(True), torch.cuda.Event(True); a.record(); g.replay(); b.record(); torch.cuda.synchronize()
            ts.append(a.elapsed_time(b) * 1000)
    return statistics.median(ts)
# FR T128: N 4608 = 72 tiles of 64 rows; 16 K slices -> 128 CTAs: clusters of CL along K, G = 128/CL groups
for (name, ntiles, T, S) in [("FR T128", 72, 128, 16), ("pool T128", 64, 128, 16), ("unp T128", 128, 128, 8), ("FR T256", 72, 256, 32)]:
    TF = 64 * T
    for CL in (4, 8):
        if 2 * TF * 4 > 220 * 1024: continue
        G = 128 // CL if S >= CL else None
        # each cluster covers S/CL... simplify: total CTAs = 128; groups of K-clusters; each CTA does NT tiles
        ncl_k = S // CL                       # clusters along K per row group
        ngrp = 128 // (CL * ncl_k)            # row groups
        NT = ntiles // ngrp
        out = torch.zeros(128 // CL * NT * TF, device="cuda")
        for nored in (0, 1):
            us = t(lambda: m.run(out, 128 // CL, CL, NT, TF, nored))
            print(f"{name} CL{CL} rowgrp{ngrp} NT{NT}: reds {(128//CL)*NT*TF/1e6:.2f}M floats, {'no red' if nored else 'red   '} {us:.1f} us")
