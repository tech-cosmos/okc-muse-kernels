# n19: DSMEM reduce-scatter bandwidth vs cluster size (H100). Each CTA holds a P-byte FP32 partial in smem;
# reduce-scatter: rank r sums slice r (P/C bytes) from all C peers (pull: ld.shared::cluster v4) and writes it out.
# Usage: python dsm.py   (prints us per kernel for C in 2,4,8,16 and P in 32/64/128 KB, grid 128 CTAs)
import torch, fcntl, sys, os
from torch.utils.cpp_extension import load_inline
src = r'''
#include <cuda.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
template<int C>
__global__ void __launch_bounds__(1024) rs(float* out, int P4, int reps) {
  extern __shared__ float4 buf[];
  cg::cluster_group cl = cg::this_cluster();
  int r = cl.block_rank();
  for (int i = threadIdx.x; i < P4; i += blockDim.x) buf[i] = make_float4(r, i, 1.f, 2.f);
  cl.sync();
  int S4 = P4 / C;
  float4 acc = make_float4(0,0,0,0);
  for (int rep = 0; rep < reps; rep++) {
    for (int i0 = threadIdx.x; i0 < S4; i0 += blockDim.x * RS_U) {
      float4 v[RS_U][C];
      #pragma unroll
      for (int u = 0; u < RS_U; u++)
      #pragma unroll
      for (int p = 0; p < C; p++) {
        float4* pb = cl.map_shared_rank(buf, (r + p) % C);
        int i = i0 + u * blockDim.x;
        v[u][p] = i < S4 ? pb[r * S4 + i] : make_float4(0,0,0,0);
      }
      #pragma unroll
      for (int u = 0; u < RS_U; u++)
      #pragma unroll
      for (int p = 0; p < C; p++) { acc.x += v[u][p].x; acc.y += v[u][p].y; acc.z += v[u][p].z; acc.w += v[u][p].w; }
    }
    cl.sync();
  }
  if (acc.x == 12345.f) out[blockIdx.x] = acc.y + acc.z + acc.w;
}
template<int C> void launch(torch::Tensor out, int64_t P, int64_t reps, int64_t grid) {
  int P4 = P / 16;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid); cfg.blockDim = dim3(RS_NT); cfg.dynamicSmemBytes = P;
  cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = C; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  cudaFuncSetAttribute(rs<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, P);
  if (C == 16) cudaFuncSetAttribute(rs<C>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  cudaLaunchKernelEx(&cfg, rs<C>, out.data_ptr<float>(), P4, (int)reps);
}
void run(torch::Tensor out, int64_t C, int64_t P, int64_t reps, int64_t grid) {
  if (C == 1) launch<1>(out, P, reps, grid); if (C == 2) launch<2>(out, P, reps, grid);
  if (C == 4) launch<4>(out, P, reps, grid); if (C == 8) launch<8>(out, P, reps, grid);
  if (C == 16) launch<16>(out, P, reps, grid);
}
'''
m = load_inline(name="n19_dsm_u%s_n%s" % (os.environ.get("U","2"), os.environ.get("NT","1024")), cpp_sources="void run(torch::Tensor, int64_t, int64_t, int64_t, int64_t);",
                cuda_sources=src, functions=["run"], extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3","-DRS_U=%s" % os.environ.get("U","2"), "-DRS_NT=%s" % os.environ.get("NT","1024")])
out = torch.zeros(1024, device="cuda")
def t(C, P, reps, grid=128):
    m.run(out, C, P, reps, grid); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    with open("/tmp/okc-gpu.lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        ts = []
        for _ in range(20):
            e0.record(); m.run(out, C, P, reps, grid); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1) * 1e3)
        fcntl.flock(f, fcntl.LOCK_UN)
    ts.sort(); return ts[len(ts) // 2]
for C in (1, 2, 4, 8, 16):
    for P in (32768, 65536, 131072):
        a, b = t(C, P, 1), t(C, P, 11)
        per = (b - a) / 10  # us per reduce-scatter pass
        recv = P * (C - 1) / C  # bytes pulled from peers per CTA per pass
        print(f"C{C:2d} P{P//1024:4d}KB: pass {per:6.2f} us  remote-pull {recv/1024:6.1f} KB/CTA -> {recv/per/1e3 if per>0 else 0:7.1f} GB/s/SM  (1 pass kernel {a:.2f})")
