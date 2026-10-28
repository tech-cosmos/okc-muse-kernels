# n6s2: RS wgmma latency: one WG, per iteration: fence, NG wgmma m64nNk16 (distinct accumulators, distinct A quads
# loaded from smem once), commit, wait<0>. cycles/iteration = issue -> completion latency of a group of NG.
# Also throughput (wait<1>) for reference.
import torch, os, sys, fcntl, subprocess, glob
from torch.utils.cpp_extension import load_inline
def gen_rs(n):
    na = n // 2; dl = ", ".join(f"%{i}" for i in range(na)); al = ", ".join(f"%{na + i}" for i in range(4))
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(na))
    return (f"template <> __device__ __forceinline__ void wg_rs<{n}>(float* d, const uint32_t* a, uint64_t desc) {{\n"
            f'  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, %{na + 5}, 0;\\n"\n'
            f'    "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{dl}}}, {{{al}}}, %{na + 4}, p, 1, 1, 0;\\n}}\\n"\n'
            f'    : {outs} : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));\n}}\n')
NS = (8, 16, 32, 64)
SRC = r"""
#include <cstdint>
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ uint64_t sw128_desc(const void* p) {
  return (uint64_t)((smem_u32(p) & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62); }
template <int N> __device__ __forceinline__ void wg_rs(float* d, const uint32_t* a, uint64_t desc);
@@GEN@@
template <int N, int NG, int W>
__global__ void __launch_bounds__(128, 1) lat(unsigned long long* out, float* sink, int iters) {
  extern __shared__ __align__(1024) char sm[];
  const int tid = threadIdx.x;
  for (int i = tid; i < (64 + 256) * 128 / 4; i += 128) reinterpret_cast<uint32_t*>(sm)[i] = 0x3F803F80u ^ (i * 2654435761u & 0x00070007u);
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  float acc[NG][N / 2];
#pragma unroll
  for (int r = 0; r < NG; ++r)
#pragma unroll
    for (int i = 0; i < N / 2; ++i) acc[r][i] = 0.f;
  uint32_t a[NG][4];
  const volatile uint32_t* vs = reinterpret_cast<const volatile uint32_t*>(sm);
#pragma unroll
  for (int r = 0; r < NG; ++r)
#pragma unroll
    for (int i = 0; i < 4; ++i) a[r][i] = vs[(tid * 16 + r * 4 + i) & 4095];
  const uint64_t db = sw128_desc(sm + 64 * 128);
  __syncthreads();
  const unsigned long long t0 = clock64();
#pragma unroll 1
  for (int it = 0; it < iters; ++it) {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
    for (int k = 0; k < NG; ++k) wg_rs<N>(acc[k], a[k], db + (k & 3) * 2);
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    if (W == 0) asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    else asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  const unsigned long long t1 = clock64();
  float s = 0.f;
#pragma unroll
  for (int r = 0; r < NG; ++r)
#pragma unroll
    for (int i = 0; i < N / 2; ++i) s += acc[r][i];
  sink[blockIdx.x * 128 + tid] = s;
  if (tid == 0) out[blockIdx.x] = t1 - t0;
}
int64_t run(int64_t n, int64_t ng, int64_t w, torch::Tensor out, torch::Tensor sink, int64_t iters) {
  const int SM = (64 + 256) * 128 + 1024;
  #define GO(NN, G, WW) if (n == NN && ng == G && w == WW) { auto k = lat<NN, G, WW>; cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SM); \
    k<<<1, 128, SM>>>((unsigned long long*)out.data_ptr(), (float*)sink.data_ptr(), (int)iters); return 0; }
  #define GN(NN) GO(NN, 1, 0) GO(NN, 4, 0) GO(NN, 1, 1) GO(NN, 4, 1)
  GN(8) GN(16) GN(32) GN(64)
  return -1;
}
"""
SRC = SRC.replace("@@GEN@@", "".join(gen_rs(n) for n in NS))
NAME = "n6s2_lat1"
lib = load_inline(NAME, cpp_sources="int64_t run(int64_t, int64_t, int64_t, torch::Tensor, torch::Tensor, int64_t);",
                  cuda_sources="#include <torch/extension.h>\n" + SRC, functions=["run"], extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"])
so = glob.glob(os.path.expanduser(f"~/.cache/torch_extensions/*/{NAME}/{NAME}.so"))[0]
sass = subprocess.run(["/usr/local/cuda-13.0/bin/cuobjdump", "-sass", so], capture_output=True, text=True).stdout
ser = {blk.split()[0]: blk.count("DEPBAR.LE gsb0, 0x0") for blk in sass.split("Function : ")[1:]}
out = torch.zeros(4, dtype=torch.int64, device="cuda"); sink = torch.zeros(128 * 4, device="cuda")
IT = 1000
for n in NS:
    row = []
    for ng in (1, 4):
        for w in (0, 1):
            lib.run(n, ng, w, out, sink, 20); torch.cuda.synchronize()
            with open('/tmp/okc-gpu.lock', 'w') as lk:
                fcntl.flock(lk, fcntl.LOCK_EX)
                lib.run(n, ng, w, out, sink, IT); torch.cuda.synchronize()
            row.append(f"NG{ng} wait<{w}> {out[0].item() / IT:6.1f} (ser0 {ser.get(f'_Z3latILi{n}ELi{ng}ELi{w}EEvPyPfi', '?')})")
    print(f"N{n:2d}: " + " | ".join(row), flush=True)
