"""n5s29: H200 HBM read-rate probe: LDG.128 (grid-stride / blocked, cache hints) vs cp.async.bulk rings.
Cold: each graph replays K kernels over K distinct buffers (> L2). Prints TB/s per variant and size."""
import os, sys, torch, fcntl, contextlib
from torch.utils.cpp_extension import load_inline
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
src = r'''
#include <cuda.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
template<int U, int HINT, bool BLK>
__global__ void __launch_bounds__(512) ldg(const uint4* __restrict__ p, size_t n16, unsigned* out) {
  unsigned acc = 0;
  size_t nt = (size_t)gridDim.x * blockDim.x;
  size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t beg, end, step;
  if (BLK) { size_t per = (n16 + gridDim.x - 1) / gridDim.x; size_t b = blockIdx.x * per; beg = b + threadIdx.x;
             end = min(n16, b + per); step = blockDim.x; }
  else { beg = tid; end = n16; step = nt; }
  for (size_t i = beg; i < end; i += step * U) {
    uint4 v[U];
#pragma unroll
    for (int u = 0; u < U; u++) {
      size_t j = i + (size_t)u * step;
      if (j < end) {
        if (HINT == 0) v[u] = __ldg(p + j);
        else if (HINT == 1) asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v[u].x), "=r"(v[u].y), "=r"(v[u].z), "=r"(v[u].w) : "l"(p + j));
        else asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v[u].x), "=r"(v[u].y), "=r"(v[u].z), "=r"(v[u].w) : "l"(p + j));
      } else v[u] = make_uint4(0,0,0,0);
    }
#pragma unroll
    for (int u = 0; u < U; u++) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
  }
  if (acc == 0x9e3779b9u) out[0] = acc;
}
__device__ __forceinline__ unsigned sa(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__global__ void __launch_bounds__(32) bulk(const char* __restrict__ p, size_t nbytes, int S, int NST, unsigned* out) {
  extern __shared__ __align__(128) char sm[];
  __shared__ __align__(8) uint64_t full[16];
  if (threadIdx.x != 0) return;
  size_t per = (nbytes / gridDim.x) / S * S; const char* base = p + per * blockIdx.x;
  int nch = per / S;
  for (int s = 0; s < NST; s++) asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(sa(&full[s])));
  asm volatile("fence.mbarrier_init.release.cluster;");
  auto issue = [&](int c) {
    int s = c % NST;
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"(sa(&full[s])), "r"(S));
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(sa(sm + (size_t)s * S)), "l"(base + (size_t)c * S), "r"(S), "r"(sa(&full[s])) : "memory");
  };
  for (int c = 0; c < NST && c < nch; c++) issue(c);
  unsigned acc = 0;
  for (int c = 0; c < nch; c++) {
    int s = c % NST; unsigned par = (c / NST) & 1, ok = 0;
    while (!ok) asm volatile("{ .reg .pred q; mbarrier.try_wait.parity.shared.b64 q, [%1], %2; selp.u32 %0,1,0,q; }" : "=r"(ok) : "r"(sa(&full[s])), "r"(par) : "memory");
    acc ^= *(volatile unsigned*)(sm + (size_t)s * S);
    if (c + NST < nch) issue(c + NST);
  }
  if (acc == 0x9e3779b9u) out[0] = acc;
}
void run_ldg(torch::Tensor buf, int64_t mode, int64_t grid, torch::Tensor out) {
  size_t n16 = buf.numel() / 16; const uint4* p = (const uint4*)buf.data_ptr(); unsigned* o = (unsigned*)out.data_ptr();
  switch (mode) {
    case 0: ldg<4,0,false><<<grid,512,0,at::cuda::getCurrentCUDAStream()>>>(p,n16,o); break;
    case 1: ldg<8,0,false><<<grid,512,0,at::cuda::getCurrentCUDAStream()>>>(p,n16,o); break;
    case 2: ldg<8,1,false><<<grid,512,0,at::cuda::getCurrentCUDAStream()>>>(p,n16,o); break;
    case 3: ldg<8,2,false><<<grid,512,0,at::cuda::getCurrentCUDAStream()>>>(p,n16,o); break;
    case 4: ldg<8,0,true><<<grid,512,0,at::cuda::getCurrentCUDAStream()>>>(p,n16,o); break;
    case 5: ldg<16,0,false><<<grid,512,0,at::cuda::getCurrentCUDAStream()>>>(p,n16,o); break;
    case 6: ldg<8,1,true><<<grid,512,0,at::cuda::getCurrentCUDAStream()>>>(p,n16,o); break;
  }
}
void run_bulk(torch::Tensor buf, int64_t S, int64_t NST, int64_t grid, torch::Tensor out) {
  int smem = S * NST;
  cudaFuncSetAttribute(bulk, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  bulk<<<grid, 32, smem, at::cuda::getCurrentCUDAStream()>>>((const char*)buf.data_ptr(), buf.numel(), S, NST, (unsigned*)out.data_ptr());
}
'''
cpp = "void run_ldg(torch::Tensor, int64_t, int64_t, torch::Tensor); void run_bulk(torch::Tensor, int64_t, int64_t, int64_t, torch::Tensor);"
m = load_inline("bwprobe_n5s29", cpp, cuda_sources=src, functions=["run_ldg", "run_bulk"],
                extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
LOCK = "/tmp/okc-gpu.lock"
@contextlib.contextmanager
def gpulock():
    if not os.path.exists("/home/daytona"):
        yield; return
    with open(LOCK, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX); yield
nsm = torch.cuda.get_device_properties(0).multi_processor_count
out = torch.zeros(4, dtype=torch.int32, device="cuda")
MB = 1 << 20
K = 6
for size_mb in [int(a) for a in (sys.argv[1:] or ["85", "280", "560"])]:
    bufs = [torch.randint(0, 255, (size_mb * MB,), dtype=torch.uint8, device="cuda") for _ in range(K)]
    variants = {}
    for mode, nm in [(0, "ldgU4"), (1, "ldgU8"), (2, "ldgU8nc256"), (3, "ldgU8cg"), (4, "ldgU8blk"), (5, "ldgU16"), (6, "ldgU8nc_blk")]:
        for cps in (1, 2, 4):
            variants[f"{nm}_g{cps}"] = (lambda b, mode=mode, cps=cps: m.run_ldg(b, mode, nsm * cps, out))
    for S, NST in [(18432, 4), (36864, 4), (18432, 8), (36864, 6), (65536, 3)]:
        for cps in (1, 2):
            if S * NST * cps > 220 * 1024: continue
            variants[f"bulk{S//1024}x{NST}_g{cps}"] = (lambda b, S=S, NST=NST, cps=cps: m.run_bulk(b, S, NST, nsm * cps, out))
    res = []
    for name, fn in variants.items():
        for b in bufs: fn(b)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for b in bufs: fn(b)
        ts = []
        with gpulock():
            for r in range(15):
                e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                e0.record(); g.replay(); e1.record(); e1.synchronize()
                ts.append(e0.elapsed_time(e1) * 1e3 / K)
        ts.sort(); t = ts[len(ts) // 2]
        res.append((size_mb * MB / t / 1e6, name, t))
        del g
    res.sort(reverse=True)
    print(f"== {size_mb} MB ({torch.cuda.get_device_name()}) top / all (TB/s, us per kernel):")
    for tb, name, t in res:
        print(f"  {name:18s} {tb:5.2f} TB/s {t:8.1f} us")
    del bufs; torch.cuda.empty_cache()
