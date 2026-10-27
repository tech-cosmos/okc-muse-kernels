import torch, fcntl, sys
from torch.utils.cpp_extension import load_inline
src = r'''
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
// each CTA streams [lo,hi) of the buffer through a smem ring of S stages x B bytes, one issuing thread
template<int S>
__global__ void __launch_bounds__(128) tr(const char* p, long n, int B, unsigned* o) {
  extern __shared__ __align__(128) char sm[];
  __shared__ __align__(8) unsigned long long bar[S];
  long per = (n / gridDim.x) & ~15L; long lo = per * blockIdx.x; long hi = blockIdx.x == gridDim.x - 1 ? n : lo + per;
  int nst = (int)((hi - lo + B - 1) / B);
  if (threadIdx.x == 0) { for (int s = 0; s < S; s++) asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"((unsigned)__cvta_generic_to_shared(&bar[s]))); asm volatile("fence.mbarrier_init.release.cluster;"); }
  __syncthreads();
  auto issue = [&](int i) {
    int s = i % S; long off = lo + (long)i * B; int sz = (int)min((long)B, hi - off);
    unsigned b = (unsigned)__cvta_generic_to_shared(&bar[s]); unsigned d = (unsigned)__cvta_generic_to_shared(sm + s * B);
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"(b), "r"(sz));
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" :: "r"(d), "l"(p + off), "r"(sz), "r"(b) : "memory");
  };
  if (threadIdx.x == 0) for (int i = 0; i < min(S, nst); i++) issue(i);
  unsigned a = 0;
  for (int i = 0; i < nst; i++) {
    int s = i % S; unsigned ph = (i / S) & 1; unsigned b = (unsigned)__cvta_generic_to_shared(&bar[s]);
    asm volatile("{ .reg .pred P; W: mbarrier.try_wait.parity.shared.b64 P, [%0], %1; @!P bra W; }" :: "r"(b), "r"(ph));
    a ^= ((unsigned*)(sm + s * B))[threadIdx.x];
    __syncthreads();
    if (threadIdx.x == 0 && i + S < nst) issue(i + S);
  }
  if (a == 0x12345) o[0] = a;
}
void trl(torch::Tensor p, torch::Tensor o, int64_t g, int64_t S, int64_t B) {
  auto st = c10::cuda::getCurrentCUDAStream();
  if (S == 2) { cudaFuncSetAttribute(tr<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 2 * B); tr<2><<<g, 128, 2 * B, st>>>((const char*)p.data_ptr(), p.numel(), B, (unsigned*)o.data_ptr()); }
  if (S == 4) { cudaFuncSetAttribute(tr<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 4 * B); tr<4><<<g, 128, 4 * B, st>>>((const char*)p.data_ptr(), p.numel(), B, (unsigned*)o.data_ptr()); }
  if (S == 8) { cudaFuncSetAttribute(tr<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, 8 * B); tr<8><<<g, 128, 8 * B, st>>>((const char*)p.data_ptr(), p.numel(), B, (unsigned*)o.data_ptr()); }
}
'''
lib = load_inline("tr71", cpp_sources="void trl(torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t);", cuda_sources=src, functions=["trl"], extra_cuda_cflags=["-arch=sm_90a"])
exec(open("/tmp/pj70/rd.py").read().split("def tg")[0].split("lib = load_inline")[0].replace("src = r'''", "_x = r'''"))
o = torch.zeros(4, dtype=torch.int32, device="cuda"); flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
exec("def tg" + open("/tmp/pj70/rd.py").read().split("def tg")[1].split("e = tg")[0])
e = tg(lambda: None)
mb = float(sys.argv[1]) if len(sys.argv) > 1 else 28.4
b = torch.empty(int(mb * 1e6) // 16 * 16, dtype=torch.uint8, device="cuda")
for g in (132, 264):
  for S, B in ((2, 32768), (4, 16384), (4, 32768), (8, 8192), (8, 16384), (4, 49152), (2, 65536)):
    if g == 264 and S * B > 110000: continue
    t = tg(lambda: lib.trl(b, o, g, S, B)) - e
    print(f"tma {mb} MB g{g} S{S} B{B}: {t:.2f} us = {mb / t:.2f} TB/s")
