# s39 probe: does a 1-block/SM SMEM fill (TMA bulk, ~192 KB/SM) on a side stream slow a concurrent m=1 GEMV?
import torch, fcntl, sys, os
from torch.utils.cpp_extension import load_inline
src = r'''
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
extern "C" __global__ void __launch_bounds__(128) fill(const char* __restrict__ w, long per_block, int chunk, unsigned long long* sink){
  extern __shared__ __align__(128) char sm[];
  __shared__ __align__(8) unsigned long long bar;
  unsigned b = (unsigned)__cvta_generic_to_shared(&bar);
  unsigned s = (unsigned)__cvta_generic_to_shared(sm);
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(b));
    asm volatile("fence.mbarrier_init.release.cluster;");
    const char* src = w + (long)blockIdx.x * per_block;
    if (chunk > 1000000) {  // paced L2 prefetch: 32 KB per SM every (chunk - 1000000) ns
      unsigned long long gap = chunk - 1000000, t0, t;
      for (long o = 0; o < per_block; o += 32768) {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(src + o), "r"(32768) : "memory");
        do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); } while (t - t0 < gap);
      }
    } else if (chunk == 999999) {  // unpaced L2 prefetch of everything
      for (long o = 0; o < per_block; o += 32768)
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(src + o), "r"(32768) : "memory");
    } else if (chunk > 0) {
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"(b), "r"((unsigned)per_block));
    for (long o = 0; o < per_block; o += chunk)
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
        :: "r"(s + (unsigned)o), "l"(src + o), "r"(chunk), "r"(b) : "memory");
    unsigned done = 0;
    while (!done) asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared.b64 p, [%1], 0; selp.u32 %0,1,0,p; }" : "=r"(done) : "r"(b));
    } else {  // throttled: one chunk of -chunk bytes in flight per SM
    int c = -chunk; unsigned ph = 0;
    for (long o = 0; o < per_block; o += c, ph ^= 1) {
      asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"(b), "r"((unsigned)c));
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
        :: "r"(s + (unsigned)o), "l"(src + o), "r"(c), "r"(b) : "memory");
      unsigned done = 0;
      while (!done) asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared.b64 p, [%1], %2; selp.u32 %0,1,0,p; }" : "=r"(done) : "r"(b), "r"(ph));
    }
    }
    sink[blockIdx.x] = *(unsigned long long*)(sm + per_block - 8);
  }
}
void run(torch::Tensor w, long per_block, int chunk, int blocks, torch::Tensor sink){
  static bool init = false;
  if (!init) { cudaFuncSetAttribute(fill, cudaFuncAttributeMaxDynamicSharedMemorySize, 200*1024); init = true; }
  fill<<<blocks, 128, per_block, c10::cuda::getCurrentCUDAStream()>>>((const char*)w.data_ptr(), per_block, chunk, (unsigned long long*)sink.data_ptr());
}
'''
m = load_inline("pj39fill", cpp_sources="void run(torch::Tensor w, long per_block, int chunk, int blocks, torch::Tensor sink);",
                cuda_sources=src.replace("#include <torch/extension.h>\n", "#include <torch/extension.h>\n", 1), functions=["run"],
                extra_cuda_cflags=["-O3", "-arch=sm_90a"], verbose=False)
dev = "cuda"
PB = int(os.environ.get("PB", 192 * 1024)); NB = int(os.environ.get("NB", 132))
wf = torch.empty(NB * PB, dtype=torch.uint8, device=dev)
wf2 = torch.empty(NB * PB, dtype=torch.uint8, device=dev)
xs = torch.randn(int(os.environ.get("XS", 1 << 16)), device=dev)
sink = torch.empty(NB, dtype=torch.int64, device=dev)
Wo = torch.randn(8192, 1024, dtype=torch.bfloat16, device=dev); x = torch.randn(1, 1024, dtype=torch.bfloat16, device=dev)
Wp = torch.randn(4096, 8192, dtype=torch.bfloat16, device=dev)  # stand-in for a following latency-ish phase, unused
flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
RD = int(os.environ.get('RD', 0)); rout = torch.empty((), dtype=torch.float16, device='cuda'); side = torch.cuda.Stream(); CH = int(os.environ.get('CH', 16384))
def mk(mode):
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        s0 = torch.cuda.current_stream()
        flush.zero_()
        e0 = torch.cuda.Event(); e0.record()
        if mode in ("fill", "both"):
            side.wait_event(e0)
            with torch.cuda.stream(side): m.run(wf, PB, CH, NB, sink)
        if mode in ("o", "both"):
            for _ in range(int(os.environ.get("NO", 2))):
                if os.environ.get("F2"): m.run(wf2, PB, 16384, NB, sink)
                elif os.environ.get("LAT"): xs.mul_(1.0001)
                else: y = torch.nn.functional.linear(x, Wo)
        if mode in ("fill", "both"):
            e1 = torch.cuda.Event(); e1.record(side); s0.wait_event(e1)
        if RD: torch.amax(wf.view(torch.float16), dim=0, out=rout)
    return g
def tm(g, reps=60):
    ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b) * 1000)
    ts.sort(); return ts[len(ts) // 2]
m.run(wf, PB, CH, NB, sink); torch.nn.functional.linear(x, Wo); torch.cuda.synchronize()
gs = {k: mk(k) for k in ("none", "o", "fill", "both")}
torch.cuda.synchronize()
with open("/tmp/okc-gpu.lock", "w") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    for r in range(3):
        t = {k: tm(g) for k, g in gs.items()}
        print(r, {k: round(v - t["none"], 2) for k, v in t.items()}, "MB", NB * PB / 1e6, "none_abs", round(t["none"], 2))
