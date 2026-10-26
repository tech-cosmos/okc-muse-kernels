"""s31: do L2 lines prefetched with an evict_last policy survive a 512 MB ordinary-read flush? (H100)
Times reading a 24 MB buffer after: (a) nothing (cold), (b) normal bulk prefetch + flush, (c) evict_last prefetch + flush,
(d) evict_last prefetch, no flush (warm reference)."""
import torch, fcntl
from torch.utils.cpp_extension import load_inline
cu = r'''
#include <torch/extension.h>
__global__ void pf(const char* p, long n, int mode) {
  for (long o = (blockIdx.x * (long)blockDim.x + threadIdx.x) * 4096; o < n; o += (long)gridDim.x * blockDim.x * 4096) {
    if (mode == 1) asm volatile("cp.async.bulk.prefetch.L2.global [%0], 4096;" :: "l"(p + o) : "memory");
    else asm volatile("{ .reg .b64 pl; createpolicy.fractional.L2::evict_last.b64 pl, 1.0; cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], 4096, pl; }" :: "l"(p + o) : "memory");
  }
}
__global__ void rd(const int4* p, long n, int* out) {
  int4 a = make_int4(0,0,0,0);
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) { int4 v = __ldcg(p + i); a.x ^= v.x; a.y ^= v.y; }
  if (a.x == 12345 && a.y == 678) out[0] = 1;
}
void run_pf(torch::Tensor t, int mode) { pf<<<132, 128>>>((const char*)t.data_ptr(), t.numel(), mode); }
void run_rd(torch::Tensor t, torch::Tensor o) { rd<<<132*4, 512>>>((const int4*)t.data_ptr(), t.numel() / 16, o.data_ptr<int>()); }
'''
m = load_inline("evl_s31", "void run_pf(torch::Tensor t, int mode); void run_rd(torch::Tensor t, torch::Tensor o);", cu,
                functions=["run_pf", "run_rd"], extra_cuda_cflags=["-arch=sm_90a", "-O3"], verbose=False)
buf = torch.zeros(int(__import__("os").environ.get("MB", "24")) << 20, dtype=torch.uint8, device="cuda"); o = torch.zeros(1, dtype=torch.int32, device="cuda")
fl = torch.empty(512 << 20, dtype=torch.uint8, device="cuda"); sink = torch.zeros(1, device="cuda")
def flush(): fl.add_(1)  # read+write 512 MB ordinary
def t_read():
    s, e = torch.cuda.Event(True), torch.cuda.Event(True); s.record(); m.run_rd(buf, o); e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) * 1e3
res = {k: [] for k in "abcd"}
lock = open("/tmp/okc-gpu.lock", "w"); fcntl.flock(lock, fcntl.LOCK_EX)
for r in range(15):
    flush(); torch.cuda.synchronize(); res["a"].append(t_read())
    flush(); m.run_pf(buf, 1); torch.cuda.synchronize(); flush(); torch.cuda.synchronize(); res["b"].append(t_read())
    flush(); m.run_pf(buf, 2); torch.cuda.synchronize(); flush(); torch.cuda.synchronize(); res["c"].append(t_read())
    flush(); m.run_pf(buf, 2); torch.cuda.synchronize(); res["d"].append(t_read())
fcntl.flock(lock, fcntl.LOCK_UN)
import statistics
for k, v in res.items(): print(k, f"{statistics.median(v):.2f} us")
