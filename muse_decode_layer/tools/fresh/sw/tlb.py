# fresh x5 (lead 09:10Z oracle (2), fable #8): cold-start TLB. Time a bare 28 MB reader (qkv-sized, 132 x K CTAs, uint4
# loads) after: A = 512 MB read flush (bench.py's), B = A + one load per 64 KB over a 4 GB buffer (TLB thrash),
# C = A + one load per 64 KB of W itself (TLB pre-warm, ~450 lines into L2), D = A + load per 2 MB of W, E = no flush.
import torch, fcntl, statistics as st, os
from torch.utils.cpp_extension import load_inline
SRC = r"""
#include <cstdint>
__global__ void rd(const uint4* __restrict__ w, long n16, uint32_t* sink) {
  uint32_t s = 0;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n16; i += (long)gridDim.x * blockDim.x) {
    uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(w + i));
    s ^= v.x ^ v.y ^ v.z ^ v.w; }
  if (s == 0x12345678u) sink[0] = s;
}
__global__ void touch(const char* p, long bytes, long stride, uint32_t* sink) {
  uint32_t s = 0;
  for (long o = (blockIdx.x * (long)blockDim.x + threadIdx.x) * stride; o < bytes; o += (long)gridDim.x * blockDim.x * stride)
    s ^= *(const volatile uint32_t*)(p + o);
  if (s == 0x12345678u) sink[1] = s;
}
void r(torch::Tensor w, torch::Tensor sink, int64_t grid) { rd<<<grid, 512>>>((const uint4*)w.data_ptr(), w.numel() / 16, (uint32_t*)sink.data_ptr()); }
void t(torch::Tensor p, torch::Tensor sink, int64_t stride) { touch<<<264, 256>>>((const char*)p.data_ptr(), p.numel(), stride, (uint32_t*)sink.data_ptr()); }
"""
lib = load_inline("fresh_tlb", cpp_sources="void r(torch::Tensor, torch::Tensor, int64_t); void t(torch::Tensor, torch::Tensor, int64_t);",
                  cuda_sources="#include <torch/extension.h>\n" + SRC, functions=["r", "t"], extra_cuda_cflags=["-O3"])
MB = int(os.environ.get("MB", "28"))
W = torch.randint(0, 255, (MB << 20,), dtype=torch.uint8, device="cuda")
FL = torch.empty(512 << 20, dtype=torch.uint8, device="cuda"); SINK = torch.zeros(4, dtype=torch.int64, device="cuda")
BIG = torch.empty(4 << 30, dtype=torch.uint8, device="cuda"); S = torch.zeros(4, dtype=torch.int32, device="cuda")
def flush(): torch.sum(FL.view(torch.int64), 0, out=SINK[0])
PRE = {'A 512MB flush': lambda: flush(), 'B A + 4GB/64KB touch': lambda: (flush(), lib.t(BIG, S, 65536)),
       'C A + W/64KB pre-touch': lambda: (flush(), lib.t(W, S, 65536)), 'D A + W/2MB pre-touch': lambda: (flush(), lib.t(W, S, 2 << 20)),
       'F A + W/4KB pre-touch': lambda: (flush(), lib.t(W, S, 4096)), 'E no flush': lambda: None}
for grid in (132, 264, 528):
    for name, pre in PRE.items():
        ts = []
        for rep in range(40):
            pre(); e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
            torch.cuda._sleep(20000)
            with open('/tmp/okc-gpu.lock', 'w') as lk:
                fcntl.flock(lk, fcntl.LOCK_EX)
                e0.record(); lib.r(W, S, grid); e1.record(); torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) * 1e3)
        m = st.median(ts[5:])
        print(f"grid {grid:4d} {name:24s} {m:7.2f} us  {MB * 1.048576 / m * 1e3 / 1e3:5.2f} TB/s", flush=True)
