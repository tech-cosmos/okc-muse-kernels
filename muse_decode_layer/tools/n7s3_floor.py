"""n7s3: batch-1 full-layer K/V gather floors (decode 1 full: 80K keys x 2 KV heads, page size 1, FP8 K+V rows of
128 B each = 20.5 MB). Times one-kernel CUDA graphs, cold L2 (512 MB read before each replay), median of N, minus an
empty-kernel graph. Variants: rows through the page table (random) vs rows 0..n-1 (dense), grid x unroll.
  python tools/n7s3_floor.py            (prints us per variant)"""
import fcntl, statistics, sys, torch
sys.path.insert(0, ".")
import reference
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
template <int U>
__global__ void __launch_bounds__(256) gfloor(const int* __restrict__ idx, int n, const uint4* __restrict__ K,
                                              const uint4* __restrict__ V, uint4* sink, int dense) {
  const int lane8 = threadIdx.x & 7;
  const int slot = (blockIdx.x * blockDim.x + threadIdx.x) >> 3;
  const int stride = (gridDim.x * blockDim.x) >> 3;
  uint4 acc = make_uint4(0, 0, 0, 0);
  for (int r0 = slot; r0 < n; r0 += stride * U) {
    int row[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      int r = r0 + u * stride;
      row[u] = r < n ? (dense ? r : __ldg(idx + r)) : -1;
    }
    uint4 k[U], v[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      if (row[u] >= 0) { k[u] = __ldcs(K + (size_t)row[u] * 8 + lane8); v[u] = __ldcs(V + (size_t)row[u] * 8 + lane8); }
      else { k[u] = make_uint4(0, 0, 0, 0); v[u] = k[u]; }
    }
#pragma unroll
    for (int u = 0; u < U; ++u) { acc.x ^= k[u].x ^ v[u].y; acc.y ^= k[u].y ^ v[u].z; acc.z ^= k[u].z ^ v[u].w; acc.w ^= k[u].w ^ v[u].x; }
  }
  if (acc.x == 0x12345678u && acc.y == 0x9abcdef0u) sink[0] = acc;
}
__global__ void empty_k(uint4* sink) { if (threadIdx.x == 999) sink[0] = make_uint4(1, 1, 1, 1); }
void run(torch::Tensor idx, int n, torch::Tensor K, torch::Tensor V, torch::Tensor sink, int dense, int blocks, int U) {
  auto s = at::cuda::getCurrentCUDAStream();
  auto a = idx.data_ptr<int>(); auto k = (const uint4*)K.data_ptr(); auto v = (const uint4*)V.data_ptr();
  auto o = (uint4*)sink.data_ptr();
  if (blocks == 0) { empty_k<<<1, 32, 0, s>>>(o); return; }
  if (U == 1) gfloor<1><<<blocks, 256, 0, s>>>(a, n, k, v, o, dense);
  if (U == 2) gfloor<2><<<blocks, 256, 0, s>>>(a, n, k, v, o, dense);
  if (U == 4) gfloor<4><<<blocks, 256, 0, s>>>(a, n, k, v, o, dense);
  if (U == 8) gfloor<8><<<blocks, 256, 0, s>>>(a, n, k, v, o, dense);
}
"""
mod = load_inline("n7s3_floor", cpp_sources="void run(torch::Tensor, int, torch::Tensor, torch::Tensor, torch::Tensor, int, int, int);",
                  cuda_sources="#include <ATen/cuda/CUDAContext.h>\n" + SRC, functions=["run"],
                  extra_cuda_cflags=["-O3", "-arch=sm_90a"], verbose=False)

for load, q_len in (("decode 1 full", 1), ("verify 1 full", 8)):
    d = reference.generate_input(batch=1, ctx=80000, layer="full", seed=99, q_len=q_len, page=1, dist="soft")
    n = int(d["seq_lens"][0])
    idx = d["page_table"][0, :n].contiguous()
    K, V = d["k_cache"], d["v_cache"]
    assert K.shape[1:] == (1, 2, 64) and K.dtype.itemsize == 1, K.shape
    sink = torch.zeros(4, dtype=torch.int32, device="cuda")
    flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda"); fs = torch.zeros((), dtype=torch.int64, device="cuda")
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    res = {}
    variants = [("empty", 0, 0, 0)]
    for dense in (0, 1):
        for mult in (2, 4, 8):
            for U in (2, 4, 8):
                variants.append((f"{'dense' if dense else 'rand '} blk {mult}x{sms} U{U}", dense, mult * sms, U))
    for name, dense, blocks, U in variants:
        g = torch.cuda.CUDAGraph()
        mod.run(idx, n, K, V, sink, dense, blocks, U); torch.cuda.synchronize()
        with torch.cuda.graph(g):
            mod.run(idx, n, K, V, sink, dense, blocks, U)
        ts = []
        with open("/tmp/okc-gpu.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for _ in range(40):
                torch.sum(flush.view(torch.int64), 0, out=fs)
                a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                a.record(); g.replay(); b.record(); b.synchronize()
                ts.append(a.elapsed_time(b) * 1000)
        res[name] = statistics.median(ts)
    e = res.pop("empty")
    print(f"== {load}: n={n} rows, {2 * 128 * n / 1e6:.1f} MB K+V, empty graph {e:.2f} us", flush=True)
    for k_, v_ in res.items():
        print(f"  {k_:28s} {v_:7.2f} us (-empty {v_ - e:6.2f}) {2 * 128 * n / (v_ - e) / 1e6:5.2f} TB/s", flush=True)
    del d, K, V, flush
    torch.cuda.empty_cache()
