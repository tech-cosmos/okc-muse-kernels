"""n64: HBM ceiling for the full-layer attention's KV stream (no math). Per key: page_table int32 + K row 128 B + V row
128 B at a random page (page=1, randperm pool as reference.generate_input). Compare to a contiguous read of the same bytes.
python dev/n64/gather_oracle.py B [ctx]"""
import sys, torch
sys.path.insert(0, ".")
import bench, reference
from torch.utils.cpp_extension import load_inline

src = r"""
#include <torch/extension.h>
#include <cstdint>
template<int U>
__global__ void gat(const int* __restrict__ pt, int maxp, const int* __restrict__ lens, int B,
                    const uint4* __restrict__ K, const uint4* __restrict__ V, unsigned long long* sink, long long total) {
  // global key index -> (b, j) via flat layout: keys of batch b are pt[b*maxp + j], j < lens[b]; we iterate b-major
  int lane8 = threadIdx.x & 7;
  long long row0 = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 3;
  long long stride = ((long long)gridDim.x * blockDim.x) >> 3;
  uint32_t acc = 0;
  for (long long r = row0; r < total; r += stride * U) {
    int pg[U]; uint4 k[U], v[U];
    #pragma unroll
    for (int u = 0; u < U; ++u) { long long rr = r + u * stride; pg[u] = rr < total ? __ldg(pt + rr) : -1; }
    #pragma unroll
    for (int u = 0; u < U; ++u) if (pg[u] >= 0) { k[u] = __ldg(K + (long long)pg[u] * 8 + lane8); v[u] = __ldg(V + (long long)pg[u] * 8 + lane8); }
    #pragma unroll
    for (int u = 0; u < U; ++u) if (pg[u] >= 0) acc ^= k[u].x ^ k[u].w ^ v[u].y ^ v[u].z;
  }
  if (acc == 0x12345678u) atomicAdd(sink, 1ull);
}
template<int U>
__global__ void seq(const uint4* __restrict__ K, const uint4* __restrict__ V, unsigned long long* sink, long long n16) {
  long long i0 = (long long)blockIdx.x * blockDim.x + threadIdx.x, st = (long long)gridDim.x * blockDim.x;
  uint32_t acc = 0;
  for (long long i = i0; i < n16; i += st * U) {
    uint4 k[U], v[U];
    #pragma unroll
    for (int u = 0; u < U; ++u) { long long ii = i + u * st; if (ii < n16) { k[u] = __ldg(K + ii); v[u] = __ldg(V + ii); } else { k[u] = make_uint4(0,0,0,0); v[u] = k[u]; } }
    #pragma unroll
    for (int u = 0; u < U; ++u) acc ^= k[u].x ^ v[u].w;
  }
  if (acc == 0x12345678u) atomicAdd(sink, 1ull);
}
void run_gat(torch::Tensor pt, torch::Tensor K, torch::Tensor V, torch::Tensor sink, int64_t total, int64_t grid, int64_t blk, int64_t U) {
  auto s = at::cuda::getCurrentCUDAStream();
  #define L(u) gat<u><<<grid, blk, 0, s>>>(pt.data_ptr<int>(), 0, nullptr, 0, (const uint4*)K.data_ptr(), (const uint4*)V.data_ptr(), (unsigned long long*)sink.data_ptr(), total)
  if (U == 1) L(1); else if (U == 2) L(2); else if (U == 4) L(4); else L(8);
}
void run_seq(torch::Tensor K, torch::Tensor V, torch::Tensor sink, int64_t n16, int64_t grid, int64_t blk, int64_t U) {
  auto s = at::cuda::getCurrentCUDAStream();
  if (U == 1) seq<1><<<grid, blk, 0, s>>>((const uint4*)K.data_ptr(), (const uint4*)V.data_ptr(), (unsigned long long*)sink.data_ptr(), n16);
  else if (U == 4) seq<4><<<grid, blk, 0, s>>>((const uint4*)K.data_ptr(), (const uint4*)V.data_ptr(), (unsigned long long*)sink.data_ptr(), n16);
  else seq<8><<<grid, blk, 0, s>>>((const uint4*)K.data_ptr(), (const uint4*)V.data_ptr(), (unsigned long long*)sink.data_ptr(), n16);
}
"""
cpp = "void run_gat(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t);\n" \
      "void run_seq(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t);"
m = load_inline("n64_gather", cpp, cuda_sources=src.replace("#include <cstdint>", "#include <cstdint>\n#include <ATen/cuda/CUDAContext.h>"),
                functions=["run_gat", "run_seq"], extra_cuda_cflags=["-O3", "-arch=sm_90a"], verbose=False)

B = int(sys.argv[1]); ctx = int(sys.argv[2]) if len(sys.argv) > 2 else 80000
g = torch.Generator(device="cuda"); g.manual_seed(1)
spread = "agentic" if B > 1 else "fixed"
lens = reference._lengths(g, B, ctx, spread, "cuda")
maxp = reference._longest(ctx, spread) + 1
pool = B * maxp
perm = torch.randperm(pool, generator=g, device="cuda").to(torch.int32)
keys = perm[: int(lens.sum())].contiguous()  # the same random-page gather as the page table, flattened
total = keys.numel()
K = torch.empty((pool, 128), dtype=torch.uint8, device="cuda").random_()
V = torch.empty((pool, 128), dtype=torch.uint8, device="cuda").random_()
sink = torch.zeros(1, dtype=torch.int64, device="cuda")
mb = total * 260 / 1e6
e0 = bench.empty_cost()
print(f"B{B} keys {total} bytes {mb:.1f} MB (pool {pool})", flush=True)
for U in (2, 4, 8):
    for grid, blk in ((132 * 8, 256), (132 * 16, 256), (132 * 4, 512), (132 * 32, 128)):
        t = bench.time_graph(lambda: m.run_gat(keys, K, V, sink, total, grid, blk, U)) - e0
        print(f"  gather U{U} grid{grid}x{blk}: {t:.1f} us  {mb / t:.3f} TB/s", flush=True)
n16 = total * 8
for U in (4, 8):
    for grid, blk in ((132 * 8, 256), (132 * 16, 256)):
        t = bench.time_graph(lambda: m.run_seq(K, V, sink, n16, grid, blk, U)) - e0
        print(f"  seq U{U} grid{grid}x{blk}: {t:.1f} us  {total * 256 / 1e6 / t:.3f} TB/s", flush=True)
