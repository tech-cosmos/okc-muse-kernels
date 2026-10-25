import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.cpp_extension import load_inline
import bench
src = r'''
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
template<int U>
__global__ void rd(const uint4* __restrict__ p, long n16, float* out) {
    uint4 acc = make_uint4(0,0,0,0);
    long i = (long)blockIdx.x * blockDim.x * U + threadIdx.x;
    const long stride = (long)gridDim.x * blockDim.x * U;
    for (; i < n16; i += stride) {
        uint4 v[U];
#pragma unroll
        for (int u = 0; u < U; ++u) { long j = i + (long)u * blockDim.x; v[u] = j < n16 ? __ldcs(p + j) : make_uint4(0,0,0,0); }
#pragma unroll
        for (int u = 0; u < U; ++u) { acc.x ^= v[u].x; acc.y ^= v[u].y; acc.z ^= v[u].z; acc.w ^= v[u].w; }
    }
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x12345678) out[0] = 1.f;
}
template<int U>
__global__ void rd2(const uint4* __restrict__ p, long n16, float* out, int mode) {
    // n16 split into 32-lane 512-B units; each warp takes a contiguous run (mode 0) or units strided by nwarps (mode 1)
    const int lane = threadIdx.x & 31; const long gw = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const long nw = ((long)gridDim.x * blockDim.x) >> 5, T = n16 / 32;
    const long t0 = gw * T / nw, t1 = (gw + 1) * T / nw;
    uint4 acc = make_uint4(0,0,0,0);
    for (long t = t0; t < t1; t += U) {
        uint4 v[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            long tt = t + u; if (tt >= t1) tt = t1 - 1;
            long phys = mode == 0 ? tt : ((tt - t0) * nw + gw);
            if (phys >= T) phys = T - 1;
            asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v[u].x), "=r"(v[u].y), "=r"(v[u].z), "=r"(v[u].w) : "l"(p + phys * 32 + lane));
        }
#pragma unroll
        for (int u = 0; u < U; ++u) { acc.x ^= v[u].x; acc.y ^= v[u].y; acc.z ^= v[u].z; acc.w ^= v[u].w; }
    }
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x12345678) out[0] = 1.f;
}
void rdk2(torch::Tensor t, torch::Tensor o, int64_t blocks, int64_t threads, int64_t u, int64_t mode) {
    long n16 = t.numel() * t.element_size() / 16;
    auto s = c10::cuda::getCurrentCUDAStream();
    if (u == 4) rd2<4><<<blocks, threads, 0, s>>>((const uint4*)t.data_ptr(), n16, o.data_ptr<float>(), mode);
    else if (u == 8) rd2<8><<<blocks, threads, 0, s>>>((const uint4*)t.data_ptr(), n16, o.data_ptr<float>(), mode);
    else rd2<16><<<blocks, threads, 0, s>>>((const uint4*)t.data_ptr(), n16, o.data_ptr<float>(), mode);
}
void rdk(torch::Tensor t, torch::Tensor o, int64_t blocks, int64_t threads, int64_t u) {
    long n16 = t.numel() * t.element_size() / 16;
    auto s = c10::cuda::getCurrentCUDAStream();
    if (u == 4) rd<4><<<blocks, threads, 0, s>>>((const uint4*)t.data_ptr(), n16, o.data_ptr<float>());
    else rd<8><<<blocks, threads, 0, s>>>((const uint4*)t.data_ptr(), n16, o.data_ptr<float>());
}
'''
m = load_inline(name="pj_rd2", cpp_sources="void rdk(torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t); void rdk2(torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t);", cuda_sources=src, functions=["rdk", "rdk2"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])
o = torch.zeros(1, device="cuda")
for mb in (8, 16.7, 37.7, 67):
    t = torch.empty(int(mb * 1e6) // 4096 * 4096, dtype=torch.uint8, device="cuda")
    dt = bench.time_graph(lambda: m.rdk(t, o, 1056, 256, 4)) - bench.empty_cost()
    print(f"{mb:5.1f} MB gridstride: {dt:6.2f} us {mb/dt:5.2f} TB/s", flush=True)
    for blocks, threads, u in ((264, 256, 8), (132, 512, 8), (264, 256, 4), (264, 512, 8), (132, 256, 16), (528, 256, 4)):
        for mode in (0, 1):
            dt = bench.time_graph(lambda: m.rdk2(t, o, blocks, threads, u, mode)) - bench.empty_cost()
            print(f"{mb:5.1f} MB b{blocks} t{threads} u{u} mode{mode}: {dt:6.2f} us {mb/dt:5.2f} TB/s", flush=True)
