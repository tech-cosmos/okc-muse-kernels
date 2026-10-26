import sys, torch, torch.nn.functional as F
sys.path.insert(0, "/home/daytona/r-proj/muse_bf16_proj")
from torch.utils.cpp_extension import load_inline
import bench
src = r'''
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
// Rewrite the capturing stream's current dependencies (the last captured kernel node) as programmatic edges from
// port `port` (1 programmatic, 2 launch completion): the next captured kernel may launch early and must
// griddepcontrol.wait before reading the upstream's results. Returns 0 if not capturing.
int64_t lc_edge(int64_t port) {
    cudaStream_t s = c10::cuda::getCurrentCUDAStream();
    cudaStreamCaptureStatus st; const cudaGraphNode_t* deps = nullptr; const cudaGraphEdgeData* ed = nullptr; size_t nd = 0;
    if (cudaStreamGetCaptureInfo(s, &st, nullptr, nullptr, &deps, &ed, &nd) != cudaSuccess || st != cudaStreamCaptureStatusActive) return 0;
    std::vector<cudaGraphNode_t> d(deps, deps + nd);
    std::vector<cudaGraphEdgeData> e(nd);
    for (size_t i = 0; i < nd; ++i) { e[i] = {}; cudaGraphNodeType t; cudaGraphNodeGetType(d[i], &t);
        if (t == cudaGraphNodeTypeKernel) { e[i].from_port = (unsigned char)port; e[i].type = cudaGraphDependencyTypeProgrammatic; } }
    cudaError_t r = cudaStreamUpdateCaptureDependencies(s, d.data(), e.data(), nd, cudaStreamSetCaptureDependencies);
    TORCH_CHECK(r == cudaSuccess, "lc_edge: ", cudaGetErrorString(r));
    return (int64_t)nd;
}
__global__ void dep(const __nv_bfloat16* x, float* y, int n) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __bfloat162float(x[i]) * 2.f;
}
void dep_launch(torch::Tensor x, torch::Tensor y, int64_t blocks, int64_t pdl) {
    cudaLaunchConfig_t c = {}; c.gridDim = blocks; c.blockDim = 128; c.stream = c10::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute a[1]; a[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; a[0].val.programmaticStreamSerializationAllowed = 1;
    c.attrs = a; c.numAttrs = pdl ? 1 : 0;
    cudaLaunchKernelEx(&c, dep, (const __nv_bfloat16*)x.data_ptr(), y.data_ptr<float>(), (int)y.numel());
}
'''
m = load_inline(name="pj_lc1", cpp_sources="int64_t lc_edge(int64_t); void dep_launch(torch::Tensor, torch::Tensor, int64_t, int64_t);",
                cuda_sources=src, functions=["lc_edge", "dep_launch"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])
W = torch.randn(2304, 8192, device="cuda").bfloat16()
e = bench.empty_cost()
for M in (16, 32, 128):
    x = torch.randn(M, 8192, device="cuda").bfloat16()
    out = torch.empty(M, 2304, device="cuda", dtype=torch.bfloat16)
    y = torch.empty(M * 2304, device="cuda")
    def run(mode):
        torch.mm(x, W.t(), out=out)
        if mode: m.lc_edge(mode)
        m.dep_launch(out, y, (M * 2304 + 127) // 128, 1 if mode else 0)
    res = []
    for mode in (0, 1, 2, 0, 2):
        run(mode); torch.cuda.synchronize()
        ok = torch.allclose(y, out.float().flatten() * 2)
        res.append(f"mode{mode} {bench.time_graph(lambda: run(mode)) - e:6.2f}{'' if ok else ' BAD'}")
    gemm = bench.time_graph(lambda: torch.mm(x, W.t(), out=out)) - e
    print(f"m{M}: gemm alone {gemm:6.2f} | " + " | ".join(res), flush=True)
