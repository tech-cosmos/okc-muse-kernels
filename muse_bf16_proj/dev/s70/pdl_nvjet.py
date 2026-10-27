# Does cuBLAS's nvjet GEMM wait (griddepcontrol.wait) when launched as a PDL secondary via a programmatic capture edge?
import torch, os, sys, fcntl
from torch.utils.cpp_extension import load_inline
src = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <c10/cuda/CUDAStream.h>
__global__ void spinw(const __nv_bfloat16* src, __nv_bfloat16* dst, long n, long ns) {
  asm volatile("griddepcontrol.launch_dependents;");
  unsigned long long t0, t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); } while (t - t0 < (unsigned long long)ns);
  for (long i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) dst[i] = src[i];
}
void spin_write(torch::Tensor src, torch::Tensor dst, int64_t ns) {
  auto s = c10::cuda::getCurrentCUDAStream();
  spinw<<<132, 256, 0, s>>>((const __nv_bfloat16*)src.data_ptr(), (__nv_bfloat16*)dst.data_ptr(), src.numel(), ns);
}
// make the next captured node depend programmatically on the current capture deps
int64_t pdl_next(int64_t port) {
  auto s = c10::cuda::getCurrentCUDAStream().stream();
  cudaStreamCaptureStatus st; unsigned long long id; cudaGraph_t g; const cudaGraphNode_t* deps; const cudaGraphEdgeData* ed; size_t n;
  if (cudaStreamGetCaptureInfo(s, &st, &id, &g, &deps, &ed, &n) != cudaSuccess || st != cudaStreamCaptureStatusActive) return -1;
  std::vector<cudaGraphNode_t> nd(deps, deps + n); std::vector<cudaGraphEdgeData> e(n);
  for (size_t i = 0; i < n; i++) { memset(&e[i], 0, sizeof(e[i])); e[i].from_port = (unsigned char)port; e[i].type = cudaGraphDependencyTypeProgrammatic; }
  auto r = cudaStreamUpdateCaptureDependencies(s, nd.data(), e.data(), n, cudaStreamSetCaptureDependencies);
  return r == cudaSuccess ? (int64_t)n : -(int64_t)r;
}

// after the GEMM node was captured: turn the edges (kernel pred -> last captured kernel node) into programmatic edges
int64_t pdl_prev(int64_t port) {
  auto s = c10::cuda::getCurrentCUDAStream().stream();
  cudaStreamCaptureStatus st; unsigned long long id; cudaGraph_t g; const cudaGraphNode_t* deps; const cudaGraphEdgeData* ed; size_t n;
  if (cudaStreamGetCaptureInfo(s, &st, &id, &g, &deps, &ed, &n) != cudaSuccess || st != cudaStreamCaptureStatusActive || n != 1) return -1;
  cudaGraphNode_t b = deps[0]; cudaGraphNodeType ty; cudaGraphNodeGetType(b, &ty); if (ty != cudaGraphNodeTypeKernel) return -2;
  cudaKernelNodeParams kp; cudaGraphKernelNodeGetParams(b, &kp); const char* nm = nullptr; if (cuFuncGetName(&nm, (CUfunction)kp.func) != CUDA_SUCCESS) nm = nullptr; cudaGetLastError();
  size_t np = 0; cudaGraphNodeGetDependencies(b, nullptr, nullptr, &np); if (np == 0) return -3;
  std::vector<cudaGraphNode_t> pr(np); std::vector<cudaGraphEdgeData> pe(np);
  cudaGraphNodeGetDependencies(b, pr.data(), pe.data(), &np);
  int64_t done = 0;
  for (size_t i = 0; i < np; i++) {
    cudaGraphNodeType pt; cudaGraphNodeGetType(pr[i], &pt); if (pt != cudaGraphNodeTypeKernel) continue;
    if (cudaGraphRemoveDependencies(g, &pr[i], &b, &pe[i], 1) != cudaSuccess) return -4;
    cudaGraphEdgeData e; memset(&e, 0, sizeof(e)); e.from_port = (unsigned char)port; e.type = cudaGraphDependencyTypeProgrammatic;
    auto r = cudaGraphAddDependencies(g, &pr[i], &b, &e, 1); if (r != cudaSuccess) return -100 - (int64_t)r;
    done++;
  }
  printf("pdl_prev: node %s, %ld programmatic edges\n", nm ? nm : "?", (long)done);
  return done;
}
'''
lib = load_inline("pdl_nvjet70", cpp_sources="void spin_write(torch::Tensor, torch::Tensor, int64_t); int64_t pdl_next(int64_t); int64_t pdl_prev(int64_t);",
                  cuda_sources=src.replace('#include <torch/extension.h>', '#include <torch/extension.h>\n#include <cuda_bf16.h>'),
                  functions=["spin_write", "pdl_next", "pdl_prev"], extra_cuda_cflags=["-arch=sm_90a"], extra_ldflags=["-lcuda"], verbose=False)
M, K, N = [int(v) for v in (sys.argv[1] if len(sys.argv) > 1 else "128,8192,4096").split(",")]
port = int(os.environ.get("PORT", "1")); ns = int(os.environ.get("NS", "20000"))
torch.manual_seed(0)
W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
xt = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
X = torch.zeros(M, K, device="cuda", dtype=torch.bfloat16)
ref = (xt.float() @ W.float().t())
O = torch.empty(M, N, device='cuda', dtype=torch.bfloat16)
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
def body(pdl):
    lib.spin_write(xt, X, ns)
    torch.mm(X, W.t(), out=O)
    if pdl: assert lib.pdl_prev(port) > 0
    return O
res = {}
for pdl in (0, 1):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2): body(0)
    torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): out = body(pdl)
    bad = 0; ts = []
    for r in range(30):
        X.zero_(); flush.zero_(); torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        with open("/tmp/okc-gpu.lock", "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX); e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3)
        err = (out.float() - ref).abs().max().item(); bad += err > 0.05 * ref.abs().max().item()
    ts.sort(); res[pdl] = ts[len(ts) // 2]
    print(f"M{M} K{K} N{N} pdl={pdl} port={port} spin={ns/1e3:.0f}us: median {ts[len(ts)//2]:.2f} us, bad {bad}/30")
print("saved", res[0] - res[1])
