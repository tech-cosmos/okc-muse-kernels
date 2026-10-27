# nvjet GEMM -> successor (pre-wait W stream, griddepcontrol.wait, then reads the GEMM output). Edge: default / programmatic port 1 / port 2.
import torch, os, sys, fcntl, importlib.util
spec = importlib.util.spec_from_file_location("pn", "/tmp/pj70/pdl_nvjet_lib.py"); 
from torch.utils.cpp_extension import load_inline
src = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <c10/cuda/CUDAStream.h>
__global__ void succ(const float4* w, long nw4, const __nv_bfloat16* o, long no, float* out, int pre) {
  float a = 0.f;
  long per = nw4 / gridDim.x, b0 = blockIdx.x * per;
  if (pre) for (long i = b0 + threadIdx.x; i < b0 + per; i += blockDim.x) { float4 v = __ldg(w + i); a += v.x + v.w; }
  asm volatile("griddepcontrol.wait;" ::: "memory");
  if (!pre) for (long i = b0 + threadIdx.x; i < b0 + per; i += blockDim.x) { float4 v = __ldg(w + i); a += v.x + v.w; }
  float s = 0.f;
  for (long i = blockIdx.x * blockDim.x + threadIdx.x; i < no; i += (long)gridDim.x * blockDim.x) s += __bfloat162float(o[i]);
  for (int d = 16; d; d >>= 1) s += __shfl_xor_sync(~0u, s, d);
  if ((threadIdx.x & 31) == 0) atomicAdd(out, s); if (a == 12345.f) out[1] = a;
}
void succ_launch(torch::Tensor w, torch::Tensor o, torch::Tensor out, int64_t pre, int64_t grid) {
  auto s = c10::cuda::getCurrentCUDAStream();
  succ<<<grid, 512, 0, s>>>((const float4*)w.data_ptr(), w.numel() / 4, (const __nv_bfloat16*)o.data_ptr(), o.numel(), out.data_ptr<float>(), pre);
}
int64_t pdl_prev(int64_t port) {
  auto s = c10::cuda::getCurrentCUDAStream().stream();
  cudaStreamCaptureStatus st; unsigned long long id; cudaGraph_t g; const cudaGraphNode_t* deps; const cudaGraphEdgeData* ed; size_t n;
  if (cudaStreamGetCaptureInfo(s, &st, &id, &g, &deps, &ed, &n) != cudaSuccess || st != cudaStreamCaptureStatusActive || n != 1) return -1;
  cudaGraphNode_t b = deps[0];
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
  return done;
}
'''
lib = load_inline("pdl_succ70", cpp_sources="void succ_launch(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t); int64_t pdl_prev(int64_t);",
                  cuda_sources=src.replace('#include <cuda.h>', '#include <cuda.h>\n#include <cuda_bf16.h>'),
                  functions=["succ_launch", "pdl_prev"], extra_cuda_cflags=["-arch=sm_90a"], extra_ldflags=["-lcuda"])
M, K, N = [int(v) for v in (sys.argv[1] if len(sys.argv) > 1 else "128,8192,4096").split(",")]
WMB = float(os.environ.get("WMB", "8")); GRID = int(os.environ.get("GRID", "132"))
torch.manual_seed(0)
W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
X = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
W2 = torch.randn(int(WMB * (1 << 20)) // 4 // (GRID * 4) * GRID * 4, device="cuda")
O = torch.empty(M, N, device="cuda", dtype=torch.bfloat16); out = torch.zeros(2, device="cuda")
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
def body(port, pre):
    torch.mm(X, W.t(), out=O); out.zero_()
    lib.succ_launch(W2, O, out, pre, GRID)
    if port >= 0: assert lib.pdl_prev(port) == 1
for pre in (1, 0):
  for port in (-1, 1, 2):
    if pre == 0 and port != 2: continue
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2): body(-1, pre)
    torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): body(port, pre)
    bad = 0; ts = []
    for r in range(40):
        O.zero_(); flush.zero_(); torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        with open("/tmp/okc-gpu.lock", "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX); e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3)
        ref = O.float().sum().item(); bad += abs(out[0].item() - ref) > 1e-3 * (abs(ref) + O.float().abs().sum().item() * 1e-3)
    ts.sort()
    print(f"M{M} K{K} N{N} W2 {WMB} MB pre={pre} edge port {port}: median {ts[len(ts)//2]:.2f} us, bad {bad}/40")
