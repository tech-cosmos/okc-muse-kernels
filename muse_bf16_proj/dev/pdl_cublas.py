import torch, sys
from torch.utils.cpp_extension import load_inline
src = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
__device__ __forceinline__ unsigned long long gt(){unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;":"=l"(t)); return t;}
__global__ void stamp0(unsigned long long* ts){ if(threadIdx.x==0 && blockIdx.x==0) ts[0]=gt(); }
__global__ void sec(unsigned long long* ts, const float* y, float* o){
  unsigned long long a = gt();
  asm volatile("griddepcontrol.wait;" ::: "memory");
  unsigned long long b = gt();
  if(threadIdx.x==0){ atomicMin(&ts[1], a); atomicMax(&ts[2], b); atomicMin(&ts[3], b);}
  if(threadIdx.x==0 && blockIdx.x==0) o[0] = __ldcg(y);
}
void run_stamp0(torch::Tensor ts){ stamp0<<<1,32,0,at::cuda::getCurrentCUDAStream()>>>((unsigned long long*)ts.data_ptr()); }
void run_sec(torch::Tensor ts, torch::Tensor y, torch::Tensor o, bool pdl, int blocks){
  cudaLaunchConfig_t cfg = {}; cfg.gridDim = blocks; cfg.blockDim = 128; cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = pdl;
  cfg.attrs = at; cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, sec, (unsigned long long*)ts.data_ptr(), (const float*)y.data_ptr(), (float*)o.data_ptr());
}
'''
m = load_inline("pdlcb", cpp_sources="void run_stamp0(torch::Tensor ts); void run_sec(torch::Tensor ts, torch::Tensor y, torch::Tensor o, bool pdl, int blocks);",
    cuda_sources=src.replace("#include <torch/extension.h>","#include <torch/extension.h>"), functions=["run_stamp0","run_sec"], extra_cuda_cflags=["-arch=sm_90a"], verbose=False)
import torch.nn.functional as F
flush = torch.empty(256<<20, dtype=torch.uint8, device="cuda")
for (M,K,N) in [(16,4096,2304*2),(32,4096,4096),(16,4096,1024),(32,8192,4096)]:
  x = torch.randn(M,K,device="cuda",dtype=torch.bfloat16); W = torch.randn(N,K,device="cuda",dtype=torch.bfloat16)
  y = torch.empty(M,N,device="cuda",dtype=torch.bfloat16); o = torch.empty(1,device="cuda")
  ts = torch.zeros(4,dtype=torch.int64,device="cuda")
  for pdl in (False, True):
    def body():
      m.run_stamp0(ts); torch.matmul(x, W.t(), out=y); m.run_sec(ts, y, o, pdl, 132)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
      body(); body()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): body()
    res=[]
    for r in range(30):
      flush.zero_(); ts[1]=2**62; ts[2]=0; ts[3]=2**62; torch.cuda.synchronize()
      g.replay(); torch.cuda.synchronize()
      t=ts.tolist(); res.append(((t[1]-t[0])/1e3,(t[3]-t[0])/1e3,(t[2]-t[0])/1e3))
    res.sort(key=lambda r:r[2]); r=res[len(res)//2]
    print(f"M{M} K{K} N{N} pdl={pdl}: sec start {r[0]:.2f} us, wait done first {r[1]:.2f} last {r[2]:.2f} (from stamp0)")
