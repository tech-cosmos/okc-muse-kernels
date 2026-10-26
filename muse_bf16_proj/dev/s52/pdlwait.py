# s52: latency of griddepcontrol.wait vs a flag (counter) handoff at a PDL boundary.
# pred: NB blocks spin SPIN ns, write x, (flag mode) fence + atomicAdd ctr; record end time. sec: launched as PDL secondary
# (pred triggers at its start), records t after wait (mode 0: griddepcontrol.wait; mode 1: spin ld.acquire ctr == epoch*NB).
import torch, os, sys, fcntl, statistics
from torch.utils.cpp_extension import load_inline
src = r'''
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
__device__ __forceinline__ unsigned long long gt() { unsigned long long v; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(v)); return v; }
__global__ void pred(float* x, int* ctr, unsigned long long* tend, long spin, int mode) {
    asm volatile("griddepcontrol.launch_dependents;");
    unsigned long long t0 = gt();
    while (gt() - t0 < (unsigned long long)spin) {}
    x[blockIdx.x * blockDim.x + threadIdx.x] = (float)t0;
    __syncthreads();
    if (threadIdx.x == 0) {
        tend[blockIdx.x] = gt();
        if (mode == 1) { __threadfence(); atomicAdd(ctr, 1); }
    }
}
__global__ void sec(const float* x, int* ctr, unsigned long long* tw, float* y, int target, int mode) {
    if (mode == 0) asm volatile("griddepcontrol.wait;" ::: "memory");
    else {
        if (threadIdx.x == 0) { int v; do { asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(ctr)); } while (v < target); }
        __syncthreads();
    }
    float v = __ldcg(x + (blockIdx.x * blockDim.x + threadIdx.x) % (132*256));
    if (threadIdx.x == 0) tw[blockIdx.x] = gt();
    y[blockIdx.x * blockDim.x + threadIdx.x] = v;
}
void run(torch::Tensor x, torch::Tensor ctr, torch::Tensor tend, torch::Tensor tw, torch::Tensor y, long spin, int mode, int target, int nbs) {
    auto st = c10::cuda::getCurrentCUDAStream();
    pred<<<132, 256, 0, st>>>(x.data_ptr<float>(), ctr.data_ptr<int>(), (unsigned long long*)tend.data_ptr<long>(), spin, mode);
    cudaLaunchConfig_t cfg = {}; cfg.gridDim = nbs; cfg.blockDim = 256; cfg.stream = st;
    cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at; cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, sec, (const float*)x.data_ptr<float>(), ctr.data_ptr<int>(), (unsigned long long*)tw.data_ptr<long>(), y.data_ptr<float>(), target, mode);
}
'''
m = load_inline('s52pdl', 'void run(torch::Tensor x, torch::Tensor ctr, torch::Tensor tend, torch::Tensor tw, torch::Tensor y, long spin, int mode, int target, int nbs);', cuda_sources=src, functions=['run'], extra_cuda_cflags=['-O3', '-arch=sm_90a'], verbose=False)
d = 'cuda'
x = torch.zeros(132 * 256, device=d); ctr = torch.zeros(1, dtype=torch.int32, device=d)
tend = torch.zeros(132, dtype=torch.long, device=d); y = torch.zeros(132 * 256, device=d)
NBS = int(os.environ.get('NBS', '132'))
tw = torch.zeros(NBS, dtype=torch.long, device=d)
res = {0: [], 1: []}
lk = open('/tmp/okc-gpu.lock', 'w')
for rep in range(40):
    for mode in (0, 1):
        ctr.zero_(); torch.cuda.synchronize()
        fcntl.flock(lk, fcntl.LOCK_EX)
        m.run(x, ctr, tend, tw, y, 3000, mode, 132, NBS); torch.cuda.synchronize()
        fcntl.flock(lk, fcntl.LOCK_UN)
        if rep < 3: continue
        e = tend.max().item(); w = tw.cpu()
        res[mode].append(((w.min().item() - e) / 1e3, (statistics.median(w.tolist()) - e) / 1e3, (w.max().item() - e) / 1e3))
for mode in (0, 1):
    a = res[mode]
    print('mode', ['gdc.wait', 'flag'][mode], 'us after last pred end: min/med/max of blocks, median over reps:',
          ['%.2f' % statistics.median(v[i] for v in a) for i in range(3)])
