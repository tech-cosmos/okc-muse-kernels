"""n18: router m128/m256 bmm(S) + rsum: plain launch vs PDL-launched rsum (griddepcontrol.wait first)."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from torch.utils.cpp_extension import load_inline
import bench, reference
src = r'''
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
template <bool PDL>
__global__ void rsum(const float4* __restrict__ ws, float4* __restrict__ out, int m, int S) {
    if (PDL) asm volatile("griddepcontrol.wait;\n" ::: "memory");
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m * 64) return;
    const int r = i >> 6, c = i & 63;
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int s = 0; s < S; ++s) {
        const float4 h = ws[((size_t)s * m + r) * 128 + c], l = ws[((size_t)s * m + r) * 128 + 64 + c];
        a.x += h.x + l.x; a.y += h.y + l.y; a.z += h.z + l.z; a.w += h.w + l.w;
    }
    out[i] = a;
}
void rs(torch::Tensor ws, torch::Tensor out, int64_t pdl, int64_t bs) {
    const int S = ws.size(0), m = ws.size(1);
    auto stream = c10::cuda::getCurrentCUDAStream();
    cudaLaunchConfig_t lc = {};
    lc.gridDim = dim3((m * 64 + bs - 1) / bs); lc.blockDim = dim3(bs); lc.stream = stream;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    lc.attrs = at; lc.numAttrs = pdl ? 1 : 0;
    const float4* a = reinterpret_cast<const float4*>(ws.data_ptr<float>());
    float4* o = reinterpret_cast<float4*>(out.data_ptr<float>());
    if (pdl) cudaLaunchKernelEx(&lc, rsum<true>, a, o, m, S); else cudaLaunchKernelEx(&lc, rsum<false>, a, o, m, S);
}
'''
mod = load_inline(name="n18rsp", cpp_sources="void rs(torch::Tensor, torch::Tensor, int64_t, int64_t);", cuda_sources=src,
                  functions=["rs"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])
W = reference.generate_weights(reference.WEIGHTS_SEED)["router"].cuda()
hi = W.to(torch.bfloat16); lo = (W - hi.float()).to(torch.bfloat16)
hlt = torch.cat([hi, lo], 0).t().contiguous()
e = bench.empty_cost()
for m in (128, 256):
    x = torch.randn(m, 8192, device="cuda").to(torch.bfloat16)
    ref = x.double() @ W.double().t()
    for S in (4, 8):
        out = torch.empty(m, 256, device="cuda")
        def f(pdl, bs=256):
            ws = torch.bmm(x.view(m, S, -1).transpose(0, 1), hlt.view(S, -1, 512), out_dtype=torch.float32)
            mod.rs(ws, out, pdl, bs)
        r = []
        for pdl, bs in ((0, 256), (1, 256), (1, 128), (0, 128)):
            out.fill_(7); f(pdl, bs); torch.cuda.synchronize()
            err = ((out.double() - ref).norm() / ref.norm()).item()
            ts = [bench.time_graph(lambda: f(pdl, bs), reps=200) - e for _ in range(3)]
            r.append(f"pdl{pdl} bs{bs} {min(ts):.2f}/{sorted(ts)[1]:.2f} (err {err:.0e})")
        print(f"m{m} S{S}: " + " | ".join(r), flush=True)
