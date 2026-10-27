"""s105: H100 bound for a one-launch pooler+router g30 split-K S2 slab GEMM at T=128 (s102 g30_slab = old g30_src @1d405c8
+ dev/s102_g30sp.patch, file given as argv[1]): time vs rows (extra rows = pooler copies; raw-BF16 lo rows ~ 1.33x
12-bit rows) vs cuBLAS FR torch.mm [pooler; hi; lo] fp32 out."""
import os, sys, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.cpp_extension import load_inline
import bench, reference
L = bench.load_sub("submission.py")
src = open(sys.argv[1]).read()
lib = load_inline(name="s105_g30slab_" + hashlib.sha256(src.encode()).hexdigest()[:10],
                  cpp_sources="void g30_slab(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, int64_t);",
                  cuda_sources=src, functions=["g30_slab"],
                  extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], extra_ldflags=["-lcuda"])
p = reference.generate_weights(0)["pooler"]
M = 128
x = torch.randn(M, p.shape[1], device="cuda").bfloat16()
e0 = bench.empty_cost()
fr = torch.cat([p, p[:512]])
print(f"cuBLAS FR [{fr.shape[0]}] fp32 out {bench.time_graph(lambda: torch.mm(x, fr.t(), out_dtype=torch.float32)) - e0:.2f} us")
for S in (2, 3):
    for extra in (0, 256, 512, 640, 768):
        w = torch.cat([p, p[:extra]]) if extra else p
        code, sm, hdr, scale = L._compress(w)
        wp = L._pack30(code, sm)
        sl = torch.empty(S, M, w.shape[0], device="cuda")
        t = bench.time_graph(lambda: lib.g30_slab(x, wp, hdr, sl, scale, 0)) - e0
        print(f"g30_slab S{S} N={w.shape[0]} (+{extra}) {t:.2f} us")
