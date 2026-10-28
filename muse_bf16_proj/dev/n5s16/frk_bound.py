"""n5s16: bound for layer NEXT 3 'pr128' (T=128 one-launch pooler + router hi/lo on g30k stream-K F32 slabs, <= 132 CTAs).
Cold-L2 graph us (layer bench.time_graph) of: cuBLAS FR torch.mm [pooler; hi; lo] FP32 out (the layer's T=128 path),
cuBLAS pooler only, and g30k (r-layer g30_src.cu, -DG30K_ON -DG30K_F32=1, MTOK 128) on N rows = pooler (+ 512 extra 12-bit rows
standing in for hi/lo: lower bound, raw-BF16 lo rows cost more bytes). usage: python frk_bound.py [G,..] env NS=7 NW=10"""
import os, sys, hashlib
LAY = os.path.expanduser("~/r-proj/muse_decode_layer")
sys.path.insert(0, LAY)
import torch
from torch.utils.cpp_extension import load_inline
import bench, reference
L = bench.load_sub(os.path.join(LAY, "submission.py"))
NS, NW = int(os.environ.get("NS", 7)), int(os.environ.get("NW", 10))
src = open(os.environ.get("SRC") or os.path.join(LAY, "g30_src.cu")).read()
dfs = ["-DG30K_ON", "-DG30K_F32=1", "-DG30_MTOK=128", f"-DG30_NW={NW}", f"-DG30_NS={NS}"] + os.environ.get("DEFS", "").split()
lib = load_inline(name="n5s16_g30k_" + hashlib.sha256((src + " ".join(dfs)).encode()).hexdigest()[:10],
                  cpp_sources="void g30k_gemm(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, int64_t, "
                  "torch::Tensor, torch::Tensor, int64_t);", cuda_sources=src, functions=["g30k_gemm"],
                  extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"] + dfs, extra_ldflags=["-lcuda"])
w = reference.generate_weights(reference.WEIGHTS_SEED)
p = w["pooler"]
r = w["router"].float() if "router" in w else None
M = 128
torch.manual_seed(1)
x = torch.randn(M, p.shape[1], device="cuda").bfloat16()
e0 = bench.empty_cost()
hi = r.bfloat16(); lo = (r - hi.float()).bfloat16()
fr = torch.cat([p, hi, lo]).contiguous()
print(f"cuBLAS FR [{fr.shape[0]}] fp32 out {bench.time_graph(lambda: torch.mm(x, fr.t(), out_dtype=torch.float32)) - e0:.2f} us")
print(f"cuBLAS pooler bf16 out {bench.time_graph(lambda: torch.mm(x, p.t())) - e0:.2f} us")
Gs = [int(g) for g in (sys.argv[1] if len(sys.argv) > 1 else "128,132").split(",")]
for extra in (0, 512):
    ww = torch.cat([p, p[:extra]]) if extra else p
    code, sm, hdr, scale = L._compress(ww)
    wp = L._pack30(code, sm)
    N = ww.shape[0]
    ref = (x.float() @ ww.float().t())
    for G in Gs:
        sl = torch.zeros(4, M, N, device="cuda")
        dummy = torch.empty(0, device="cuda")
        tk = torch.zeros(1, dtype=torch.int32, device="cuda")
        fn = lambda: lib.g30k_gemm(x, wp, hdr, sl, scale, 0, dummy, tk, G)  # noqa: E731
        fn(); torch.cuda.synchronize()
        err = ((sl.sum(0) - ref).norm() / ref.norm()).item()
        t = bench.time_graph(fn) - e0
        print(f"g30k F32 NS{NS} NW{NW} N={N} G={G}: {t:.2f} us err {err:.1e}", flush=True)
