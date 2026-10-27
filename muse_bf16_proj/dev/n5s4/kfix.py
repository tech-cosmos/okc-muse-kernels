"""proj n5s4: g30 fixed-cost decomposition. Builds dev/g30_src.cu with extra defines per variant and times qkv (N 2304)
at K = 768 / 2048 / 8192 (m=16, S3, 16/16 rings), 1-kernel graph after a cold flush, minus an empty-kernel graph.
Run from muse_bf16_proj/: python dev/n5s4/kfix.py "VAR1|VAR2 ..." (each VAR = comma-separated -D defines, '' = base)."""
import sys, hashlib, torch
sys.path.insert(0, ".")
import bench, reference
from torch.utils.cpp_extension import load_inline
L = bench.submission
SRC = open("dev/g30_src.cu").read()
M, S = 16, 3
def build(v):
    dfs = [f"-DG30_MTOK={M}", f"-DG30_S={S}", "-DG30_NW=16", "-DG30_NS=16"] + [f"-D{d}" for d in v.split(",") if d]
    return load_inline(name="pjn5s4_" + hashlib.sha256((SRC + " ".join(dfs)).encode()).hexdigest()[:10],
                       cpp_sources="void g30_gemm(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, int64_t);",
                       cuda_sources=SRC, functions=["g30_gemm"],
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", *dfs], extra_ldflags=["-lcuda"])
W = reference.generate_weights(reference.WEIGHTS_SEED)["qkv_gate"]
o = torch.zeros(1, device="cuda")
e0 = bench.time_graph(lambda: o.zero_())
data = []
for K in (768, 2048, 8192):
    w = W[:, :K].contiguous()
    code, sm, hdr, scale = L._compress(w)
    data.append((K, L._pack30(code, sm), hdr, scale, torch.randn(M, K, device="cuda").bfloat16(), w))
for v in sys.argv[1].split("|"):
    lib = build(v)
    res = []
    for K, P, hdr, scale, x, w in data:
        out = torch.empty(M, W.shape[0], device="cuda", dtype=torch.bfloat16)
        lib.g30_gemm(x, P, hdr, out, scale, 0)
        err = ((out.float() - (x.float() @ w.float().t())).norm() / (x.float() @ w.float().t()).norm()).item()
        t = min(bench.time_graph(lambda: lib.g30_gemm(x, P, hdr, out, scale, 0)) for _ in range(2)) - e0
        res.append(f"K{K} {t:.2f}{'' if err < 5e-3 else '(wrong)'}")
    print(f"[{v or 'base'}] " + " ".join(res), flush=True)
