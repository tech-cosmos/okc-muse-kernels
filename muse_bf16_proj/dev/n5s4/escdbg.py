"""proj n5s4: timing-only bisection of g30 ESCM's escape cost (debug defines; outputs may be wrong). After a 3 us pred.
python dev/n5s4/escdbg.py M key S "VAR|VAR" (VAR = comma-separated defines for dev/n5s4/g30e_src.cu; 'OLD' = dev/g30_src.cu)"""
import sys, hashlib, torch
sys.path.insert(0, ".")
import bench, reference
from torch.utils.cpp_extension import load_inline
L = bench.submission
M, key, S = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
def build(v):
    src = open("dev/g30_src.cu" if v.startswith("OLD") else "dev/n5s4/g30e_src.cu").read()
    dfs = [f"-DG30_MTOK={M}", f"-DG30_S={S}", "-DG30_NW=16", "-DG30_NS=16"] + [f"-D{d}" for d in v.split(",")[v.startswith('OLD'):] if d]
    return load_inline(name="pjn5s4d_" + hashlib.sha256((src + " ".join(dfs)).encode()).hexdigest()[:10],
                       cpp_sources="void g30_gemm(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, int64_t);",
                       cuda_sources=src, functions=["g30_gemm"],
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", *dfs], extra_ldflags=["-lcuda"])
w = reference.generate_weights(reference.WEIGHTS_SEED)[key]; N, K = w.shape
code, sm, hdr, scale = L._compress(w)
P = L._pack30(code, sm)
xs = torch.randn(M, K, device="cuda").bfloat16(); x = xs.clone()
out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
pred = lambda: L._g19.g19_pred(xs, x, 3000)  # noqa: E731
ep = bench.time_graph(pred)
libs = {v: build(v) for v in sys.argv[4].split("|")}
res = {}
for r in range(3):
    for v, lib in libs.items():
        res.setdefault(v, []).append(bench.time_graph(lambda: (pred(), lib.g30_gemm(x, P, hdr, out, scale, 1))) - ep)
print(f"M{M} {key} S{S} after 3 us pred: " + "  ".join(f"[{v or 'escm'}] {min(t):.2f}" for v, t in res.items()), flush=True)
