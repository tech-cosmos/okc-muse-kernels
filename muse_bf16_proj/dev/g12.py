# gemv12 sweep: python dev/g12.py "M,KCU,WG,PF;..." bk [bk ...]  (bk = blocks per k-chunk)   (H100, graph replay, cold L2); env SH=qkv,o,...
import os, sys, torch, torch.nn.functional as F, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.cpp_extension import load_inline
import bench
cfgs = [tuple(map(int, c.split(","))) for c in sys.argv[1].split(";")]
bks = [int(g) for g in (sys.argv[2:] or ["32"])]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemv12.cu")).read()
src = os.environ.get("PRE", "") + "\n" + src.replace("G12_CONFIGS", " ".join("G12_CASE(%d,%d,%d,%d)" % c for c in cfgs))
mod = load_inline(name="pj_g12_" + hashlib.md5(src.encode()).hexdigest()[:10], cpp_sources="void g12_gemm(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t);",
                  cuda_sources=src, functions=["g12_gemm"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"])
def pack(w, kc):
    n, k = w.shape
    return w.view(n // 8, 8, k // kc, kc // 32, 4, 8).permute(2, 0, 3, 1, 4, 5).contiguous()
torch.manual_seed(0)
SH = [("qkv", 2304, 8192, 2304), ("o", 8192, 1024, 8192), ("pool", 4096, 8192, 4096), ("unp1k", 8192, 4096, 1024), ("unp", 8192, 4096, 8192)]
only = os.environ.get("SH")
ws = torch.zeros(8 * 32 * 8192, device="cuda"); cnt = torch.zeros(1024, dtype=torch.int32, device="cuda")
e = bench.empty_cost()
for nm, nw, k, n in SH:
    if only and nm not in only.split(","): continue
    W = (torch.randn(nw, k, device="cuda") / k ** 0.5).bfloat16()
    packs = {}
    for M in sorted({c[0] for c in cfgs}):
        if nm == "unp1k" and M > 8: continue
        if nm == "unp" and M <= 8: continue
        x = torch.randn(M, k, device="cuda").bfloat16()
        ref = x.double() @ W[:n].double().T
        Wn = W[:n]
        tc = bench.time_graph(lambda: F.linear(x, Wn)) - e
        line = f"{nm:5s} m{M}: cublas {tc:6.2f}"
        for (m_, u, wg, pf) in cfgs:
            if m_ != M or k % (1024 * u): continue
            if u not in packs: packs[u] = pack(W, 1024 * u)
            for bk in bks:
                if (n // 8) % bk or (n // 8 // bk) * wg > 32: continue
                out = torch.empty(M, n, dtype=torch.bfloat16, device="cuda")
                mod.g12_gemm(packs[u], x, out, ws, cnt, nw // 8, u, wg, pf, bk); torch.cuda.synchronize()
                err = ((out.double() - ref).norm() / ref.norm()).item()
                assert err < 5e-3 and cnt.abs().max().item() == 0, (err, nm, m_, u, wg, pf, bk)
                dt = bench.time_graph(lambda: mod.g12_gemm(packs[u], x, out, ws, cnt, nw // 8, u, wg, pf, bk)) - e
                line += f" | KC{u}W{wg}P{pf} b{(k // (1024 * u)) * bk}x{(n // 8 // bk) * wg * 32} {dt:6.2f}"
        print(line, flush=True)
    del W, packs
