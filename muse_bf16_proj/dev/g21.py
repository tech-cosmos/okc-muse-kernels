# gemv21 (TMA-ring m=1 GEMV over gemv13's 12-bit packing) sweep: python dev/g21.py "WG,NS;..." [blocks ...]
# env SH=qkv,pool,... selects shapes; G13=0 skips the gemv13 reference timing. Checks vs FP64 + exact probe columns.
import os, sys, torch, torch.nn.functional as F, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torch.utils.cpp_extension import load_inline
import bench, reference
from g13 import pack13

cfgs = [tuple(map(int, c.split(","))) for c in sys.argv[1].split(";")]  # (wg, ng, ns)
grids = [int(g) for g in (sys.argv[2:] or ["132"])]
SH = [("qkv", 2304, 8192), ("pool", 4096, 8192), ("o", 8192, 1024), ("unp1k", 1024, 4096)]
only = os.environ.get("SH")
SH = [s for s in SH if not only or s[0] in only.split(",")]
d = os.path.dirname(os.path.abspath(__file__))
src = "".join("#define %s\n" % x for x in os.environ.get("DEFS", "").split() if x) + open(os.path.join(d, "gemv21.cu")).read()
src = src.replace("G21_CONFIGS", " ".join("G21_CASE(%d,%d,%d,%d)" % (k, w, g, s) for (w, g, s) in cfgs for k in sorted({k for _, _, k in SH})
                                          if k % (16 * 32 * w) == 0))
mod = load_inline(name="pj_g21_" + hashlib.md5(src.encode()).hexdigest()[:10],
                  cpp_sources="void g21_gemv(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, int64_t, int64_t, int64_t, int64_t, torch::Tensor, int64_t, torch::Tensor);",
                  cuda_sources=src, functions=["g21_gemv"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"])
g13 = None
if os.environ.get("G13", "1") == "1":
    s13 = open(os.path.join(d, "gemv13.cu")).read().replace("G13_CONFIGS", "G13_CASE(8192,2,4,1,1,256) G13_CASE(4096,1,4,1,1,256) G13_CASE(1024,1,4,1,1,256)")
    g13 = load_inline(name="pj_g13_" + hashlib.md5(s13.encode()).hexdigest()[:10],
                      cpp_sources="void g13_gemv(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);",
                      cuda_sources=s13, functions=["g13_gemv"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])
Wr = reference.generate_weights(reference.WEIGHTS_SEED)
names = {"qkv": "qkv_gate", "o": "o", "pool": "pooler", "unp1k": "unpooler"}
e = bench.empty_cost()
none = torch.empty(0, dtype=torch.int32, device="cuda")
for nm, n, k in SH:
    W = Wr[names[nm]][:n].contiguous()
    if W.shape[1] != k:
        W = W[:, :k].contiguous()
    sm, nib, hdr, scale = pack13(W)
    x = torch.randn(1, k, device="cuda").bfloat16()
    ref = x.double() @ W.double().T
    line = f"{nm:5s} n{n} k{k}: cublas {bench.time_graph(lambda: F.linear(x, W)) - e:6.2f}"
    if g13 is not None:
        u = 2 if k == 8192 else 1
        o13 = torch.empty(1, n, dtype=torch.bfloat16, device="cuda")
        line += f" g13 {bench.time_graph(lambda: g13.g13_gemv(sm, nib, hdr, x, o13, scale, u, 4, 1, 1, 132, 256)) - e:6.2f}"
    er = (hdr[:, :, 1] != 2.0 ** -15).nonzero()
    probe_cols = [0, 5] + ([int(hdr[er[0, 0], er[0, 1], 0].view(torch.int32))] if len(er) else [])
    for (w, g, s) in cfgs:
        if k % (16 * 32 * w):
            continue
        for b in grids:
            out = torch.empty(1, n, dtype=torch.bfloat16, device="cuda")
            qg = torch.zeros(4, dtype=torch.int32, device="cuda")
            tk = torch.zeros(1, dtype=torch.int64, device='cuda')
            mod.g21_gemv(sm, nib, hdr, x, out, scale, w, g, s, b, qg, n // 2, tk); torch.cuda.synchronize()
            if not os.environ.get("DEFS"): assert qg[0].item() == n // 2 and qg[1].item() == n - n // 2, qg
            err = ((out.double() - ref).norm() / ref.norm()).item()
            if os.environ.get("DEFS"): probe_cols = []; err = 0
            assert err < 5e-3, (err, nm, w, s, b)
            for col in probe_cols:
                xp = torch.zeros(1, k, device="cuda", dtype=torch.bfloat16); xp[0, col] = 32.0
                op = torch.empty(1, n, dtype=torch.bfloat16, device="cuda")
                mod.g21_gemv(sm, nib, hdr, xp, op, scale, w, g, s, b, none, 0, tk)
                exp = (W[:, col].double() * 32).bfloat16()
                assert torch.equal(op[0], exp), (nm, col, (op[0].double() - exp.double()).abs().max().item())
            dt = bench.time_graph(lambda: mod.g21_gemv(sm, nib, hdr, x, out, scale, w, g, s, b, none, 0, tk)) - e
            line += f" | W{w}G{g}S{s} g{b} {dt:6.2f}"
            assert tk.item() % (n + 4 * b) == 0, tk.item()
    print(line, flush=True)
