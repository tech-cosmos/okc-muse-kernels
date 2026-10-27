# s71 g13t sweep (H100): python dev/s71/g13t.py "NS,NV,blocks;..."  env SH=qkv,pool,unp. Exactness vs gemv13 (bitwise),
# FP64 error, escape-column probes; times vs gemv13's main configs (bench.time_graph, cold L2, empty graph subtracted).
import os, sys, torch, hashlib
D = os.path.dirname(os.path.abspath(__file__)); P = os.path.dirname(os.path.dirname(D))
sys.path.insert(0, P); sys.path.insert(0, os.path.dirname(D))
from torch.utils.cpp_extension import load_inline
import bench, reference
import submission as _sub
cfgs = [tuple(map(int, c.split(","))) for c in sys.argv[1].split(";")]
src = "".join("#define %s\n" % d_ for d_ in os.environ.get("DEFS", "").split(",") if d_) + open(os.path.join(D, "gemv13t.cu")).read()
mod = load_inline(name="pj_g13t_" + hashlib.md5(src.encode()).hexdigest()[:10], cpp_sources="void g13t_gemv(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, int64_t, int64_t, int64_t);",
                  cuda_sources=src, functions=["g13t_gemv"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"])
g13src = open(os.path.join(os.path.dirname(D), "gemv13.cu")).read().replace("G13_CONFIGS", "G13_CASE(8192,1,2,1,1,512) G13_CASE(8192,2,4,1,1,256)")
g13 = load_inline(name="pj_g13r_" + hashlib.md5(g13src.encode()).hexdigest()[:10], cpp_sources="void g13_gemv(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);",
                  cuda_sources=g13src, functions=["g13_gemv"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])
Wr = reference.generate_weights(reference.WEIGHTS_SEED)
SH = [("qkv", "qkv_gate", 2304), ("pool", "pooler", 4096)]
only = os.environ.get("SH"); SH = [s for s in SH if not only or s[0] in only.split(",")]
e = bench.empty_cost()
for nm, key, n in SH:
    W = Wr[key][:n].contiguous(); assert W.shape[1] == 8192
    sm, nib, hdr, scale = _sub._pack13(W)
    x = torch.randn(1, 8192, device="cuda").bfloat16()
    ref = x.double() @ W.double().T
    o13 = torch.empty(1, n, dtype=torch.bfloat16, device="cuda")
    g13.g13_gemv(sm, nib, hdr, x, o13, scale, 1, 2, 1, 1, 132, 512)
    line = f"{nm:5s} g13 U1V2T512 {bench.time_graph(lambda: g13.g13_gemv(sm, nib, hdr, x, o13, scale, 1, 2, 1, 1, 132, 512)) - e:6.2f}"
    line += f" U2V4 {bench.time_graph(lambda: g13.g13_gemv(sm, nib, hdr, x, o13, scale, 2, 4, 1, 1, 132, 256)) - e:6.2f}"
    er = (hdr[:, :, 1] != 2.0 ** -15).nonzero()
    probe_cols = [0, 5] + ([int(hdr[er[0, 0], er[0, 1], 0].view(torch.int32))] if len(er) else [])
    for ns, nv, b in cfgs:
        out = torch.zeros(1, n, dtype=torch.bfloat16, device="cuda")
        mod.g13t_gemv(sm, nib, hdr, x, out, scale, ns, nv, b); torch.cuda.synchronize()
        err = ((out.double() - ref).norm() / ref.norm()).item()
        assert err < 5e-3 or os.environ.get("DEFS"), (err, nm, ns, nv, b)
        if os.environ.get("DEFS"): dt = bench.time_graph(lambda: mod.g13t_gemv(sm, nib, hdr, x, out, scale, ns, nv, b)) - e; line += f" | PROBE t{ns},{nv},g{b} {dt:6.2f}"; continue
        assert torch.equal(out, o13), (nm, ns, nv, b, (out.float() - o13.float()).abs().max().item())
        for col in probe_cols:
            xp = torch.zeros(1, 8192, device="cuda", dtype=torch.bfloat16); xp[0, col] = 32.0
            op = torch.empty(1, n, dtype=torch.bfloat16, device="cuda")
            mod.g13t_gemv(sm, nib, hdr, xp, op, scale, ns, nv, b)
            assert torch.equal(op[0], (W[:, col].double() * 32).bfloat16()), (nm, col)
        dt = bench.time_graph(lambda: mod.g13t_gemv(sm, nib, hdr, x, out, scale, ns, nv, b)) - e
        line += f" | t{ns},{nv},g{b} {dt:6.2f}"
    print(line, flush=True)
