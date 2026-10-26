# gemv23 (m = 128/256 BF16 SS-wgmma GEMM) test + timing: python dev/g23.py "M,BW,NS[,OB[,CL[,GRID]]];..."  env SH=qkv,o,pool,unp
# H100 graph replay, cold L2; cuBLAS F.linear for reference. GRID default: SMs (OB=1 needs GRID = n / BW).
import os, sys, torch, torch.nn.functional as F, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.cpp_extension import load_inline
import bench, reference
d = os.path.dirname(os.path.abspath(__file__))
NSM = torch.cuda.get_device_properties(0).multi_processor_count
cfgs = []
for c in sys.argv[1].split(";"):
    v = list(map(int, c.split(",")))
    while len(v) < 6: v.append(0 if len(v) == 3 else 1 if len(v) == 4 else -1)
    cfgs.append(tuple(v))
src = open(os.path.join(d, "gemv23.cu")).read()
src = "".join("#define %s\n" % d_.replace("=", " ") for d_ in os.environ.get("DEFS", "").split(",") if d_) + src
src = src.replace("G23_CONFIGS", " ".join("G23_CASE(%d,%d,%d,%d,%d)" % c[:5] for c in sorted(set(cfgs))))
mod = load_inline(name="pj_g23_" + hashlib.md5(src.encode()).hexdigest()[:10],
                  cpp_sources="void g23_gemm(torch::Tensor, int64_t, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);",
                  cuda_sources=src, functions=["g23_gemm"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo", "-Xptxas=-v"], verbose=bool(os.environ.get("VERB")))
SH = [("qkv", "qkv_gate", 2304), ("o", "o", 8192), ("pool", "pooler", 4096), ("unp", "unpooler", 8192)]
only = os.environ.get("SH")
Wr = reference.generate_weights(reference.WEIGHTS_SEED)
e = bench.empty_cost()


def pack23(w, bw):
    """[tile][k step][bw rows][64 k] with the SW128 16-B chunk swizzle (chunk c of row r at c ^ (r % 8))."""
    n, k = w.shape
    p = w.view(n // bw, bw, k // 64, 8, 8).permute(0, 2, 1, 3, 4)
    r = torch.arange(bw, device=w.device) % 8
    idx = (torch.arange(8, device=w.device)[None, :] ^ r[:, None])
    p = torch.gather(p, 3, idx.view(1, 1, bw, 8, 1).expand(p.shape).contiguous())
    return p.contiguous()

for nm, layer, n in SH:
    if only and nm not in only.split(","): continue
    W = Wr[layer][:n].contiguous()
    k = W.shape[1]
    for M in sorted({c[0] for c in cfgs}):
        x = torch.randn(M, k, device="cuda").bfloat16()
        ref = x.double() @ W.double().T
        tc = bench.time_graph(lambda: F.linear(x, W)) - e
        line = f"{nm:5s} m{M}: cublas {tc:6.2f}"
        for (m_, bw, ns, ob, cl, gr) in cfgs:
            if m_ != M: continue
            WP = pack23(W, bw)
            G = gr if gr > 0 else (n // bw * cl if (ob or cl > 1) else min(NSM, n // bw * (k // 64)))
            if ob and G != n // bw * cl: line += f" | {bw},{ns},ob G{G} n/a"; continue
            out = torch.empty(M, n, device="cuda", dtype=torch.bfloat16 if ob else torch.float32)
            if not os.environ.get("SKIPCHK"):
                for rep in range(2):
                    out.fill_(1.0)
                    mod.g23_gemm(WP, n, x, out, bw, ns, cl, G, 0, 0); torch.cuda.synchronize()
                    err = ((out.double() - ref).norm() / ref.norm()).item()
                    assert err < (4e-3 if ob else 1e-5), (err, nm, m_, bw, ns, G)
            dt = bench.time_graph(lambda: mod.g23_gemm(WP, n, x, out, bw, ns, cl, G, 0, int(os.environ.get("NPRE", 99)))) - e
            line += f" | {bw},{ns}{',ob' if ob else ''}{f',c{cl}' if cl > 1 else ''} G{G} {dt:6.2f}"
        print(line, flush=True)
    del W
