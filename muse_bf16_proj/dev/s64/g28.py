# s64 gemv28 sweep: python dev/s64/g28.py "M,BN,NS,ND;..."  env SH=qkv,o,pool,unp  GRID  PRED (ns, <0 none)  NPRE
# Times = graph(zero out, pred, gemm) - graph(zero out, pred), cold L2. Check vs x @ W'^T (escapes = 2^-15 * scale).
import os, sys, torch, hashlib, torch.nn.functional as F
D = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.dirname(os.path.dirname(D))
sys.path.insert(0, P2)
from torch.utils.cpp_extension import load_inline
import bench, reference
_t = open(os.path.join(os.path.dirname(D), "sub_template.py")).read()
G13_ESC = 8
exec(_t[_t.index("def _compress"):_t.index("def _pack13")])
cfgs = [tuple(map(int, c.split(","))) for c in sys.argv[1].split(";")]
cfgs = [c + (1,)[len(c) - 4:] for c in cfgs]  # KA (64-k atoms per stage) 1 by default
DEFS = [d_ for d_ in os.environ.get("DEFS", "").split(",") if d_]

def wg(bn):
    nacc = bn // 2
    regs = ",".join("%%%d" % i for i in range(nacc))
    outs = ", ".join('"+f"(d[%d])' % i for i in range(nacc))
    return ('template <> __device__ __forceinline__ void wgmma_ss<%d>(float* d, unsigned long long da, unsigned long long db) {\n'
            '    asm volatile("wgmma.mma_async.sync.aligned.m64n%dk16.f32.bf16.bf16 {%s}, %%%d, %%%d, 1, 1, 1, 0, 0;"\n'
            '                 : %s : "l"(da), "l"(db));\n}\n') % (bn, bn, regs, nacc, nacc + 1, outs)

src = "".join("#define %s\n" % d_.replace("=", " ") for d_ in DEFS) + open(os.path.join(D, "gemv28.cu")).read()
src = src.replace("WGMMA_DEFS", wg(64) + wg(128) + wg(256))
src = src.replace("G28_CONFIGS", " ".join("G28_CASE(%d,%d,%d,%d,%d)" % c for c in sorted(set(cfgs))))
mod = load_inline(name="pj_g28_" + hashlib.md5(src.encode()).hexdigest()[:10],
                  cpp_sources="torch::Tensor g28_trace(); void g28_gemm(torch::Tensor, double, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t); void g28_pred(torch::Tensor, torch::Tensor, int64_t);",
                  cuda_sources=src, functions=["g28_trace", "g28_gemm", "g28_pred"], extra_ldflags=["-lcuda"],
                  extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"] + (["-Xptxas", "-v"] if os.environ.get("PTXV") else []))


def pack28(w, bn):
    c = _compress(w)
    assert c is not None
    code, sm, hdr, scale = c
    n, k = w.shape
    # [tile, r, ks, c, 8] -> [tile, ks, r, c, 8]
    perm = lambda t: t.view(n // bn, bn, k // 64, 8, 8).permute(0, 2, 1, 3, 4).contiguous()
    b3, nw = _dec3(perm(sm), perm(code))
    st = (n // bn) * (k // 64)
    P = torch.cat([b3.reshape(st, bn * 64), nw.contiguous().view(torch.uint8).reshape(st, bn * 32)], 1).contiguous()
    return P, hdr, scale


SH = [("qkv", "qkv_gate", 2304), ("o", "o", 8192), ("pool", "pooler", 4096), ("unp", "unpooler", 8192)]
only = os.environ.get("SH")
GRID = int(os.environ.get("GRID", torch.cuda.get_device_properties(0).multi_processor_count))
PRED = int(os.environ.get("PRED", 3000))
NPRE = int(os.environ.get("NPRE", 99))
Wr = reference.generate_weights(reference.WEIGHTS_SEED)
for nm, layer, n in SH:
    if only and nm not in only.split(","): continue
    W = Wr[layer][:n].contiguous()
    k = W.shape[1]
    packs = {}
    for M in sorted({c[0] for c in cfgs}):
        x = torch.randn(M, k, device="cuda").bfloat16()
        xsrc = x.clone()
        out = torch.zeros(M, n, device="cuda")
        pre = (lambda: (out.zero_(), mod.g28_pred(xsrc, x, PRED))) if PRED >= 0 else (lambda: out.zero_())
        e = bench.time_graph(pre)
        tc = bench.time_graph(lambda: (pre(), F.linear(x, W))) - e
        line = f"{nm:5s} m{M} k{k}: pred {e:6.2f} cublas {tc:6.2f}"
        for (m_, bn, ns, nd, ka) in cfgs:
            if m_ != M or n % bn or k % (64 * ka): continue
            if bn not in packs:
                P, hdr, scale = pack28(W, bn)
                Wp = W.clone()
                er = (hdr[:, :, 1] != 2.0 ** -15).nonzero()
                if len(er):
                    cols = hdr[er[:, 0], er[:, 1], 0].view(torch.int32).long()
                    Wp[er[:, 0], cols] = torch.tensor(2.0 ** -15 * scale, dtype=torch.bfloat16, device="cuda")
                packs[bn] = (P, scale, Wp)
            P, scale, Wp = packs[bn]
            ref = x.double() @ Wp.double().T
            cf = (bn, ns, nd, ka, GRID, NPRE)
            try:
                for rep in range(3):
                    out.zero_(); mod.g28_gemm(P, scale, x, out, *cf); torch.cuda.synchronize()
                    err = ((out.double() - ref).norm() / ref.norm()).item()
                    if "NOCHK" not in DEFS: assert err < 1e-5, (err, nm, M, cf)
            except (RuntimeError, AssertionError) as ex:
                line += f" | {bn},{ns},{nd},{ka} ERR {str(ex).splitlines()[0][:80]}"; print(line, flush=True); continue
            dt = bench.time_graph(lambda: (pre(), mod.g28_gemm(P, scale, x, out, *cf))) - e
            line += f" | {bn},{ns},{nd},{ka} {dt:6.2f}"
            if os.environ.get("TR"):
                pre(); mod.g28_gemm(P, scale, x, out, *cf); torch.cuda.synchronize()
                tr = mod.g28_trace().view(3, 256).cpu(); t0 = tr[2, 0].item()
                nst = int((tr[2] > 0).sum())
                for j in range(min(nst, 40)):
                    print("   st %2d prod %8.3f dec %8.3f mma %8.3f" % (j, (tr[0, j] - t0) / 1e3 if tr[0, j] else -1, (tr[1, j] - t0) / 1e3, (tr[2, j] - t0) / 1e3))
        print(line, flush=True)
    del W, packs
