"""proj n5s7: g30k (stream-K, FP32 slabs, the layer's T=16/32 pooler) with G30_GX (escape records per CTA, gxk.py) vs g30k.
Exactness: one-hot x over every escape column (+ random ones): sum of the slabs must reproduce W bit-exactly; random x rel err.
Timing: 1-kernel graphs, cold L2, standalone and after a 3 us PDL predecessor (proj's g19_pred).
Run from muse_bf16_proj/: python dev/n5s7/gxkm.py <g30_src.cu> [M=32] [G=104,120] [rounds=2]"""
import os, sys, hashlib, torch
sys.path.insert(0, ".")
import bench, reference
from torch.utils.cpp_extension import load_inline
L = bench.submission
SRC = open(sys.argv[1]).read()
M = int(sys.argv[2]) if len(sys.argv) > 2 else 32
GS = [int(g) for g in (sys.argv[3] if len(sys.argv) > 3 else "104,120").split(",")]
R = int(sys.argv[4]) if len(sys.argv) > 4 else 2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import gxk


def build(extra=()):
    dfs = ["-DG30K_ON", "-DG30K_F32=1", f"-DG30_MTOK={M}", "-DG30_NW=16", "-DG30_NS=16", *extra]
    return load_inline(name="pjn5s7k_" + hashlib.sha256((SRC + " ".join(dfs)).encode()).hexdigest()[:10],
                       cpp_sources="void g30k_gemm(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, int64_t, "
                                   "torch::Tensor, torch::Tensor, int64_t);",
                       cuda_sources=SRC, functions=["g30k_gemm"],
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", *dfs], extra_ldflags=["-lcuda"])


libs = {"old": build(), "gx": build(["-DG30_GX=1"] + os.environ.get("NEWDEFS", "").split())}
w = reference.generate_weights(reference.WEIGHTS_SEED)["pooler"]
N, K = w.shape
code, sm, hdr, scale = L._compress(w)
P = L._pack30(code, sm)
noz = torch.empty(0, device="cuda")
esc_cols = torch.unique(hdr[:, :, 0][hdr[:, :, 1] != 2.0 ** -15].contiguous().view(torch.int32)).tolist()
cols = esc_cols + torch.randperm(K)[:64].tolist()
for G in GS:
    KP = K // 128; Pu = N // 64 * KP
    nb = KP // (Pu // G) + 2
    HT = {"old": hdr, "gx": gxk.g30k_gx_tables(hdr, K, G)}
    print(f"G{G} gx records", None if HT["gx"] is None else tuple(HT["gx"].shape), flush=True)
    buf = torch.zeros(nb, M, N, device="cuda")
    run = lambda nm, x, pdl=0: libs[nm].g30k_gemm(x, P, HT[nm], buf, scale, pdl, noz, noz, G)  # noqa: E731
    msg = []
    for nm in libs:
        bad = 0
        for p in range(0, len(cols), M):
            cs = (cols[p:p + M] + cols[:M])[:M]
            xo = torch.zeros(M, K, device="cuda", dtype=torch.bfloat16)
            xo[torch.arange(M), torch.tensor(cs)] = 1
            buf.zero_(); run(nm, xo)
            o = buf.sum(0).bfloat16()
            bad += int((o.view(torch.int16) != w[:, cs].t().contiguous().view(torch.int16)).sum())
        x = torch.randn(M, K, device="cuda").bfloat16()
        ref = x.float() @ w.float().t()
        buf.zero_(); run(nm, x)
        msg.append(f"{nm}: onehot bad {bad}, err {((buf.sum(0) - ref).norm() / ref.norm()).item():.2e}")
    print(f"M{M} G{G} esc cols {len(esc_cols)} | " + " | ".join(msg), flush=True)
    xs = torch.randn(M, K, device="cuda").bfloat16(); x = xs.clone()
    pred = lambda: L._g19.g19_pred(xs, x, 3000)  # noqa: E731
    o0 = torch.zeros(1, device="cuda")
    e0, ep = bench.time_graph(lambda: o0.zero_()), bench.time_graph(pred)
    res = {}
    for r in range(R):
        for nm in libs:
            res.setdefault(nm, []).append((bench.time_graph(lambda: run(nm, x)) - e0, bench.time_graph(lambda: (pred(), run(nm, x, 1))) - ep))
    print(f"M{M} G{G} us standalone | after 3 us pred: " + "  ".join(
        f"{nm} " + "/".join(f"{a:.2f}" for a, _ in v) + " | " + "/".join(f"{b:.2f}" for _, b in v) for nm, v in res.items()), flush=True)
