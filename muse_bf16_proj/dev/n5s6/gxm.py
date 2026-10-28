"""proj n5s6 (from n5s4 escm.py): g30 G30_GX (escapes via prepare() records + TMA x boxes + WG1 fixup walk); was: g30 ESCM (escapes fixed after the main loop) vs g30 (escapes patched into A) vs no escape path (timing only).
Exactness: one-hot x over every escape column (+ random columns) must reproduce W bit-exactly; random x: rel err vs FP32.
Timing: 1-kernel graphs, cold L2, standalone (minus an empty graph) and after a 3 us PDL predecessor (minus the pred alone).
Run from muse_bf16_proj/: python dev/n5s4/escm.py [M=32] [keys=qkv_gate:3,pooler:2,unpooler:1] [rounds=2]"""
import os, sys, hashlib, torch
sys.path.insert(0, ".")
import bench, reference
from torch.utils.cpp_extension import load_inline
L = bench.submission
M = int(sys.argv[1]) if len(sys.argv) > 1 else 32
KEYS = [(k.split(":")[0], int(k.split(":")[1])) for k in (sys.argv[2] if len(sys.argv) > 2 else "qkv_gate:3,pooler:2,unpooler:1").split(",")]
R = int(sys.argv[3]) if len(sys.argv) > 3 else 2
OLD = NEW = open("dev/g30_src.cu").read()
sys.path.insert(0, "dev/n5s6"); import gx
def build(src, S, extra=()):
    dfs = [f"-DG30_MTOK={M}", f"-DG30_S={S}", "-DG30_NW=16", "-DG30_NS=16", *extra]
    return load_inline(name="pjn5s6g_" + hashlib.sha256((src + " ".join(dfs)).encode()).hexdigest()[:10],
                       cpp_sources="void g30_gemm(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, int64_t);",
                       cuda_sources=src, functions=["g30_gemm"],
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", *dfs], extra_ldflags=["-lcuda"])
Wall = reference.generate_weights(reference.WEIGHTS_SEED)
for key, S in KEYS:
    libs = {"old": build(OLD, S), "gx": build(NEW, S, ["-DG30_GX=1"] + os.environ.get("NEWDEFS", "").split()), "noesc": build(OLD, S, ["-DG30_ESC=0"])}
    w = Wall[key]; N, K = w.shape
    code, sm, hdr, scale = L._compress(w)
    P = L._pack30(code, sm)
    HT = {nm: (gx.g30_gx_tables(hdr, K, S) if nm == 'gx' else hdr) for nm in libs}
    print('gx records', None if HT['gx'] is None else tuple(HT['gx'].shape), 'max ncols', None if HT['gx'] is None else int(HT['gx'][:, 0].max()), flush=True)
    esc_cols = torch.unique(hdr[:, :, 0][hdr[:, :, 1] != 2.0 ** -15].contiguous().view(torch.int32)).tolist()
    cols = esc_cols + torch.randperm(K)[:64].tolist()
    msg = []
    for nm in ("old", "gx"):
        lib, bad = libs[nm], 0
        for p in range(0, len(cols), M):
            cs = (cols[p:p + M] + cols[:M])[:M]
            xo = torch.zeros(M, K, device="cuda", dtype=torch.bfloat16)
            xo[torch.arange(M), torch.tensor(cs)] = 1
            o = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
            lib.g30_gemm(xo, P, HT[nm], o, scale, 0)
            bad += int((o.view(torch.int16) != w[:, cs].t().contiguous().view(torch.int16)).sum())
        x = torch.randn(M, K, device="cuda").bfloat16()
        ref = x.float() @ w.float().t()
        o = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        lib.g30_gemm(x, P, HT[nm], o, scale, 0)
        msg.append(f"{nm}: onehot bad {bad}, err {((o.float() - ref).norm() / ref.norm()).item():.2e}")
    print(f"M{M} {key} S{S} esc cols {len(esc_cols)} | " + " | ".join(msg), flush=True)
    xs = torch.randn(M, K, device="cuda").bfloat16(); x = xs.clone()
    out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    pred = lambda: L._g19.g19_pred(xs, x, 3000)  # noqa: E731
    o0 = torch.zeros(1, device="cuda")
    e0, ep = bench.time_graph(lambda: o0.zero_()), bench.time_graph(pred)
    res = {}
    for r in range(R):
        for nm, lib in libs.items():
            res.setdefault(nm, []).append((bench.time_graph(lambda: lib.g30_gemm(x, P, HT[nm], out, scale, 0)) - e0,
                                           bench.time_graph(lambda: (pred(), lib.g30_gemm(x, P, HT[nm], out, scale, 1))) - ep))
    print(f"M{M} {key} S{S} us standalone | after 3 us pred: " + "  ".join(
        f"{nm} " + "/".join(f"{a:.2f}" for a, _ in v) + " | " + "/".join(f"{b:.2f}" for _, b in v) for nm, v in res.items()), flush=True)
