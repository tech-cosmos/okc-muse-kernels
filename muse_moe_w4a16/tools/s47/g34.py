"""moe s47: g34 (g30 at m16/m32) standalone H100: correctness vs F.linear on the layer's real pooler/unpooler weights
(escapes included) + timing = graph(pred 3 us + g34 as PDL secondary) - graph(pred), cold L2 (layer bench.time_graph).
usage: python g34.py MT name:S[,S..] ...   env DEFS="-DG30_NS=8 -DG30_NW=16"."""
import os, sys, hashlib
H = os.path.dirname(os.path.abspath(__file__))
L = os.path.join(H, "../../../muse_decode_layer")
sys.path.insert(0, L)
os.chdir(L)
import torch, torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import bench, reference
layer = bench.load_sub("submission.py")
MT = int(sys.argv[1])
DEFS = os.environ.get("DEFS", "").split() + [f"-DMT={MT}"]
src = open(os.path.join(H, "g34.cu")).read()
mod = load_inline("g34_" + hashlib.md5((" ".join(DEFS) + src + open(os.path.join(H, "helpers.inc")).read()).encode()).hexdigest()[:10],
                  cpp_sources="void g34_gemm(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, int64_t, int64_t); void g34_pred(int64_t);",
                  cuda_sources=src, functions=["g34_gemm", "g34_pred"],
                  extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo", "-I" + H] + DEFS,
                  extra_ldflags=["-lcuda"], verbose=False)
W = reference.generate_weights(reference.WEIGHTS_SEED)
PRED = int(os.environ.get("PRED", 3000))
e0 = bench.time_graph(lambda: mod.g34_pred(PRED))
for arg in sys.argv[2:]:
    name, Ss = arg.split(":")
    w = W[{"pool": "pooler", "unp": "unpooler"}[name]]
    c = layer._compress(w)
    code, sm, hdr, scale = c
    wp = layer._pack30(code, sm)
    hdr = hdr.contiguous()
    N, K = w.shape
    torch.manual_seed(1)
    x = torch.randn(MT, K, device="cuda").bfloat16()
    ref = F.linear(x, w).float()
    for S in map(int, Ss.split(",")):
        out = torch.zeros(MT, N, device="cuda")
        mod.g34_gemm(x, wp, hdr, out, scale, S, 0)
        torch.cuda.synchronize()
        err = ((out - ref).norm() / ref.norm()).item()
        out2 = torch.zeros(MT, N, device="cuda")
        mod.g34_gemm(x, wp, hdr, out2, scale, S, 0)
        det = torch.equal(out, out2)
        def f():
            mod.g34_pred(PRED)
            mod.g34_gemm(x, wp, hdr, out, scale, S, 1)
        R = int(os.environ.get("R", 5))
        ts, tcs = [], []
        for _ in range(R):
            ts.append(bench.time_graph(f) - e0)
            tcs.append(bench.time_graph(lambda: (mod.g34_pred(PRED), torch.mm(x, w.t()))) - e0)
        ts.sort(); tcs.sort()
        t, tc = ts[R // 2], tcs[R // 2]
        print("   reps g34", " ".join(f"{v:.2f}" for v in ts))
        print(f"m{MT} {name} S{S} {' '.join(DEFS)}: err {err:.2e} det {det} | g34 {t:.2f} us ({N * K * 1.5 / 1e6 / t:.2f} TB/s 12b) | cuBLAS {tc:.2f}", flush=True)
