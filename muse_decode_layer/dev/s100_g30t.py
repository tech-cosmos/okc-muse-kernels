"""s100: layer g30 (g30_src.cu via submission.py) on the reference unpooler: exactness vs F.linear + cold-L2 timing."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F
import bench, reference
L = bench.load_sub("submission.py")
L._G30_SRC = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "g30_src.cu")).read()  # the dev copy
L._G30_SRC = "".join(f"#define {d}\n" for d in os.environ.get("DEFS", "").split(",") if d) + L._G30_SRC
L._g30_build()
M = int(os.environ.get("M", 128))
lib = L._G30["lib"]
W = reference.generate_weights(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
for key, w in [(k, W["unpooler"] if k == "unpool" else W[k]) for k in os.environ.get("KEYS", "unpool").split(",")]:
    c = L._compress(w)
    code, sm, hdr, scale = c
    nesc = int((hdr[:, :, 1] != 2.0 ** -15).sum())
    wp = L._pack30(code, sm)
    x = torch.randn(M, w.shape[1], device="cuda").bfloat16()
    out = torch.empty(M, w.shape[0], device="cuda", dtype=torch.bfloat16)
    lib.g30_gemm(x, wp, hdr, out, scale, 0)
    ref32 = x.float() @ w.float().t()
    ref = F.linear(x, w)
    e = ((out.float() - ref32).norm() / ref32.norm()).item()
    ec = ((ref.float() - ref32).norm() / ref32.norm()).item()
    mx = (out.float() - ref32).abs().max().item()
    e0 = bench.empty_cost()
    t = bench.time_graph(lambda: lib.g30_gemm(x, wp, hdr, out, scale, 0)) - e0
    tc = bench.time_graph(lambda: F.linear(x, w)) - e0
    print(f"M{M} {key} {tuple(w.shape)} escapes {nesc}: g30 err {e:.2e} (max {mx:.2e}) vs cuBLAS err {ec:.2e} | g30 {t:.2f} us cuBLAS {tc:.2f} us")
