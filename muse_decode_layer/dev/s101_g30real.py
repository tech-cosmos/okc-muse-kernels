"""s101: split-K g30 (dev/s101_g30split.cu, 9-arg g30_gemm) on the real (reference) weights incl. escapes, cold-L2 graph us vs cuBLAS. usage: W=qkv_gate SPLIT=3 python dev/s101_g30real.py [seed]; SRC=<cu> PRE="#define X 1;..." NOESC=1"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import bench, reference
L = bench.load_sub("submission.py")
L._G30_SRC = open(os.environ.get("SRC") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "s101_g30split.cu")).read()
L._G30_SRC = os.environ.get("PRE", "").replace(";", "\n") + "\n" + L._G30_SRC
from torch.utils.cpp_extension import load_inline
lib = load_inline(name="s101_g30_" + __import__("hashlib").sha256(L._G30_SRC.encode()).hexdigest()[:10],
                  cpp_sources="void g30_gemm(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, int64_t, torch::Tensor, torch::Tensor, int64_t);",
                  cuda_sources=L._G30_SRC, functions=["g30_gemm"],
                  extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], extra_ldflags=["-lcuda"])
WN = os.environ.get("W", "qkv_gate")
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
w = reference.generate_weights(seed)[WN]
code, sm, hdr, scale = L._compress(w)
wp = L._pack30(code, sm)
x = torch.randn(128, w.shape[1], device="cuda").bfloat16()
e0 = bench.empty_cost()
tc = bench.time_graph(lambda: torch.mm(x, w.t())) - e0
ref = torch.mm(x, w.t()).float()
nesc = int((hdr[:, :, 1] != 2.0 ** -15).sum())
if os.environ.get("NOESC"):  # timing ablation: no escape candidates (wrong results)
    hdr = hdr.clone(); hdr[:, :, 1] = 2.0 ** -15
for sp in [int(a) for a in os.environ.get("SPLIT", "3").split(",")]:
    ws = torch.empty(sp * 128 * w.shape[0], device="cuda"); cnt = torch.zeros(w.shape[0] // 64, dtype=torch.int32, device="cuda")
    out = torch.empty(128, w.shape[0], device="cuda", dtype=torch.bfloat16)
    for pdl in (0, 1):
        t = bench.time_graph(lambda: lib.g30_gemm(x, wp, hdr, out, scale, pdl, ws, cnt, sp)) - e0
        err = ((out.float() - ref).norm() / ref.norm()).item()
        print(f"{WN} seed {seed} split {sp} pdl {pdl} esc {nesc}: g30 {t:.2f} us err {err:.1e} | cuBLAS {tc:.2f} us", flush=True)
