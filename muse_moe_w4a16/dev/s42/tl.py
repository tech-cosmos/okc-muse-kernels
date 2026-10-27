"""s42 timeline: python tl.py N K G  (DEFS env as xr2.py). Prints per-tile stamps (us from kernel min) for cluster 0."""
import os, sys, torch
from torch.utils.cpp_extension import load_inline
HERE = os.path.dirname(os.path.abspath(__file__))
DEFS = os.environ.get("DEFS", "").split()
mod = load_inline("xr2tl_" + str(abs(hash(" ".join(DEFS) + os.environ.get("SRC", ""))) % 10**8),
                  cpp_sources="void xr_run(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t G, torch::Tensor wp);\ntorch::Tensor xr_tl();",
                  cuda_sources=open(os.path.join(HERE, os.environ.get("SRC", "xr2_src.cu"))).read(), functions=["xr_run", "xr_tl"],
                  extra_cuda_cflags=["-O3", "-gencode", "arch=compute_90a,code=sm_90a"] + DEFS + ["-DXR_TLON"], extra_ldflags=["-lcuda"])
N, K, G = map(int, sys.argv[1:4])
x = torch.randn(128, K, device="cuda", dtype=torch.bfloat16)
w = (torch.randn(N, K, device="cuda") * 0.02).to(torch.bfloat16)
out = torch.zeros(128, N, device="cuda", dtype=torch.bfloat16)
fl = torch.ones(64 << 20, device="cuda")
for _ in range(3):
    fl.sum(); mod.xr_run(x, w, out, G, w); torch.cuda.synchronize()
tl = mod.xr_tl().double()
S = K // 512
used = tl[: G * S]
t0 = used[used > 0].min()
tl = (tl - t0) / 1000
print("end of kernel (max stamp) %.2f us" % ((used.max() - t0) / 1000))
for cta in [0 * G, 1 * G, 7 * G]:
    print("cta y=%d: tile: prodstart | cons done | after freeb+store | after bar | red got ready | red done | WG1 done | WG1 after freeb" % (cta // G))
    for t in range(N // 64 // G):
        print("  t%d " % t + " ".join("%7.2f" % v for v in tl[cta, t].tolist()))
