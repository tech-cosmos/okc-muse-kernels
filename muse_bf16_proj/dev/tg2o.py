"""H100: tinygemm2 PDL (S4/8/16 stages) after a 3 us PDL predecessor vs the proj pick (no PDL): o m8/m1, unpool slice m1."""
import os, sys, torch
sys.path.insert(0, "/home/daytona/r-proj/muse_bf16_proj"); sys.path.insert(0, "/home/daytona/r-proj/muse_bf16_proj/dev")
import bench
from torch.utils.cpp_extension import load_inline
S = bench.submission
src = open("/home/daytona/r-proj/muse_bf16_proj/dev/tinygemm2.cu").read()
mod = load_inline(name="pj_tg2s", cpp_sources="void tg2s(torch::Tensor, torch::Tensor, torch::Tensor, bool, int64_t);",
                  cuda_sources="#define TG2_STAGES\n" + src, functions=["tg2s"], extra_ldflags=["-lcuda"],
                  extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])
w, st = bench.state()
for layer, k, n, m in (("o", 1024, 8192, 8), ("o", 1024, 8192, 1), ("unpooler", 4096, 1024, 1)):
    W = st[layer][:n]; xs = torch.randn(m, k, device="cuda").to(torch.bfloat16); x = xs.clone()
    out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda"); pick = st["pick"].get((layer, m))
    for _ in range(2):
        fns = [lambda: S._g19.g19_pred(xs, x, 3000), lambda: (S._g19.g19_pred(xs, x, 3000), S._run(st, layer, x, out, pick))]
        for s_ in (4, 8, 16):
            fns.append(lambda s_=s_: (S._g19.g19_pred(xs, x, 3000), mod.tg2s(x, W, out, True, s_)))
        med = S._graph_medians(fns); p = med[0] * 1e3
        print(f"{layer} m{m} pred {p:.2f}; +pick {pick} {med[1]*1e3-p:+.2f}; tg2 PDL S4/8/16 " + " ".join(f"{t*1e3-p:+.2f}" for t in med[2:]), flush=True)
