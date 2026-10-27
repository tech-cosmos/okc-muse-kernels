"""s41 xr prototype: build + correctness + cold-L2 graph timing vs cuBLAS (H100).
usage: python xr.py [shape ...]  shape = name:N:K:G  (default pool:4096:8192:8 unp:8192:4096:16)
env: DEFS="-DXR_NS=10 ..." (compile defines), REPS (default 50)."""
import os, sys, fcntl, statistics, torch
from torch.utils.cpp_extension import load_inline

HERE = os.path.dirname(os.path.abspath(__file__))
DEFS = os.environ.get("DEFS", "").split()
tag = "xr_" + str(abs(hash(" ".join(DEFS))) % 10**8)
mod = load_inline(tag, cpp_sources="void xr_run(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t G, torch::Tensor wp);",
                  cuda_sources=open(os.path.join(HERE, "xr_src.cu")).read(), functions=["xr_run"],
                  extra_cuda_cflags=["-O3", "-gencode", "arch=compute_90a,code=sm_90a", "-lineinfo"] + DEFS,
                  extra_ldflags=["-lcuda"], verbose=False)

REPS = int(os.environ.get("REPS", "50"))
flush = torch.ones(64 << 20, dtype=torch.float32, device="cuda")
fsum = torch.empty(1, dtype=torch.int64, device="cuda")


def gtime(fn):
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn(); fn()
        with torch.cuda.graph(g, stream=s):
            fn()
    torch.cuda.synchronize()
    ts = []
    with open("/tmp/okc-gpu.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for _ in range(REPS):
            fsum = flush.sum()
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record(); g.replay(); b.record()
            torch.cuda.synchronize()
            ts.append(a.elapsed_time(b) * 1000)
        fcntl.flock(lk, fcntl.LOCK_UN)
    return statistics.median(ts)


shapes = sys.argv[1:] or ["pool:4096:8192:8", "unp:8192:4096:16"]
for sh in shapes:
    name, N, K, G = sh.split(":"); N, K, G = int(N), int(K), int(G)
    torch.manual_seed(0)
    x = torch.randn(128, K, device="cuda", dtype=torch.bfloat16)
    w = (torch.randn(N, K, device="cuda") * 0.02).to(torch.bfloat16)
    out = torch.zeros(N, 128, device="cuda", dtype=torch.float32)
    # packed: [N/64][K/64][64 rows][8 chunks of 16 B], chunk c of row r stored at c ^ (r % 8) (= TMA SWIZZLE_128B)
    wb = w.view(N // 64, 64, K // 64, 8, 8).permute(0, 2, 1, 3, 4)  # [nt][kb][r][c][8]
    r = torch.arange(64, device="cuda").view(64, 1); c = torch.arange(8, device="cuda").view(1, 8)
    src = (c ^ (r % 8))  # position p holds chunk p ^ (r%8)
    wp = torch.gather(wb, 3, src.view(1, 1, 64, 8, 1).expand(N // 64, K // 64, 64, 8, 8)).contiguous()
    mod.xr_run(x, w, out, G, wp)
    ref = (w.float() @ x.float().t())
    err = ((out - ref).abs().max() / ref.abs().max()).item()
    def ours():
        out.zero_()
        mod.xr_run(x, w, out, G, wp)
    def ours_nz():
        mod.xr_run(x, w, out, G, wp)
    def cub():
        torch.mm(x, w.t())
    t0, t1, tc = gtime(ours), gtime(ours_nz), gtime(cub)
    mb = N * K * 2 / 1e6
    print(f"{name} N{N} K{K} G{G} {' '.join(DEFS)}: relerr {err:.2e} | xr+zero {t0:.2f} us ({mb / t0:.2f} TB/s) "
          f"| xr {t1:.2f} | cuBLAS {tc:.2f} ({mb / tc:.2f} TB/s)", flush=True)
    del x, w, out
