# n29 oracle: T=256 second boundary (_boundary NOS=2) with its pre-expert inputs (s FP32 8 MB, x1 BF16 4 MB) L2-cold vs warm.
# f (the unpool output) is always warm (it was just written). Median over reps, GPU lock around the timed loop.
import sys, os, fcntl, statistics, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import submission as L
T, N = int(os.environ.get("T", 256)), L.HIDDEN
dev = "cuda"
g = torch.Generator(device=dev).manual_seed(0)
f = torch.randn(T, N, device=dev, generator=g).bfloat16()
s = torch.randn(T, N, device=dev, generator=g)
x1 = torch.randn(T, N, device=dev, generator=g).bfloat16()
a1, b1 = torch.rand(N, device=dev) + 0.5, torch.rand(N, device=dev) + 0.5
rb = torch.rand(T, device=dev) + 0.5
alpha, beta = torch.rand(N, device=dev) + 0.5, torch.rand(N, device=dev) + 0.5
pf, nx = torch.rand(N, device=dev), torch.rand(N, device=dev)
s2, hid = torch.empty_like(s), torch.empty_like(f)
flush = torch.empty(200 << 20, dtype=torch.uint8, device=dev)
def run():
    L._boundary[(T,)](f, s, alpha, beta, pf, nx, s2, hid, hid, NB=N, N=N, BGAIN=True, H32=False,
                      EPS_POST=1e-6, EPS=1e-6, PDL=False, RB=False, num_warps=16, launch_pdl=False,
                      SPL=1, x1_ptr=x1, a1_ptr=a1, b1_ptr=b1, rb_ptr=rb, NOS=2)
run(); torch.cuda.synchronize()
def touch(ts):
    for t in ts:
        t.view(torch.uint8).sum(dtype=torch.int32)  # read into L2 (8+4+4 MB fit)
res = {}
from torch.profiler import profile, ProfilerActivity
with open("/tmp/okc-gpu.lock", "w") as lk:
    fcntl.flock(lk, fcntl.LOCK_EX)
    for mode in ["cold", "warm", "cold", "warm"]:
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            for _ in range(60):
                flush.zero_()
                touch([f] if mode == "cold" else [s, x1, f])
                run()
            torch.cuda.synchronize()
        ts = [e.device_time for e in p.events() if e.name.startswith("_boundary")]
        res.setdefault(mode, []).append(statistics.median(ts))
    fcntl.flock(lk, fcntl.LOCK_UN)
print(f"T={T} bnd2 NOS=2 us: " + ", ".join(f"{k} {v}" for k, v in res.items()))
