"""bnd2 vs Triton _boundary on the captured args: max diff + chained per-launch time (TRIG on the Triton one)."""
import sys, os, torch, fcntl, statistics
sys.path.insert(0, ".")
import bench
from torch.utils.cpp_extension import load_inline
sub = bench.load_sub(sys.argv[1])
src = "#include <torch/extension.h>\n#include <c10/cuda/CUDAStream.h>\n#include <cuda_bf16.h>\n" + open("/tmp/moeend/cb/bnd2_src.cu").read()
lib = load_inline("bnd2_t", cpp_sources="void bnd2_launch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, std::vector<torch::Tensor>, int64_t, double, double, double, int64_t, int64_t);",
                  cuda_sources=src, functions=["bnd2_launch"], extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"],
                  build_directory=os.environ.get("BD") or None, verbose=False)
A = torch.load("/tmp/moeend/bnd_args.pt", weights_only=False)
def tg(fn, n=20, reps=50):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn(); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            for _ in range(n):
                fn()
    ts = []
    with open("/tmp/okc-gpu.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for _ in range(reps):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) * 1000 / n)
    return statistics.median(ts)
for nm, (g, a, k) in A.items():
    br, s1, al, be, bg, ga, s2, hid, _ = a
    T = s1.shape[0]
    sc = [k["sc_ptr"], k["ids_ptr"], k["tw_ptr"]] if k.get("SC") else []
    mode = 2 if sc else (1 if br.dtype == torch.float32 else 0)
    nbc = 1 if k["NB"] == 1024 else 8
    s2r, hr = torch.empty_like(s2), torch.empty_like(hid)
    sub._boundary[g](*a[:6], s2r, hr, hr, **{kk: v for kk, v in k.items()}); torch.cuda.synchronize()
    tri = tg(lambda: sub._boundary[g](*a, **dict(k, TRIG=True)))
    G2 = (T * 8,); tri8 = tg(lambda: sub._boundary[G2](*a, **dict(k, TRIG=True, SPL=8)))
    out = []
    for spl in (2, 4, 8):
        s2c, hc = torch.zeros_like(s2), torch.zeros_like(hid)
        cfg = nbc * 100 + mode * 10 + spl + 1000
        f = lambda: lib.bnd2_launch(br, s1, al, be, bg, ga, s2c, hc, sc, k.get("SC_LOCAL", 0), k.get("SC_EPS", 0.0), k["EPS_POST"], k["EPS"], cfg, 1)
        f(); torch.cuda.synchronize()
        d1 = (s2c - s2r).abs().max().item(); d2 = (hc.float() - hr.float()).abs().max().item()
        nb = (hc != hr).sum().item()
        out.append(f"spl{spl} {tg(f):.2f} (ds {d1:.1e} dh {d2:.1e} nbad {nb})")
    print(f"{nm:18s} triton(spl {k["SPL"]}) {tri:.2f} tri-spl8 {tri8:.2f} | " + " | ".join(out))
