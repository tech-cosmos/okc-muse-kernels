"""n7s5: cuBLASLt algo-66 screen for the layer's FR GEMM (T=128/256): out[T, 4608] FP32 = h[T, 8192] @ [pooler; hi; lo]^T.
Usage: python frlt.py 128,256   (H100; cold L2 like the judge; baseline = torch.mm(out_dtype=float32) = layer main)."""
import os, sys, time, fcntl, torch
from torch.utils.cpp_extension import load_inline
HERE = os.path.dirname(os.path.abspath(__file__))


class bench:  # self-contained copy of muse_bf16_proj/bench.py time_graph (cold L2: 512 MB read before each replay)
    _F = _S = _E = None

    @staticmethod
    def time_graph(fn, reps=100):
        if bench._F is None:
            bench._F = torch.empty(512 << 20, dtype=torch.uint8, device="cuda"); bench._S = torch.empty(1, dtype=torch.int64, device="cuda")
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn(); torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s):
                fn()
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
        ts = []
        with open("/tmp/okc-gpu.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for _ in range(reps):
                torch.sum(bench._F.view(torch.int64), 0, out=bench._S)
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record(); g.replay(); b.record(); torch.cuda.synchronize()
                ts.append(a.elapsed_time(b) * 1000)
        ts.sort()
        return ts[len(ts) // 2]

    @staticmethod
    def empty_cost():
        if bench._E is None:
            bench._E = bench.time_graph(lambda: None, reps=200)
        return bench._E
CPP = open(os.path.join(HERE, "frlt_cpp.inc")).read()
mod = load_inline("pj_lt_f32", CPP, functions=["lt_desc", "lt_algos", "lt_mm", "lt_enum"], extra_ldflags=["-lcublasLt"],
                  with_cuda=True, verbose=False)
torch.manual_seed(0)
K, NP, NR = 8192, 4096, 256
wp = (torch.randn(NP, K, device="cuda") * 0.02).bfloat16()
r32 = torch.randn(NR, K, device="cuda") * 0.02
hi = r32.bfloat16(); lo = (r32 - hi.float()).bfloat16()
W = torch.cat([wp, hi, lo]).contiguous()
N = W.shape[0]
wsb = 64 << 20
ws = torch.empty(wsb, dtype=torch.uint8, device="cuda")
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
for m in [int(a) for a in sys.argv[1].split(",") if a != "0"]:
    h = torch.randn(m, K, device="cuda").bfloat16()
    ref = torch.mm(h, W.t(), out_dtype=torch.float32)
    tc = bench.time_graph(lambda: torch.mm(h, W.t(), out_dtype=torch.float32)) - bench.empty_cost()
    al = mod.lt_enum(m, N, K, wsb, 10**7)
    out = torch.empty(m, N, dtype=torch.float32, device="cuda")
    n = al.shape[0]
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(n)]
    best = {}
    t0 = time.time()
    for rep in range(2):
        with open("/tmp/okc-gpu.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            sts = []
            for i in range(n):
                flush.zero_()
                ev[i][0].record()
                sts.append(mod.lt_mm(W, h, out, al[i], ws))
                ev[i][1].record()
            torch.cuda.synchronize()
        for i in range(n):
            if sts[i] == 0:
                best[i] = min(best.get(i, 1e9), ev[i][0].elapsed_time(ev[i][1]) * 1000)
    order = sorted(best, key=best.get)
    top = []
    for i in order[:40]:
        a = al[i].clone()
        mod.lt_mm(W, h, out, a, ws); torch.cuda.synchronize()
        err = ((out - ref).norm() / ref.norm()).item()
        if not err < 1e-5: continue
        top.append((bench.time_graph(lambda: mod.lt_mm(W, h, out, a, ws), reps=50) - bench.empty_cost(), i))
    top.sort()
    tc2 = bench.time_graph(lambda: torch.mm(h, W.t(), out_dtype=torch.float32)) - bench.empty_cost()
    print(f"T={m} FR torch.mm {tc:.2f}/{tc2:.2f} | valid {n} ran {len(best)} {time.time()-t0:.0f}s | best",
          " ".join(f"{t:.2f}{mod.lt_desc(al[i]).tolist()[1:]}" for t, i in top[:6]), flush=True)
    os.makedirs("/tmp/pj", exist_ok=True)
    torch.save({"top": [(t, al[i].clone(), mod.lt_desc(al[i]).tolist()) for t, i in top], "torch_mm": (tc, tc2)}, f"/tmp/pj/frlt_{m}.pt")
