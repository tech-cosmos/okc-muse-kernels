# n7s2: bound for an L2 pre-fill of the T=256 unpool W (cuBLAS F.linear, [8192,4096] BF16 = 67 MB).
# Graph: flush(256 MB) -> pf (read first P MB of W, rows-major prefix of each row? -> here: first P MB contiguous) -> gemm.
# Reports T(flush+pf+gemm) - T(flush+pf) per P (cold-L2 gemm time with P MB of its W resident).
import torch, fcntl, sys
import torch.nn.functional as F
torch.manual_seed(0)
W = torch.randn(8192, 4096, device="cuda", dtype=torch.bfloat16)
x = torch.randn(256, 4096, device="cuda", dtype=torch.bfloat16)
fl = torch.empty(256 << 20, device="cuda", dtype=torch.uint8)
sink = torch.empty(1, device="cuda")
mode = sys.argv[1] if len(sys.argv) > 1 else "rows"
def pf(P):
    if P == 0: return
    if mode == "rows":  # first P/67 fraction of every row (k prefix), like UPF
        kb = int(4096 * P / 67.1)
        sink.copy_(W[:, :kb].float().sum().reshape(1)) if kb else None
    else:  # first rows (n prefix)
        r = int(8192 * P / 67.1)
        sink.copy_(W[:r].float().sum().reshape(1))
def graph(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2): fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    return g
def tm(g, n=60):
    ts = []
    with open("/tmp/okc-gpu.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for _ in range(n):
            a = torch.cuda.Event(True); b = torch.cuda.Event(True)
            a.record(); g.replay(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b) * 1000)
    ts.sort(); return ts[len(ts) // 2]
for P in [0, 10, 20, 30, 40]:
    def f1(P=P): fl.add_(1); pf(P)
    def f2(P=P): fl.add_(1); pf(P); F.linear(x, W)
    a = tm(graph(f1)); b = tm(graph(f2))
    print(f"{mode} P={P:3d} MB  gemm after pf {b - a:6.1f} us  (pf+flush {a:.1f})", flush=True)
