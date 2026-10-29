# n7s3: T=256 layer _boundary (bnd1 after the o GEMM, bnd2 after unpool) on the H100: time after a cuBLAS producer with a
# cold L2 (flush -> F.linear -> X) minus (flush -> F.linear), vs a same-bytes copy kernel and an empty kernel.
import torch, fcntl, sys, os
import torch.nn.functional as F
import triton, triton.language as tl
sys.path.insert(0, os.path.expanduser("~/r-proj/muse_decode_layer"))
import submission as L
torch.manual_seed(0)
T, N = 256, 8192
W = torch.randn(N, 4096, device="cuda", dtype=torch.bfloat16) * 0.02
x = torch.randn(T, 4096, device="cuda", dtype=torch.bfloat16)
s = torch.randn(T, N, device="cuda")
al, be, g1, g2 = [torch.rand(N, device="cuda") + 0.5 for _ in range(4)]
so = torch.empty_like(s); h = torch.empty(T, N, device="cuda", dtype=torch.bfloat16)
fl = torch.zeros(64 << 20, device="cuda", dtype=torch.float32)
sink = torch.empty((), device="cuda")
RF = os.environ.get("RF", "1") == "1"  # read flush (clean L2, like the GEMM before the boundary) vs write flush
def flush():
    torch.sum(fl, 0, out=sink) if RF else fl.add_(1)
a = torch.empty(T, N, device="cuda", dtype=torch.bfloat16)

@triton.jit
def _copy(b_ptr, s_ptr, so_ptr, h_ptr, N: tl.constexpr):
    row = tl.program_id(0).to(tl.int64); c = tl.arange(0, N)
    v = tl.load(s_ptr + row * N + c); b = tl.load(b_ptr + row * N + c).to(tl.float32)
    tl.store(so_ptr + row * N + c, v + b); tl.store(h_ptr + row * N + c, (v * b).to(tl.bfloat16))
@triton.jit
def _empty(p):
    pass

def bnd(bg, spl=1, nw=16):
    L._boundary[(T * spl,)](a, s, al, be, g2, g1, so, h, h, NB=N, N=N, BGAIN=bg, H32=False, EPS_POST=L.POST_NORM_EPS,
                            EPS=L.RMS_EPS, PDL=False, RB=False, num_warps=nw, SPL=spl)
VAR = {"none": lambda: None, "empty": lambda: _empty[(T,)](a),
       "copy": lambda: _copy[(T,)](a, s, so, h, N=N, num_warps=16),
       "bnd1": lambda: bnd(False), "bnd2": lambda: bnd(True), "bnd1s2": lambda: bnd(False, 2), "bnd1w8": lambda: bnd(False, 1, 8),
       "bnd1w32": lambda: bnd(False, 1, 32)}
extra = {}
if len(sys.argv) > 2:  # extra variants from a module: name -> fn(a, s, al, be, g1, g2, so, h)
    import importlib.util
    sp = importlib.util.spec_from_file_location("ext", sys.argv[2]); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    for k, f in m.VARIANTS.items():
        VAR[k] = (lambda f=f: f(a, s, al, be, g1, g2, so, h))
def graph(fn):
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(2): fn()
    torch.cuda.current_stream().wait_stream(st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    return g
def tm(g, n=80):
    ts = []
    for _ in range(n):
        e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
        e0.record(); g.replay(); e1.record(); e1.synchronize(); ts.append(e0.elapsed_time(e1) * 1000)
    ts.sort(); return ts[len(ts) // 2]
names = sys.argv[1].split(",") if len(sys.argv) > 1 else list(VAR)
gs = {}
for k in names:
    def f(k=k): flush(); torch.matmul(x, W.t(), out=a); VAR[k]()
    gs[k] = graph(f)
res = {k: [] for k in names}
with open("/tmp/okc-gpu.lock", "w") as lk:
    fcntl.flock(lk, fcntl.LOCK_EX)
    for r in range(3):
        for k in names: res[k].append(tm(gs[k]))
base = sorted(res["none"])[1] if "none" in res else 0
for k in names:
    print(f"{k:10s} {sorted(res[k])[1] - base:7.2f} us  {['%.1f' % v for v in res[k]]}")
