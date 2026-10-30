"""n10s1: fixed first-op costs on this GPU (cold-L2 graph minus empty graph): 1 nop kernel, 2 nops, tiny cold reads."""
import sys, torch, triton, triton.language as tl
sys.path.insert(0, ".")
import bench
@triton.jit
def _rd(p, n, out, BLOCK: tl.constexpr):
    pid = tl.program_id(0); np_ = tl.num_programs(0)
    acc = tl.zeros([BLOCK], dtype=tl.int64)
    for i in range(pid * BLOCK, n, np_ * BLOCK):
        o = i + tl.arange(0, BLOCK)
        acc += tl.load(p + o, mask=o < n, other=0)
    tl.store(out + pid, tl.sum(acc, 0))
@triton.jit
def _nop(out):
    pass
buf = torch.ones(256 << 20, dtype=torch.uint8, device="cuda").view(torch.int64)
sink = torch.zeros(8192, dtype=torch.int64, device="cuda")
e0 = bench.empty_cost()
for name, fn in [("nop x1", lambda: _nop[(1,)](sink)), ("nop x2", lambda: (_nop[(1,)](sink), _nop[(1,)](sink))),
                 ("nop 2112 ctas", lambda: _nop[(2112,)](sink)),
                 ("read 8 KB 1 cta", lambda: _rd[(1,)](buf, 1024, sink, BLOCK=1024, num_warps=8)),
                 ("read 1 MB 132", lambda: _rd[(132,)](buf, 1 << 17, sink, BLOCK=1024, num_warps=8)),
                 ("read 2 MB 264", lambda: _rd[(264,)](buf, 1 << 18, sink, BLOCK=1024, num_warps=8))]:
    ts = sorted(bench.time_graph(fn, reps=40) - e0 for _ in range(3))
    print(f"{name:18s} {ts[1]:6.2f} us", flush=True)
