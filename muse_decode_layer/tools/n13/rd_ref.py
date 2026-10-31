"""n13s1: bare Triton int64 reader of 28.3 MB (qkv 12-bit bytes), cold-L2 graph minus empty, N CTAs."""
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
x = torch.empty(2304 * 8192 * 3 // 2 // 8, dtype=torch.int64, device="cuda").random_()
sink = torch.zeros(8192, dtype=torch.int64, device="cuda")
e0 = bench.empty_cost()
for n in (108, 132, 264, 528, 1056):
    for r in range(2):
        print(n, f"{bench.time_graph(lambda: _rd[(n,)](x, x.numel(), sink, BLOCK=1024, num_warps=8)) - e0:.2f}", flush=True)
