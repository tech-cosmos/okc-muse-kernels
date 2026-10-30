"""n10s1: standalone cold-L2 graph of a bare reader (Triton int64 grid-stride) of MB bytes on N CTAs, minus empty graph:
slope/intercept = per-byte stream rate and first-op start cost on this GPU."""
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
buf = torch.ones(256 << 20, dtype=torch.uint8, device="cuda").view(torch.int64)
sink = torch.zeros(8192, dtype=torch.int64, device="cuda")
e0 = bench.empty_cost()
print(f"empty graph {e0:.2f} us")
for mb in (7.08, 14.16, 28.3, 56.6, 113.2):
    n = int(mb * 1e6) // 8
    row = []
    for ctas in (132, 264, 528, 1056, 2112):
        for bl, nw in ((1024, 8), (2048, 8)):
            t = bench.time_graph(lambda: _rd[(ctas,)](buf, n, sink, BLOCK=bl, num_warps=nw), reps=30) - e0
            row.append((t, ctas, bl))
    best = min(row)
    print(f"{mb:7.2f} MB best {best[0]:6.2f} us (ctas {best[1]}, block {best[2]}) = {mb / best[0]:.2f} TB/s;  all: " +
          " ".join(f"{t:.1f}" for t, _, _ in row), flush=True)
