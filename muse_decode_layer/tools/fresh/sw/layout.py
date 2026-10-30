"""fresh x5 oracle (5), codex #10: expert-weight slab placement. Same bytes, same kernels; every moe-state tensor > 4 MB is
re-copied into a fresh buffer at byte offset OFF (1 KB multiples keep TMA/swizzle alignment), then the scored shapes are timed.
OFF 0 twice (control + noise). python tools/fresh/sw/layout.py <submission.py> <filter> <rounds>"""
import sys, statistics, torch
sys.path.insert(0, ".")
import bench
sub = bench.load_sub(sys.argv[1]); flt = sys.argv[2]; rounds = int(sys.argv[3])
w, st = bench.weights_state(sub)
ms = st["moe"]
big = [k for k, v in ms.items() if torch.is_tensor(v) and v.is_cuda and v.nbytes > (4 << 20)]
print("rebased:", ", ".join(f"{k} {ms[k].nbytes / 1e6:.0f}MB" for k in big), flush=True)
names = [(f"{l} {c['layer']}", c) for l, c in bench.SCORED if flt in f"{l} {c['layer']}"]
data = {n: bench.make_input(c, w, 99) for n, c in names}
OFFS = [0, 1024, 8192, 65536 + 3072, (1 << 20) + 5120, 0]
bufs = {}
def rebase(off):
    for k in big:
        v = ms[k]; nb = v.nbytes
        buf = torch.empty(nb + (2 << 20), dtype=torch.uint8, device="cuda")
        new = buf[off:off + nb].view(v.dtype).view(v.shape); new.copy_(v)
        ms[k] = new; bufs[k] = buf  # old buffer freed
    torch.cuda.synchronize(); torch.cuda.empty_cache()
e0 = bench.empty_cost()
res = {}
for r in range(rounds):
    for vi, off in enumerate(OFFS):
        rebase(off)
        for n, _ in names:
            res.setdefault((n, vi), []).append(bench.time_graph(lambda: sub.kernel_fn(data[n], st)) - e0)
for n, _ in names:
    print(f"{n:18s}", "  ".join(f"[{OFFS[i] // 1024}K] {statistics.median(res[(n, i)]):7.2f}" for i in range(len(OFFS))), flush=True)
