import torch, fcntl, triton, triton.language as tl
torch.manual_seed(0)
NTOK = 600000  # cache rows (page 1): 2 heads x 64 B fp8 per row
kc = torch.randint(0, 255, (NTOK, 128), dtype=torch.uint8, device="cuda")
vc = torch.randint(0, 255, (NTOK, 128), dtype=torch.uint8, device="cuda")
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda"); sink = torch.zeros((), dtype=torch.int64, device="cuda")

@triton.jit
def gat(kc, vc, slots, out, N, ROWB: tl.constexpr, BLK: tl.constexpr):
    pid = tl.program_id(0)
    r = pid * BLK + tl.arange(0, BLK)
    s = tl.load(slots + r, mask=r < N, other=0).to(tl.int64)
    c = tl.arange(0, ROWB // 8)
    k = tl.load((kc + s[:, None] * 128).to(tl.pointer_type(tl.int64), bitcast=True) + c[None, :])
    v = tl.load((vc + s[:, None] * 128).to(tl.pointer_type(tl.int64), bitcast=True) + c[None, :])
    tl.store(out + pid, tl.sum(tl.sum(k ^ v, 1), 0))

for B in (16, 32):
    n = B * 2048
    slots = torch.randperm(NTOK, device="cuda")[:n].to(torch.int32)
    for ROWB in (64, 128):
        for BLK in (32, 64, 128):
            out = torch.zeros(triton.cdiv(n, BLK), dtype=torch.int64, device="cuda")
            f = lambda: gat[(triton.cdiv(n, BLK),)](kc, vc, slots, out, n, ROWB=ROWB, BLK=BLK, num_warps=4)
            f(); torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g): f()
            ts = []
            with open("/tmp/okc-gpu.lock", "w") as lk:
                fcntl.flock(lk, fcntl.LOCK_EX)
                for _ in range(30):
                    torch.sum(flush.view(torch.int64), 0, out=sink)
                    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                    e0.record(); g.replay(); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1) * 1e3)
                fcntl.flock(lk, fcntl.LOCK_UN)
            ts.sort(); t = ts[len(ts) // 2]
            print(f"B{B} rowB {ROWB} blk {BLK}: {t:.2f} us  {2 * n * ROWB / t / 1e6:.2f} TB/s (useful 128B/tok/head-pair)")
