# read-bound probe: time of reading the active experts' decode weights vs kernel_fn, CUDA graph, cold L2
import sys, fcntl, torch, triton, triton.language as tl
sys.path.insert(0, '.')
import bench, reference, submission
w, st = bench.state()
@triton.jit
def rd(p, eids, ebytes4, out, BLK: tl.constexpr):
    e = tl.load(eids + tl.program_id(1))
    base = p + e.to(tl.int64) * ebytes4
    acc = tl.zeros([BLK], tl.int32)
    n = ebytes4 // BLK
    for i in range(tl.program_id(0), n, tl.num_programs(0)):
        acc ^= tl.load(base + i * BLK + tl.arange(0, BLK))
    tl.store(out + tl.program_id(0) * BLK + tl.arange(0, BLK), acc)
out = torch.empty(1 << 22, dtype=torch.int32, device='cuda')
bufs = [st[k].view(torch.int32).view(32, -1) for k in ("w13q", "w13s2", "w2q", "w2s2")]
flush = torch.empty(512 << 20, dtype=torch.uint8, device='cuda')
def tg(fn, reps=30):
    g = torch.cuda.CUDAGraph(); fn(); torch.cuda.synchronize()
    with torch.cuda.graph(g): fn()
    ts = []
    with open('/tmp/okc-gpu.lock', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for _ in range(reps):
            flush.zero_(); a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1e3)
    ts.sort(); return ts[len(ts) // 2]
for bs in [int(a) for a in sys.argv[1:]]:
    for sd in (7, 8, 9):
        data = reference.generate_input(bs=bs, sharding='ep8', seed=sd)
        ids = data['topk_ids']; c = torch.bincount(ids[(ids >= 0) & (ids < 32)].long(), minlength=32)
        e = torch.nonzero(c).flatten().to(torch.int32); ne = e.numel()
        def r():
            for b in bufs:
                rd[(1056 // ne + 1, ne)](b, e, b.shape[1], out, BLK=1024, num_warps=8)
        tr = tg(r); tk = tg(lambda: submission.kernel_fn(data, st))
        mb = ne * 28.3
        print(f"bs {bs} seed {sd} experts {ne} max {int(c.max())}  read {tr:.1f} us ({mb/tr:.2f} TB/s)  kernel {tk:.1f} us  ratio {tk/tr:.2f}", flush=True)
