"""s64: K/V gather floor on the layer's real decode 16/32 sliding inputs (bench.make_input caches + page table, window 2048,
page-table trip included): whole 128 B rows vs csl's pattern (64 B head halves from separate programs) vs one head.
H100: B32 12.3 / 12.9 / 10.9 us, B16 9.5 / 9.8 / 8.7 us.  python tools/gather_real.py"""
import sys, fcntl, torch, triton, triton.language as tl
sys.path.insert(0, ".")
import bench, reference
sub = bench.load_sub("submission.py")
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda"); sink = torch.zeros((), dtype=torch.int64, device="cuda")
w = reference.generate_weights(reference.WEIGHTS_SEED)

@triton.jit
def gat(kc, vc, pt, seqs, out, MP, ROWB: tl.constexpr, BLK: tl.constexpr, HALF: tl.constexpr):
    b = tl.program_id(1); pid = tl.program_id(0)
    hh = 0
    if HALF:
        hh = pid % 2
        pid = pid // 2
    seq = tl.load(seqs + b)
    t = seq - 2048 + pid * BLK + tl.arange(0, BLK)
    t = tl.minimum(tl.maximum(t, 0), seq - 1)
    s = tl.load(pt + b * MP + t).to(tl.int64)
    c = tl.arange(0, ROWB // 8)
    k = tl.load((kc + s[:, None] * 128 + hh * 64).to(tl.pointer_type(tl.int64), bitcast=True) + c[None, :])
    v = tl.load((vc + s[:, None] * 128 + hh * 64).to(tl.pointer_type(tl.int64), bitcast=True) + c[None, :])
    tl.store(out + b * 1024 + pid, tl.sum(tl.sum(k ^ v, 1), 0))

for load in ("decode 32", "decode 16"):
    c = next(c for l, c in bench.SCORED if l == load and c["layer"] == "sliding")
    B = c["batch"]; d0 = bench.make_input(c, w, 9100); cfg = d0["config"]
    print(load, "k_cache", tuple(d0["k_cache"].shape), d0["k_cache"].dtype, "pt", tuple(d0["page_table"].shape), "page", cfg["page_size"])
    kc, vc = d0["k_cache"].view(torch.uint8).view(-1, 128), d0["v_cache"].view(torch.uint8).view(-1, 128)
    out = torch.zeros(B * 1024, dtype=torch.int64, device="cuda")
    for ROWB, HALF in ((128, 0), (64, 1), (64, 0)):
        for BLK in (32, 64):
            f = lambda: gat[(2048 // BLK * (2 if HALF else 1), B)](kc, vc, d0["page_table"], d0["seq_lens"], out, d0["page_table"].shape[1], ROWB=ROWB, BLK=BLK, HALF=HALF, num_warps=4)
            f(); torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g): f()
            ts = []
            with open("/tmp/okc-gpu.lock", "w") as lk:
                fcntl.flock(lk, fcntl.LOCK_EX)
                for _ in range(40):
                    torch.sum(flush.view(torch.int64), 0, out=sink)
                    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                    e0.record(); g.replay(); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1) * 1e3)
                fcntl.flock(lk, fcntl.LOCK_UN)
            ts.sort(); print(f"  rowB {ROWB} half {HALF} blk {BLK}: {ts[len(ts)//2]:.2f} us")
