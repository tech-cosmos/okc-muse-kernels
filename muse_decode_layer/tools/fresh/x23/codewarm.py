"""x23 oracle: how much of each load is COLD CODE (kernel SASS fetched from DRAM after the judge's L2 flush)?
Two identical twins (state A, state B = prepare() twice on the same weights, TUNE pinned off; inputs A, B equal values,
separate buffers). Per rep: 512 MB flush read, then
  base: replay gA (timed)
  warm: replay gB (untimed: brings every kernel's code into L2, plus B's last ~50 MB of data), then replay gA (timed):
        A's data is cold (L2 holds B's lines), the code is L2-warm.
Run from muse_decode_layer/: python tools/fresh/x23/codewarm.py FILE 'decode 1 sliding,...' [rounds]"""
import fcntl, statistics, sys, torch
sys.path.insert(0, ".")
import bench, reference

path = sys.argv[1]
flts = sys.argv[2].split(",")
rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
sub = bench.load_sub(path)
w = reference.generate_weights(reference.WEIGHTS_SEED)
sA = sub.prepare(w)
sB = sub.prepare(w)
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
sink = torch.zeros((), dtype=torch.int64, device="cuda")


def run(gA, gB, mode, reps=30):
    ts = []
    for _ in range(reps):
        torch.sum(flush.view(torch.int64), 0, out=sink)
        if mode == "warm":
            gB.replay()
        elif mode == "warmflush":  # control: B, then flush again (code evicted again) -> should equal base
            gB.replay()
            torch.sum(flush.view(torch.int64), 0, out=sink)
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); gA.replay(); b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000)
    return statistics.median(ts)


for load, c in bench.SCORED:
    name = f"{load} {c['layer']}"
    if name not in flts:
        continue
    dA = bench.make_input(c, w, 99)
    dB = bench.make_input(c, w, 99)
    gA = bench.capture(lambda: sub.kernel_fn(dA, sA))
    gB = bench.capture(lambda: sub.kernel_fn(dB, sB))
    oA = sub.kernel_fn(dA, sA); oB = sub.kernel_fn(dB, sB); torch.cuda.synchronize()
    eq = all(torch.equal(x, y) for x, y in zip(oA if isinstance(oA, (list, tuple)) else [oA],
                                                oB if isinstance(oB, (list, tuple)) else [oB]))
    res = {"base": [], "warm": [], "warmflush": []}
    with open("/tmp/okc-gpu.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for r in range(rounds):
            for m in ("base", "warm", "warmflush", "warm", "base"):
                res[m].append(run(gA, gB, m))
    med = {m: statistics.median(v) for m, v in res.items()}
    print(f"{name:18s} eq={eq} base {med['base']:7.2f} warm {med['warm']:7.2f} warmflush {med['warmflush']:7.2f} "
          f"| warm-base {med['warm'] - med['base']:+6.2f} us  raw {[round(x, 1) for x in res['base']]} {[round(x, 1) for x in res['warm']]}",
          flush=True)
    del gA, gB, dA, dB
    torch.cuda.empty_cache()
