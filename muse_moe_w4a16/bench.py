"""Local harness for muse_moe_w4a16 (moe agent).

  python bench.py test              # the judge's 5 tests + the 6 scored shapes, against reference.py
  python bench.py bench [reps]      # scored shapes as CUDA-graph replays, cold L2 (512 MB read), median
  python bench.py sweep '<json>'    # {"bs": [[p13, p2], ...]}: time plan overrides for that bs

Timing holds /tmp/okc-gpu.lock only around the replay loop. Several routings (seeds) per shape,
since the judge draws fresh routing every repetition.
"""
import fcntl, json, math, statistics, sys, time, torch

sys.path.insert(0, ".")
import reference
import submission

TESTS = [dict(bs=4), dict(bs=33), dict(bs=257), dict(bs=128, dist="skewed"), dict(bs=1)]
SCORED = [("decode 1", 1), ("decode 16", 16), ("decode 32", 32), ("verify 1", 8), ("verify 16", 128), ("verify 32", 256)]
# SGLang on H200 from the judge (job 97f135310fc24612), us.
SGLANG_H200 = {1: 80.16, 16: 187.97, 32: 210.29, 8: 143.28, 128: 475.94, 256: 613.41}
FLOOR = {1: 11.4, 16: 52.5, 32: 70.8, 8: 37.3, 128: 101.3, 256: 108.9}

_W = None
_ST = None


def state():
    global _W, _ST
    if _ST is None:
        t = time.time()
        _W = reference.generate_weights(reference.WEIGHTS_SEED)
        _ST = submission.prepare(_W)
        print(f"prepare {time.time() - t:.1f}s", flush=True)
    return _W, _ST


def test():
    w, st = state()
    bad = 0
    cases = [dict(c, seed=1000 + i) for i, c in enumerate(TESTS)] + [dict(bs=b, seed=2000 + b) for _, b in SCORED]
    for c in cases:
        data = reference.generate_input(sharding="ep8", **c)
        out = submission.kernel_fn(data, st)
        torch.cuda.synchronize()
        ok, msg = reference.check_implementation(data, out, w)
        bad += not ok
        print(f"{'PASS' if ok else 'FAIL'} {c} {msg}", flush=True)
    return bad


_FLUSH = None
_SINK = torch.zeros((), dtype=torch.int64, device="cuda")


def time_graph(fn, reps=50):
    global _FLUSH
    if _FLUSH is None:
        _FLUSH = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
        with torch.cuda.graph(g, stream=s):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    times = []
    with open("/tmp/okc-gpu.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for _ in range(reps):
            torch.sum(_FLUSH.view(torch.int64), 0, out=_SINK)
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            g.replay()
            b.record()
            torch.cuda.synchronize()
            times.append(a.elapsed_time(b) * 1000)
    del g
    return statistics.median(times)


_EMPTY = None


def empty_cost():
    global _EMPTY
    if _EMPTY is None:
        _EMPTY = time_graph(lambda: None, reps=100)
    return _EMPTY


def run_shape(bs, plan=None, seeds=(7, 8, 9), reps=30):
    _, st = state()
    ts = []
    old = submission.PLAN_OVERRIDE
    if plan:
        submission.PLAN_OVERRIDE = lambda M: tuple(plan)
    try:
        for sd in seeds:
            data = reference.generate_input(bs=bs, sharding="ep8", seed=sd)
            ts.append(time_graph(lambda: submission.kernel_fn(data, st), reps) - empty_cost())
    finally:
        submission.PLAN_OVERRIDE = old
    return statistics.median(ts)


def bench(reps=30):
    ratios = []
    for load, bs in SCORED:
        t = run_shape(bs, reps=reps)
        sg = SGLANG_H200[bs]
        ratios.append(sg / t)
        print(f"{load:10s} bs {bs:3d} {t:8.2f} us   sglang(H200) {sg:7.2f}  x{sg / t:5.2f}  floor {FLOOR[bs]}", flush=True)
    print("geomean vs SGLang-on-H200:", round(math.exp(sum(map(math.log, ratios)) / len(ratios)), 3))


def sweep(spec):
    for bs, plans in spec.items():
        res = []
        for p in plans:
            try:
                res.append((run_shape(int(bs), p), p))
            except Exception as e:  # noqa: BLE001
                res.append((float("inf"), f"{p} {e}"))
        res.sort(key=lambda r: r[0])
        print(f"== bs {bs}")
        for t, p in res:
            print(f"   {t:8.2f} us  {p}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "test":
        sys.exit(test())
    elif cmd == "bench":
        bench(int(sys.argv[2]) if len(sys.argv) > 2 else 30)
    elif cmd == "sweep":
        sweep(json.loads(sys.argv[2]))
