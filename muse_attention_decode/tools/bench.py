"""GPU harness mirroring the judge: the 16 tests against reference.py, and the 12 scored shapes
timed as CUDA-graph replays with a cold L2 before each, median of reps, minus an empty replay.
(Session 76: the cold L2 now comes from READING a 512 MB buffer, as the judge does, and the subtracted replay is a truly
empty graph, as the judge's ~4 us fixed cost is; BENCH_OLD=1 restores the old write-flush / one-tiny-kernel baseline,
which under-counted every shape by ~2.3 us of kernel launch and over-counted DRAM-bound shapes by dirty-line write-backs.)

  python bench.py test                 # correctness, the judge's 16 tests at full size
  python bench.py bench                # the scored shapes with submission._plan as is
  python bench.py sweep '<json>'       # {"shape filter": [plan override, ...]}: time each override
"""
import fcntl, importlib, json, math, os, statistics, sys, torch

OLD = os.environ.get("BENCH_OLD") == "1"

sys.path.insert(0, ".")
import reference
import submission

TESTS = [
    dict(batch=1, ctx=10, heads="tp8", layer="sliding"),
    dict(batch=3, ctx=2047, heads="tp8", layer="sliding"),
    dict(batch=5, ctx=5000, heads="tp8", layer="sliding", q_len=8, spread="mixed", page=1),
    dict(batch=4, ctx=3000, heads="tp8", layer="full"),
    dict(batch=8, ctx=8192, heads="dp", layer="full", spread="mixed"),
    dict(batch=2, ctx=100000, heads="tp8", layer="full", q_len=8, page=1),
    dict(batch=4, ctx=6000, heads="tp8", layer="sliding", dist="edges", page=1),
    dict(batch=4, ctx=6000, heads="tp8", layer="full", q_len=8, dist="edges"),
    dict(batch=1, ctx=1048576, heads="tp8", layer="full", q_len=8),
    dict(batch=3, ctx=4000, heads="tp8", layer="sliding", q_len=8, page=1, kv_scale=0.015625),
    dict(batch=2, ctx=9000, heads="tp8", layer="full", kv_scale=0.015625),
    dict(batch=4, ctx=80000, heads="tp8", layer="full", spread="mixed", dist="soft"),
    dict(batch=4, ctx=6000, heads="tp8", layer="sliding", q_len=8, dist="soft", page=1),
    dict(batch=2, ctx=20000, heads="dp", layer="full", q_len=8, dist="soft"),
    dict(batch=4, ctx=80000, heads="tp8", layer="full", spread="mixed", dist="coherent", page=1),
    dict(batch=2, ctx=20000, heads="dp", layer="full", q_len=8, dist="coherent"),
]
# (load, batch, q_len) x layer; SGLang's H200 times from the judge, for reference.
SCORED = []
SGLANG_H200 = {}
_sg = [20.94, 31.23, 27.25, 200.06, 31.30, 383.65, 22.94, 35.92, 29.17, 204.18, 34.42, 387.94]
for load, batch, q_len in [("decode 1", 1, 1), ("decode 16", 16, 1), ("decode 32", 32, 1),
                           ("verify 1", 1, 8), ("verify 16", 16, 8), ("verify 32", 32, 8)]:
    for layer, dist in (("sliding", "soft"), ("full", "coherent")):
        c = dict(batch=batch, ctx=80000, heads="tp8", layer=layer, page=1, q_len=q_len, dist=dist)
        if batch > 1:
            c["spread"] = "agentic"
        SCORED.append((load, c))
for (load, c), t in zip(SCORED, _sg):
    SGLANG_H200[(load, c["layer"])] = t
WEIGHT = {"sliding": 46, "full": 16}


def test():
    bad = 0
    for i, c in enumerate(TESTS):
        data = reference.generate_input(seed=4242 + i, **c)
        out = submission.kernel_fn(data)
        torch.cuda.synchronize()
        ok, msg = reference.check_implementation(data, reference.collect(data, out))
        bad += not ok
        print(f"{'PASS' if ok else 'FAIL'} {i} {c} {msg[:90]}", flush=True)
        del data
        torch.cuda.empty_cache()
    return bad


_FLUSH = None
_SINK = torch.zeros((), dtype=torch.int64, device="cuda")


def time_graph(fn, reps=40):
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
    # Two agents share this GPU: only one times at a time (compiles and captures run unlocked).
    with open("/tmp/okc-gpu.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for _ in range(reps):
            if OLD:
                _FLUSH.zero_()
            else:
                torch.sum(_FLUSH.view(torch.int64), 0, out=_SINK)
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            g.replay()
            b.record()
            torch.cuda.synchronize()
            times.append(a.elapsed_time(b) * 1000)
    return statistics.median(times)


_EMPTY = None


def empty_cost():
    global _EMPTY
    if _EMPTY is None:
        x = torch.zeros(1, device="cuda")
        _EMPTY = time_graph((lambda: x.add_(1)) if OLD else (lambda: None), reps=100)
    return _EMPTY


_DATA = {}


def data_for(load, c):
    key = (load, c["layer"])
    if key not in _DATA:
        _DATA[key] = reference.generate_input(seed=99, **c)
    return _DATA[key]


def run_shape(load, c, plan=None):
    data = data_for(load, c)
    orig = submission._plan
    if plan:
        submission._plan = lambda cfg, mp, dev: {**orig(cfg, mp, dev), **plan}
    try:
        t = time_graph(lambda: submission.kernel_fn(data)) - empty_cost()
    finally:
        submission._plan = orig
    return t


def bench():
    loads = {}
    for load, c in SCORED:
        t = run_shape(load, c)
        sg = SGLANG_H200[(load, c["layer"])]
        print(f"{load:10s} {c['layer']:8s} {t:8.2f} us   (sglang on H200 {sg:7.2f})", flush=True)
        loads.setdefault(load, [0, 0])
        loads[load][0] += WEIGHT[c["layer"]] * t
        loads[load][1] += WEIGHT[c["layer"]] * sg
    ratios = [sg / me for me, sg in loads.values()]
    print("H100 time vs SGLang-on-H200, geomean over loads:",
          round(math.exp(sum(map(math.log, ratios)) / len(ratios)), 3))


def sweep(spec):
    for flt, plans in spec.items():
        for load, c in SCORED:
            if flt not in f"{load} {c['layer']}":
                continue
            res = []
            for p in plans:
                try:
                    res.append((run_shape(load, c, p), p))
                except Exception as e:  # noqa: BLE001
                    res.append((float("inf"), {**p, "err": str(e)[:60]}))
            res.sort(key=lambda x: x[0])
            print(f"== {load} {c['layer']}", flush=True)
            for t, p in res[:6]:
                print(f"   {t:8.2f} us  {p}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "test":
        sys.exit(test())
    elif cmd == "bench":
        bench()
    elif cmd == "sweep_all":
        import itertools as it
        sl = [dict(block_n=n, warps=w, stages=2, splits=sp, hpp=h, fuse=f) for n, w, sp, h, f in
              it.product((32, 64), (2, 4), (8, 16, 32), (1, 2), (False, True))]
        fu = [dict(block_n=n, warps=w, stages=3, splits=sp, hpp=h, fuse=f) for n, w, sp, h, f in
              it.product((32, 64), (1, 2, 4), (16, 32, 64), (1, 2), (False, True))]
        # The fused merge unrolls over splits: only with few of them (it loses at batch 1 anyway).
        fu1 = [dict(block_n=n, warps=w, stages=3, splits=sp, hpp=h, fuse=False) for n, w, sp, h in
               it.product((32, 64), (1, 2, 4), (64, 128, 256), (1, 2))]
        shapes = sys.argv[2].split(",") if len(sys.argv) > 2 else [
            f"{load} {layer}" for load in ("decode 1", "decode 16", "decode 32", "verify 1", "verify 16", "verify 32")
            for layer in ("sliding", "full")]
        for shape in shapes:
            sweep({shape: sl if shape.endswith("sliding") else fu1 if shape.endswith(" 1 full") else fu})
    else:
        sweep(json.loads(sys.argv[2]))
