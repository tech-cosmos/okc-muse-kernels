"""Local harness for muse_decode_layer (H100; SGLang absent, so the experts run a dequantizing fallback).

  python bench.py test [file]        # the judge's 13 tests against reference.py
  python bench.py bench [file] [filter]  # scored shapes: CUDA-graph replay, cold L2 (read 512 MB), median, minus empty replay
  python bench.py prof [file] filter # torch profiler kernel table of one graph replay of the matching shapes
Default file: submission.py (build with `python build.py`)."""
import fcntl, importlib.util, math, statistics, sys, torch

sys.path.insert(0, ".")
import reference

TESTS = [
    dict(batch=1, ctx=10, layer="sliding"),
    dict(batch=1, ctx=3000, layer="full"),
    dict(batch=8, ctx=2047, layer="sliding"),
    dict(batch=9, ctx=5000, layer="sliding", spread="mixed", page=1),
    dict(batch=3, ctx=6000, layer="full", q_len=8, dist="outliers"),
    dict(batch=5, ctx=5000, layer="sliding", q_len=8, spread="mixed", page=1),
    dict(batch=33, ctx=20000, layer="full", spread="mixed", page=1),
    dict(batch=2, ctx=300000, layer="sliding", q_len=8),
    dict(batch=1, ctx=1048576, layer="full", q_len=8),
    dict(batch=4, ctx=7000, layer="sliding", q_len=8, page=1, kv_scale=0.015625),
    dict(batch=1, ctx=80000, layer="full", dist="soft", page=1),
    dict(batch=4, ctx=6000, layer="sliding", q_len=8, dist="soft"),
    dict(batch=16, ctx=20000, layer="full", spread="mixed", dist="soft", page=1),
]
SCORED = []
for load, batch, q_len in [("decode 1", 1, 1), ("decode 16", 16, 1), ("decode 32", 32, 1),
                           ("verify 1", 1, 8), ("verify 16", 16, 8), ("verify 32", 32, 8)]:
    for layer in ("sliding", "full"):
        c = dict(batch=batch, ctx=80000, layer=layer, page=1, q_len=q_len, dist="soft")
        if batch > 1:
            c["spread"] = "agentic"
        SCORED.append((load, c))
# SGLang on the judge's H200 (run 53d9d188), us.
SGLANG_H200 = dict(zip([f"{l} {c['layer']}" for l, c in SCORED],
                       [172.72, 183.10, 265.55, 437.74, 261.33, 612.56, 266.83, 272.30, 506.48, 692.02, 723.60, 1087.28]))
WEIGHT = {"sliding": 46, "full": 16}


def load_sub(path):
    spec = importlib.util.spec_from_file_location("sub", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_W = {}


def weights_state(sub):
    if "w" not in _W:
        _W["w"] = reference.generate_weights(reference.WEIGHTS_SEED)
        _W["s"] = sub.prepare(_W["w"])
    return _W["w"], _W["s"]


def make_input(c, weights, seed):
    ls = reference.case_fields({**c, "seed": seed}, weights)["local_start"]
    getattr(reference.generate_input, "cache_clear", lambda: None)()
    return reference.generate_input(**c, seed=seed, local_start=ls)


def test(sub):
    weights, state = weights_state(sub)
    bad = 0
    for i, c in enumerate(TESTS):
        data = make_input(c, weights, 4242 + i)
        out = sub.kernel_fn(data, state)
        torch.cuda.synchronize()
        ok, msg = reference.check_implementation(data, reference.collect(data, out), weights)
        bad += not ok
        print(f"{'PASS' if ok else 'FAIL'} {i} {c} {msg[:160]}", flush=True)
        del data, out
        getattr(reference.generate_input, "cache_clear", lambda: None)()
        torch.cuda.empty_cache()
    return bad


_FLUSH = None
_SINK = None


def capture(fn):
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
    return g


def time_graph(fn, reps=40):
    global _FLUSH, _SINK
    if _FLUSH is None:
        _FLUSH = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
        _SINK = torch.zeros((), dtype=torch.int64, device="cuda")
    g = capture(fn)
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
    return statistics.median(times)


_EMPTY = []


def empty_cost():
    if not _EMPTY:
        _EMPTY.append(time_graph(lambda: None, reps=100))
    return _EMPTY[0]


def bench(sub, flt=""):
    weights, state = weights_state(sub)
    loads = {}
    for load, c in SCORED:
        name = f"{load} {c['layer']}"
        if flt not in name:
            continue
        data = make_input(c, weights, 99)
        t = time_graph(lambda: sub.kernel_fn(data, state)) - empty_cost()
        sg = SGLANG_H200[name]
        print(f"{name:16s} {t:8.2f} us   (sglang H200 {sg:8.2f})", flush=True)
        loads.setdefault(load, [0, 0])
        loads[load][0] += WEIGHT[c["layer"]] * t
        loads[load][1] += WEIGHT[c["layer"]] * sg
        del data
        getattr(reference.generate_input, "cache_clear", lambda: None)()
    ratios = [sg / me for me, sg in loads.values()]
    print("geomean SGLang-H200 / ours-H100:", round(math.exp(sum(map(math.log, ratios)) / len(ratios)), 3))


def prof(sub, flt):
    from torch.profiler import profile, ProfilerActivity
    weights, state = weights_state(sub)
    for load, c in SCORED:
        name = f"{load} {c['layer']}"
        if flt not in name:
            continue
        data = make_input(c, weights, 99)
        g = capture(lambda: sub.kernel_fn(data, state))
        with open("/tmp/okc-gpu.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with profile(activities=[ProfilerActivity.CUDA]) as p:
                for _ in range(5):
                    g.replay()
                torch.cuda.synchronize()
        print("==", name)
        print(p.key_averages().table(sort_by="cuda_time_total", row_limit=30, max_name_column_width=70))


if __name__ == "__main__":
    cmd = sys.argv[1]
    sub = load_sub(sys.argv[2] if len(sys.argv) > 2 else "submission.py")
    if cmd == "test":
        sys.exit(test(sub))
    elif cmd == "bench":
        bench(sub, sys.argv[3] if len(sys.argv) > 3 else "")
    elif cmd == "prof":
        prof(sub, sys.argv[3])
