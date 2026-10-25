"""Local harness for muse_bf16_proj, mirroring the judge.

  python bench.py test            # the 9 judge tests + the 30 scored shapes, against reference.py
  python bench.py bench [filter]  # scored shapes as CUDA-graph replays, cold L2 (read 512 MB), median,
                                  # minus an empty replay; the lock /tmp/okc-gpu.lock only around timing
SUB=other.py python bench.py ...  selects another submission file.
"""
import fcntl, importlib.util, math, os, statistics, sys, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference

_spec = importlib.util.spec_from_file_location("submission", os.environ.get("SUB", "submission.py"))
submission = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(submission)

TESTS = [(1, "qkv_gate", "normal"), (7, "pooler", "normal"), (33, "o", "normal"),
         (130, "unpooler", "normal"), (5, "unpooler", "normal"), (8, "router", "normal"),
         (9, "router", "outliers"), (512, "qkv_gate", "outliers"), (1000, "unpooler", "outliers")]
LAYERS = ["qkv_gate", "o", "pooler", "unpooler", "router"]
LOADS = [("decode 1", 1), ("decode 16", 16), ("decode 32", 32), ("verify 1", 8), ("verify 16", 128), ("verify 32", 256)]
# SGLang's H200 per-shape times (us) from the judge; fill in from the first check.
SGLANG_H200 = {}

_W = None
_STATE = None


def state():
    global _W, _STATE
    if _STATE is None:
        _W = reference.generate_weights(reference.WEIGHTS_SEED)
        _STATE = submission.prepare(_W)
        torch.cuda.synchronize()
    return _W, _STATE


def test():
    w, st = state()
    bad = 0
    cases = [(m, l, d) for m, l, d in TESTS] + [(m, l, "normal") for _, m in LOADS for l in LAYERS]
    for i, (m, layer, dist) in enumerate(cases):
        data = reference.generate_input(m, layer, seed=1234 + i, dist=dist)
        out = submission.kernel_fn(data, st)
        torch.cuda.synchronize()
        ok, msg = reference.check_implementation(data, out, w)
        bad += not ok
        print(f"{'PASS' if ok else 'FAIL'} m={m} {layer} {dist}: {msg[:110]}", flush=True)
    return bad


_FLUSH = None
_SINK = torch.zeros((), dtype=torch.int64, device="cuda")


def time_graph(fn, reps=100):
    global _FLUSH
    if _FLUSH is None:
        _FLUSH = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
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
    return statistics.median(times)


_EMPTY = None


def empty_cost():
    global _EMPTY
    if _EMPTY is None:
        _EMPTY = time_graph(lambda: None, reps=200)
    return _EMPTY


def bench(flt=""):
    w, st = state()
    ratios = []
    for load, m in LOADS:
        mine = sg = 0.0
        have_sg = True
        for layer in LAYERS:
            if flt and flt not in f"{load} {layer}":
                continue
            data = reference.generate_input(m, layer, seed=99)
            t = time_graph(lambda: submission.kernel_fn(data, st)) - empty_cost()
            n, k = data["config"]["n"], data["config"]["k"]
            floor = reference.speed_of_light_ns(m, layer, {"hbm_bytes_per_s": 3.35e12, "bf16_dense_flops_per_s": 989e12}) / 1e3
            s = SGLANG_H200.get((load, layer))
            print(f"{load:10s} {layer:9s} m={m:4d} n={n:5d} k={k:5d} {t:7.2f} us  (H100 floor {floor:5.2f}"
                  f"{'' if s is None else f', sglang H200 {s:6.2f}'})", flush=True)
            mine += 62 * t
            if s is None:
                have_sg = False
            else:
                sg += 62 * s
        if have_sg and mine:
            ratios.append(sg / mine)
    if ratios and len(ratios) == len(LOADS):
        print("vs SGLang-on-H200 geomean:", round(math.exp(sum(map(math.log, ratios)) / len(ratios)), 3))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "test":
        sys.exit(test())
    bench(sys.argv[2] if len(sys.argv) > 2 else "")
