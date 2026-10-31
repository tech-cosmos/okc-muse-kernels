"""Local harness for muse_bf16_proj, mirroring the judge.

  python bench.py test            # the 9 judge tests + the 30 scored shapes, against reference.py
  python bench.py esc             # adversarial escapes: 12-bit compression round trip + outputs
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
SGLANG_H200 = {('decode 1', 'qkv_gate'): 14.24, ('decode 1', 'o'): 6.66, ('decode 1', 'pooler'): 21.66, ('decode 1', 'unpooler'): 6.98, ('decode 1', 'router'): 10.59, ('decode 16', 'qkv_gate'): 14.19, ('decode 16', 'o'): 6.67, ('decode 16', 'pooler'): 21.34, ('decode 16', 'unpooler'): 18.69, ('decode 16', 'router'): 14.82, ('decode 32', 'qkv_gate'): 14.5, ('decode 32', 'o'): 7.01, ('decode 32', 'pooler'): 21.63, ('decode 32', 'unpooler'): 18.78, ('decode 32', 'router'): 15.31, ('verify 1', 'qkv_gate'): 14.46, ('verify 1', 'o'): 6.69, ('verify 1', 'pooler'): 21.31, ('verify 1', 'unpooler'): 7.15, ('verify 1', 'router'): 26.3, ('verify 16', 'qkv_gate'): 17.15, ('verify 16', 'o'): 7.17, ('verify 16', 'pooler'): 24.78, ('verify 16', 'unpooler'): 20.08, ('verify 16', 'router'): 16.51, ('verify 32', 'qkv_gate'): 19.87, ('verify 32', 'o'): 9.22, ('verify 32', 'pooler'): 30.67, ('verify 32', 'unpooler'): 24.16, ('verify 32', 'router'): 17.73}  # check c77613c2

_W = None
_STATE = None


def state():
    global _W, _STATE
    if _STATE is None:
        _W = reference.generate_weights(reference.WEIGHTS_SEED)
        _STATE = submission.prepare(_W)
        torch.cuda.synchronize()
        _force(_STATE)
    return _W, _STATE


def _force(st):
    """FORCE=g22: use the first packed gemv22 config at m 128/256 (correctness tests of a kernel the tuner did not pick)."""
    if os.environ.get("FORCE") == "g22":
        for layer, cfgs in submission.BF16_G22.items():
            for c in cfgs:
                if ("g22", layer, c[1], c[3]) in st and (layer, c[0]) not in st.get("_forced", set()):
                    st["pick"][(layer, c[0])] = ("g22", c)
                    st.setdefault("_forced", set()).add((layer, c[0]))
        print("forced:", sorted((k, v) for k, v in st["pick"].items() if v[0] == "g22"))
    if os.environ.get("FORCE") == "gx":  # n19: g30 with G30_GX=1 escape records (tuner candidate the H100 never picks)
        for (layer, m), cfgs in submission.BF16_G30.items():
            for c in cfgs:
                if len(c) > 3 and ("g30gx", layer, c[0]) in st:
                    st["pick"][(layer, m)] = ("g30", (m, *c))
        print("forced:", sorted((k, v) for k, v in st["pick"].items() if v[0] == "g30" and len(v[1]) > 4))


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


def test_esc():
    """Weights with nasty escapes (zeros, -0, BF16 denormals, values far below the 12-bit window, just below it), 2-3 per row in 3/4 of the rows: prepare() must round-trip them bit-exactly (else fall back) and outputs must pass."""
    global _W, _STATE
    w = reference.generate_weights(reference.WEIGHTS_SEED)
    g = torch.Generator(device="cuda").manual_seed(7)
    C = {}  # escapes go to 32 fixed columns per layer, so spike rows of x can probe them exactly
    for name in ["qkv_gate", "o", "pooler", "unpooler"]:
        t = w[name]
        n, k = t.shape
        mx = t.float().abs().max().item()
        vals = torch.tensor([0.0, -0.0, 2.0 ** -133, -3 * 2.0 ** -131, 2.0 ** -100, -1.37 * 2.0 ** -60,
                             mx * 2.0 ** -17, -mx * 1.9 * 2.0 ** -18, 1.5 * 2.0 ** -126], device="cuda").to(torch.bfloat16)
        if os.environ.get("ESCV") == "bf16":  # n19: only escapes g30 keeps (BF16-exact, normal): exercises g30/GX escape paths
            vals = torch.tensor([0.0, -0.0, 2.0 ** -100, -1.37 * 2.0 ** -60, mx * 2.0 ** -17, -mx * 1.9 * 2.0 ** -18],
                                device="cuda").to(torch.bfloat16)
        C[name] = torch.randperm(k, device="cuda", generator=g)[:32]
        for j in range(2 if k == 8192 else 3):  # + natural escapes (up to 6 per row at k = 8192) <= G13_ESC = 8
            rows = torch.arange(n, device="cuda")[torch.rand(n, device="cuda", generator=g) < 0.75]
            cols = C[name][torch.randint(0, 32, (len(rows),), device="cuda", generator=g)]
            t[rows, cols] = vals[torch.randint(0, len(vals), (len(rows),), device="cuda", generator=g)]
    from_sub = submission._compress
    for name in ["qkv_gate", "o", "pooler", "unpooler"]:
        print(name, "compressed" if from_sub(w[name]) is not None else "FALLBACK (uncompressed)", flush=True)
    _W, _STATE = w, submission.prepare(w)
    torch.cuda.synchronize()
    _force(_STATE)
    bad = 0
    for i, (m, layer) in enumerate([(m, l) for m in (1, 8, 16, 32) + ((128, 256) if os.environ.get("FORCE") else ()) for l in LAYERS[:4]]):
        data = reference.generate_input(m, layer, seed=99 + i, dist="normal")
        sp = list(range(0, m, max(1, m // 8)))  # spike rows: x = 8 at an escape column -> out = 8 * W[:, col] exactly
        for j, r in enumerate(sp):
            data["x"][r] = 0
            data["x"][r, C[layer][(j + i) % 32]] = 8
        out = submission.kernel_fn(data, _STATE)
        torch.cuda.synchronize()
        ok, msg = reference.check_implementation(data, out, w)
        ref = reference.ref_kernel(data, w)
        exact = torch.equal(out[sp].view(torch.int16), ref[sp].view(torch.int16))
        pk = _STATE["pick"].get((layer, m)) or ("", ())
        gx = pk[0] == "g30" and len(pk[1]) > 4  # n19: GX fixups = FP32 2^-15 + (v - 2^-15) (lead 10-25 23:30Z)
        if (m >= 128 or gx) and not exact:  # gemv22 adds escape fixups by separate FP32 red.adds (ordinary rounding): near-exact
            exact = bool(((out[sp].float() - ref[sp].float()).abs() <= ref[sp].float().abs() * 2 ** -7 + 1e-12).all())
        ok = ok and exact
        bad += not ok
        print(f"{'PASS' if ok else 'FAIL'} esc m={m} {layer}: spike rows exact {exact}; {msg}", flush=True)
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
    if cmd == "esc":
        sys.exit(test_esc())
    bench(sys.argv[2] if len(sys.argv) > 2 else "")
