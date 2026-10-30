"""The judge's 13 tests on other input seeds + scored shapes under judge-like routers on held-out seeds (correctness).
python tools/n11/tests_seeds.py <file> <test seed bases comma> <routers comma> <scored seeds per shape>"""
import sys, torch
sys.path.insert(0, "."); import bench, reference as R
sub = bench.load_sub(sys.argv[1])
W0, st = bench.weights_state(sub)
bad = 0
for base in [int(x) for x in sys.argv[2].split(",")]:
    for i, c in enumerate(bench.TESTS):
        d = bench.make_input(c, W0, base + i); out = sub.kernel_fn(d, st); torch.cuda.synchronize()
        ok, msg = R.check_implementation(d, R.collect(d, out), W0); bad += not ok
        print("PASS" if ok else "FAIL", "test", i, "seed", base + i, msg[:100], flush=True)
        del d, out; getattr(R.generate_input, "cache_clear", lambda: None)(); torch.cuda.empty_cache()
for rs in [int(x) for x in sys.argv[3].split(",") if x]:
    W = dict(W0, **R._router(torch.Generator(device="cuda").manual_seed(rs), "cuda", W0["pre_ffn_norm"]))
    saved, sub.TUNE = sub.TUNE, {}; s2 = sub.prepare(W); sub.TUNE = saved; s2["tune"] = st["tune"]
    for k, (load, c) in enumerate(bench.SCORED):
        for j in range(int(sys.argv[4])):
            d = bench.make_input(c, W, 801 + k + 1000 * j); out = sub.kernel_fn(d, s2); torch.cuda.synchronize()
            ok, msg = R.check_implementation(d, R.collect(d, out), W); bad += not ok
            print("PASS" if ok else "FAIL", "router", rs % 10000, load, c["layer"], 801 + k + 1000 * j, msg[:90], flush=True)
            del d, out; getattr(R.generate_input, "cache_clear", lambda: None)(); torch.cuda.empty_cache()
    del s2, W
print("bad", bad)
