# bitwise: outputs with hot=32 vs hot=H on T=256 inputs (sliding + full), several routers
import sys, os, torch
os.chdir("/home/daytona/r-layer/muse_decode_layer"); sys.path.insert(0, ".")
import reference as R, bench
sub = bench.load_sub(sys.argv[1]); sub.TUNE = {}
H = int(sys.argv[2])
for rs in [int(x) for x in sys.argv[3].split(",")]:
    W = R.generate_weights(R.WEIGHTS_SEED, router_seed=1000 + rs); st = sub.prepare(W)
    for layer in ("sliding", "full"):
        c = dict(batch=32, ctx=80000 if layer == "sliding" else 20000, layer=layer, page=1, q_len=8, dist="soft", spread="agentic")
        for s in (1, 2):
            d = bench.make_input(c, W, s); outs = []
            for h in (32, H, 32, H):
                for k in ((256, True), (256, False)): sub.TUNE_BASE[k] = dict(sub.TUNE_BASE.get(k, {}), hot=h)
                o = sub.kernel_fn(d, st); torch.cuda.synchronize(); outs.append([t.clone() for t in o])
            eq = [all(torch.equal(a, b) for a, b in zip(outs[0], x)) for x in outs[1:]]
            ok, msg = R.check_implementation(d, R.collect(d, sub.kernel_fn(d, st)), W)
            print(f"router {rs} {layer} in {s}: equal-to-hot32 {eq} check {ok} {msg[:80]}", flush=True)
            del d, outs; getattr(R.generate_input, "cache_clear", lambda: None)(); torch.cuda.empty_cache()
    del st, W; torch.cuda.empty_cache()
