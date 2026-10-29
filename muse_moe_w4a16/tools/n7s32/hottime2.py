# v32 sliding layer time (H100, graph replay, cold L2) vs routing structure, per router seed
import sys, os, torch
os.chdir("/home/daytona/r-layer/muse_decode_layer"); sys.path.insert(0, ".")
import reference as R, bench
sub = bench.load_sub(sys.argv[1] if len(sys.argv) > 1 else "submission.py")
sub.TUNE = {}
H = os.environ.get("HOT")
if H is not None:
    for k in ((256, True), (256, False)): sub.TUNE_BASE[k] = dict(sub.TUNE_BASE.get(k, {}), hot=int(H))
print("base256", sub.TUNE_BASE.get((256, True)))
c = dict(batch=32, ctx=80000, layer=os.environ.get("LAYER", "sliding"), page=1, q_len=8, dist="soft", spread="agentic")
for rs in [int(x) for x in sys.argv[2].split(",")]:
    W = R.generate_weights(R.WEIGHTS_SEED, router_seed=1000 + rs)
    st = sub.prepare(W)
    for s in [int(x) for x in sys.argv[3].split(",")]:
        d = bench.make_input(c, W, s)
        ids = R.ref_front(d, W)["biased"].topk(8, dim=-1).indices
        ls = int(d["local_start"]); loc = ids[(ids >= ls) & (ids < ls + 32)] - ls
        cnt = sorted(torch.bincount(loc, minlength=32).tolist(), reverse=True)
        nh = sum(65 <= x <= 128 for x in cnt); big = sum(x > 128 for x in cnt)
        t = bench.time_graph(lambda: sub.kernel_fn(d, st)) - bench.empty_cost()
        print(f"router {rs} in {s}: {t:7.1f} us  nh{nh}+{big} routes {sum(cnt)} distinct {sum(x>0 for x in cnt)} top {cnt[:5]}", flush=True)
        del d; getattr(R.generate_input, "cache_clear", lambda: None)(); torch.cuda.empty_cache()
    del st, W; torch.cuda.empty_cache()
