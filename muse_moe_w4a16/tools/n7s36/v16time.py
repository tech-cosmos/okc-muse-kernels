# n7s36: v16 (T=128) / any-T layer time (H100, graph replay, cold L2) vs routing structure, per router seed
# python v16time.py <submission.py> <routers,..> <inputs,..>   env BATCH (16) QLEN (8) LAYER (sliding)
import sys, os, torch
os.chdir("/home/daytona/r-layer/muse_decode_layer"); sys.path.insert(0, ".")
import reference as R, bench
sub = bench.load_sub(sys.argv[1])
sub.TUNE = {}
import json
TB = json.loads(os.environ.get("TB", "{}"))
for k in list(sub.TUNE_BASE):
    if k[0] == int(os.environ.get("BATCH", 16)) * int(os.environ.get("QLEN", 8)): sub.TUNE_BASE[k] = dict(sub.TUNE_BASE[k], **TB)
print("TB", TB, flush=True)
B, Q = int(os.environ.get("BATCH", 16)), int(os.environ.get("QLEN", 8))
c = dict(batch=B, ctx=80000, layer=os.environ.get("LAYER", "sliding"), page=1, q_len=Q, dist="soft", spread="agentic")
for rs in [int(x) for x in sys.argv[2].split(",")]:
    W = R.generate_weights(R.WEIGHTS_SEED, router_seed=1000 + rs)
    st = sub.prepare(W)
    for s in [int(x) for x in sys.argv[3].split(",")]:
        d = bench.make_input(c, W, s)
        ids = R.ref_front(d, W)["biased"].topk(8, dim=-1).indices
        ls = int(d["local_start"]); loc = ids[(ids >= ls) & (ids < ls + 32)] - ls
        cnt = sorted(torch.bincount(loc, minlength=32).tolist(), reverse=True)
        t = bench.time_graph(lambda: sub.kernel_fn(d, st)) - bench.empty_cost()
        b = [sum(lo <= x <= hi for x in cnt) for lo, hi in ((1, 8), (9, 16), (17, 32), (33, 64), (65, 128), (129, 9999))]
        print(f"router {rs} in {s}: {t:7.1f} us routes {sum(cnt)} distinct {sum(x>0 for x in cnt)} "
              f"b[1-8,9-16,17-32,33-64,65-128,>128]={b} top {cnt[:6]}", flush=True)
        del d; getattr(R.generate_input, "cache_clear", lambda: None)(); torch.cuda.empty_cache()
    del st, W; torch.cuda.empty_cache()
