"""Nearby (non-scored) shapes: correctness + H100 graph time, to spot cliffs between the scored token counts.
python tools/n11/nearby.py <file>"""
import sys, torch
sys.path.insert(0, "."); import bench, reference as R
sub = bench.load_sub(sys.argv[1])
W, st = bench.weights_state(sub)
CASES = [(b, q, ctx, l) for l in ("sliding", "full") for (b, q, ctx) in
         [(1, 1, 80000), (2, 1, 80000), (4, 1, 80000), (8, 1, 80000), (16, 1, 80000), (24, 1, 80000), (32, 1, 80000), (48, 1, 80000), (64, 1, 80000),
          (1, 8, 80000), (2, 8, 80000), (4, 8, 80000), (8, 8, 80000), (16, 8, 80000), (24, 8, 80000), (32, 8, 80000),
          (1, 4, 80000), (16, 4, 80000), (16, 1, 40000), (16, 1, 120000), (32, 8, 40000), (1, 1, 200000)]]
for b, q, ctx, l in CASES:
    c = dict(batch=b, ctx=ctx, layer=l, page=1, q_len=q, dist="soft")
    if b > 1:
        c["spread"] = "agentic"
    try:
        d = bench.make_input(c, W, 77)
        out = sub.kernel_fn(d, st); torch.cuda.synchronize()
        ok, msg = R.check_implementation(d, R.collect(d, out), W)
        t = bench.time_graph(lambda: sub.kernel_fn(d, st), reps=20) - bench.empty_cost()
        print(f"{'PASS' if ok else 'FAIL'} b{b:3d} q{q} ctx{ctx:7d} {l:7s} T={b*q:4d} {t:8.1f} us {'' if ok else msg[:80]}", flush=True)
    except Exception as ex:
        print(f"ERR  b{b} q{q} ctx{ctx} {l}: {type(ex).__name__} {str(ex)[:120]}", flush=True)
    d = out = None; getattr(R.generate_input, "cache_clear", lambda: None)(); torch.cuda.empty_cache()
