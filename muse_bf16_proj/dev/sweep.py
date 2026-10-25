"""python dev/sweep.py 'm:layer' '[plan, ...]'   time cand.kernel_fn with each plan (PJ_CONFIGS must include them)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SUB", "dev/cand.py")
import bench, reference
c = bench.submission
w, st = bench.state()
for shape in sys.argv[1].split(","):
    m, layer = shape.split(":"); m = int(m)
    d = reference.generate_input(m, layer, seed=99)
    ref = None
    for p in json.loads(sys.argv[2]):
        c.PLAN[(m, layer)] = tuple(p)
        try:
            out = c.kernel_fn(d, st)
            ok, msg = reference.check_implementation(d, out, w)
            t = bench.time_graph(lambda: c.kernel_fn(d, st)) - bench.empty_cost()
            print(f"{m:4d} {layer:9s} {p} {t:7.2f} us {'' if ok else 'FAIL ' + msg[:60]}", flush=True)
        except Exception as e:
            print(f"{m:4d} {layer:9s} {p} ERR {str(e)[:80]}", flush=True)
