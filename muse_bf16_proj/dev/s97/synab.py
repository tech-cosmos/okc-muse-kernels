import sys, statistics
sys.path.insert(0, ".")
import bench
sub = bench.load_sub(sys.argv[1]); rounds = int(sys.argv[2])
sub.TUNE = {}
w, st = bench.weights_state(sub)
dev = st["unpool_slice"].device
data = {}
for sl, ctx in ((False, 80000), (True, 3000)):
    data[f"synth32 {'sl' if sl else 'full'}"] = sub._synth_input(32, dev, ctx, 1, sl, st["router"])
for l, c in bench.SCORED:
    if l == "decode 32":
        data[f"real {l} {c['layer']}"] = bench.make_input(c, w, 99)
stm = ["TUNE_BASE[(32,False)].pop('g30n:unpool',None); TUNE_BASE[(32,True)].pop('g30n:unpool',None)",
       "TUNE_BASE[(32,False)]['g30n:unpool']=True; TUNE_BASE[(32,True)]['g30n:unpool']=True"]
res = {(n, i): [] for n in data for i in range(2)}
e0 = bench.empty_cost()
for r in range(rounds):
    for i, s in enumerate(stm if r % 2 == 0 else stm[::-1]):
        i = stm.index(s); exec(s, vars(sub))
        for n in data:
            res[(n, i)].append(bench.time_graph(lambda: sub.kernel_fn(data[n], st)) - e0)
for n in data:
    print(f"{n:28s}", "  ".join(f"[{i}] {statistics.median(res[(n, i)]):7.2f} ({min(res[(n, i)]):7.2f})" for i in range(2)), flush=True)
