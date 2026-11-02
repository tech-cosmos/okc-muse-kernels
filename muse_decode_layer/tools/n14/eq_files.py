"""layer n14: outputs of two submission files on the same inputs (eager): ids equal? max |hidden diff|.
python tools/n14/eq_files.py <filter> <n_inputs> <router_seeds> <file A> <file B>"""
import sys, importlib.util, torch
sys.path.insert(0, ".")
import bench, reference as R
flt, n, rss, fa, fb = sys.argv[1], int(sys.argv[2]), [int(x) for x in sys.argv[3].split(",")], sys.argv[4], sys.argv[5]
mods = []
for i, f in enumerate((fa, fb)):
    spec = importlib.util.spec_from_file_location(f"s{i}", f); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); mods.append(m)
for rs in rss:
    w = R.generate_weights(R.WEIGHTS_SEED, router_seed=rs)
    sts = [m.prepare(w) for m in mods]
    for load, c in bench.SCORED:
        name = f"{load} {c['layer']}"
        if flt not in name:
            continue
        neq = nfail = 0; hd = 0.0; saved = 0
        for i in range(n):
            d = bench.make_input(c, w, 5001 + 13 * i + 1000 * rs)
            o = [m.kernel_fn(d, st) for m, st in zip(mods, sts)]
            torch.cuda.synchronize()
            ia, ib = o[0][2], o[1][2]
            neq += not torch.equal(ia, ib)
            hd = max(hd, float((o[0][0].float() - o[1][0].float()).abs().max()))
            fr = R.ref_front(d, w); ls = fr["local_start"]
            nfail += R.check_routing(ia, fr["biased"], ls) is not None
            saved += R._local_load(fr["biased"].topk(8, dim=-1).indices, ls)[1] - R._local_load(ia.long(), ls)[1]
        print(f"rs {rs} {name:18s} ids differ {neq}/{n}  max|dh| {hd:.3g}  A fails {nfail}  A saved {saved / n:.2f}", flush=True)
