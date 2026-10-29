"""Graph-replay A/B equality: capture kernel_fn of file A and file B per scored shape, replay both on the SAME fresh inputs
(cold L2) and print the worst relative L2 difference per output (atomics-order noise is ~1e-6; a wrong expert / stale
routing shows as >= 1e-3). python tools/graph_eq.py <file A> <file B> [n_inputs] [filter] [router_seed]
Each file runs in its own process (two modules in one process share the load_inline .so's host statics, e.g. moe_set_ss):
the children dump outputs to /tmp, the parent compares."""
import sys, torch
sys.path.insert(0, ".")
import bench, reference

import subprocess
fa, fb = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 30
flt = sys.argv[4] if len(sys.argv) > 4 else ""
rs = int(sys.argv[5]) if len(sys.argv) > 5 else 11
if fa != "--child":
    outs = []
    for i, f in enumerate((fa, fb)):
        o = f"/tmp/graph_eq_{i}.pt"
        subprocess.run([sys.executable, __file__, "--child", f, str(n), flt, str(rs), o], check=True)
        outs.append(torch.load(o))
    bad = 0
    for name in outs[0]:
        worst = None
        for i, (a, b) in enumerate(zip(outs[0][name], outs[1][name])):
            r = [float((x - y).norm() / y.norm().clamp_min(1e-30)) for x, y in zip(a, b)]
            worst = r if worst is None else [max(p, q) for p, q in zip(worst, r)]
            if max(r) > 1e-3:
                bad += 1
                print("   BIG", name, i, r)
        print(f"{name:18s} worst rel L2 per output {['%.2e' % x for x in worst]}")
    print("bad replays:", bad)
    sys.exit(0)
dump = sys.argv[6]
w = reference.generate_weights(reference.WEIGHTS_SEED, router_seed=rs)
subs = [bench.load_sub(fb)]
sts = [m.prepare(w) for m in subs]
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
sink = torch.zeros((), dtype=torch.int64, device="cuda")


def capture(m, st, data):
    m.kernel_fn(data, st)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        m.kernel_fn(data, st)
        with torch.cuda.graph(g, stream=s):
            out = m.kernel_fn(data, st)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    return g, out


def flat(o):
    return [t for t in (o if isinstance(o, (tuple, list)) else (o,)) if isinstance(t, torch.Tensor)]


res = {}
for load, c in bench.SCORED:
    name = f"{load} {c['layer']}"
    if flt not in name:
        continue
    datas = [bench.make_input(c, w, 7000)]
    gs = [capture(m, st, d) for m, st, d in zip(subs, sts, datas)]
    res[name] = []
    for i in range(n):
        d = bench.make_input(c, w, 9001 + 31 * i)
        outs = []
        for (g, out), data in zip(gs, datas):
            for k, v in d.items():
                if isinstance(v, torch.Tensor):
                    data[k].copy_(v)
            torch.sum(flush.view(torch.int64), 0, out=sink)
            g.replay()
            torch.cuda.synchronize()
            outs.append([t.float().cpu() for t in flat(out)])
        res[name].append(outs[0])
        del d
    del gs, datas
    torch.cuda.empty_cache()
torch.save(res, dump)
