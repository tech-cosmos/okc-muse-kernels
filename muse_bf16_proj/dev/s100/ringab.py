"""proj s100: in-layer A/B of the g30 ring config (NW, NS) at T=32: g30n unpool and g30k pooler, swapped in one prepared state.
python ringab.py <sub with TUNE={} and G30N_DEFAULT_ON={("unpool",32)}> <rounds>; run from a r-layer muse_decode_layer/ dir."""
import sys, statistics
sys.path.insert(0, ".")
import bench
sub = bench.load_sub(sys.argv[1]); R = int(sys.argv[2])
w, st = bench.weights_state(sub)
T = 32
n_old = st["g30n"][("unpool", T)]
k_old = st["g30k"][("pooler", T)][104]
n16 = sub._g30n_build((T, 1, 16, 16)); k16 = sub._g30k_build((T, 16, 16))
V = {"unp20/8": (n_old, None), "unp16/16": ((n16,) + n_old[1:], None),
     "unp16+k104_16": ((n16,) + n_old[1:], (k16,) + k_old[1:]), "unp20+k104_20": (n_old, k_old)}
names = [(f"{l} {c['layer']} s{sd}", c, sd) for l, c in bench.SCORED if l == f"decode {T}" for sd in (99, 7)]
data = {nm: bench.make_input(c, w, sd) for nm, c, sd in names}
e0 = bench.empty_cost()
res = {}
keys = list(V)
for r in range(R):
    for v in (keys if r % 2 == 0 else keys[::-1]):
        ne, ke = V[v]
        st["g30n"][("unpool", T)] = ne
        if ke is None:
            sub.G30K_DEFAULT = {}
        else:
            st["g30k"][("pooler", T)][104] = ke
            sub.G30K_DEFAULT = {("pooler", T): 104}
        for nm, c, sd in names:
            res.setdefault((nm, v), []).append(bench.time_graph(lambda: sub.kernel_fn(data[nm], st)) - e0)
for nm, c, sd in names:
    print(f"{nm:22s}", "  ".join(f"{v} {statistics.median(res[(nm, v)]):7.2f}" for v in keys), flush=True)
# bitwise: every variant gives the same outputs on every input
import torch
for nm, c, sd in names:
    outs = []
    for v in keys:
        ne, ke = V[v]
        st["g30n"][("unpool", T)] = ne
        sub.G30K_DEFAULT = {} if ke is None else {("pooler", T): 104}
        if ke is not None:
            st["g30k"][("pooler", T)][104] = ke
        o = sub.kernel_fn(data[nm], st); torch.cuda.synchronize()
        outs.append([t.clone() for t in o])
    same = lambda a, b: all(torch.equal(x, y) for x, y in zip(a, b))  # noqa: E731
    print(nm, "unp16 == unp20:", same(outs[0], outs[1]), " k16 == k20:", same(outs[2], outs[3]))
