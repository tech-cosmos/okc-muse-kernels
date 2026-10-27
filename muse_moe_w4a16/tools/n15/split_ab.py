# n15: moe_hot + reduced twm grid (HOT_TWG) on the layer's verify 32 routings, H100, ABAB rounds
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import layer_ab as L
S = L.submission
V = [("h32", {256: 32}, 0)] + [(f"s{k}", {256: 32}, k) for k in (33, 66, 100, 132, 200)]
S.TWM_ORDER_M = {}
loads = sys.argv[1:] or ["verify 32"]
for rnd in range(int(os.environ.get("RND", "2"))):
    for load in loads:
        r = {n: [] for n, _, _ in V}
        for (l, sd), loc in sorted(L.R.items()):
            if l != load: continue
            data = L.mk(loc, sd)
            ref = None
            for n, hm, tg in V:
                S.HOT_M, S.TWF_SPLIT_M = hm, ({128: tg, 256: tg} if tg else {})
                o = S.kernel_fn(data, L.st).float()
                if ref is None: ref = o
                assert (o - ref).abs().max().item() < 0.02, n
                r[n].append(L.tg(lambda: S.kernel_fn(data, L.st)))
        print(f"r{rnd} {load}: " + "  ".join(f"{n} {sum(v)/len(v):.1f}" for n, v in r.items()), flush=True)
