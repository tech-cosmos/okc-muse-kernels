"""ROUTING-SLACK SIZE ORACLE (lead 2026-10-31 09:55Z). Measurement only; NEVER in submission.py.
For reference routings on the scored shapes under judge-like routers (tools/n11), the minimum number of distinct
local experts a routing could touch while still passing reference.check_routing (exact search over local-expert sets).
python tools/fresh/slack/slack.py [inputs_small] [inputs_big]   (run in muse_decode_layer/)"""
import sys, itertools, torch
sys.path.insert(0, "."); import reference as R

ROUTERS = [None, 6023193148244720467, 7745419892846091016, 5745108551026741284, 4125531254498798136]
LOADS = [("d1", 1, 1), ("d16", 16, 1), ("d32", 32, 1), ("v1", 1, 8), ("v16", 16, 8), ("v32", 32, 8)]
NS, NB = (int(sys.argv[1]) if len(sys.argv) > 1 else 8), (int(sys.argv[2]) if len(sys.argv) > 2 else 3)


def min_distinct(biased, local_start):
    """(ref routes, ref distinct, min distinct, max local routes at that minimum, floor_d, floor_r)."""
    T = biased.shape[0]
    top = biased.topk(R.TOP_K + 1, dim=-1)
    v8, v9 = top.values[:, 7:8], top.values[:, 8:9]
    ref_ids = top.indices[:, :8]
    keep_tol = R.KEEP_TOL[0] if T <= R.KEEP_TOKENS else R.KEEP_TOL[1]
    loc = lambda e: (e - local_start) % R.N_EXPERTS < R.LOCAL_EXPERTS
    cand = biased >= v8 - R.ROUTE_TOL                      # any expert allowed for this token
    ar = torch.arange(R.N_EXPERTS, device=biased.device)
    isloc = loc(ar)
    nonloc_n = (cand & ~isloc).sum(1).tolist()             # non-local candidates per token
    L, M = [], 0
    for t in range(T):
        lt = 0
        for e in ar[cand[t] & isloc].tolist():
            lt |= 1 << ((e - local_start) % R.N_EXPERTS)
        L.append(lt)
        for s in range(8):
            e = int(ref_ids[t, s])
            if loc(e) and float(top.values[t, s] - v9[t, 0]) > keep_tol:
                M |= 1 << ((e - local_start) % R.N_EXPERTS)
    ref_r, ref_d = R._local_load(ref_ids, local_start)
    floor_r = ref_r * (1 - R.LOCAL_SLACK[0]) - R.LOCAL_SLACK[1]
    floor_d = ref_d * (1 - R.DISTINCT_SLACK[0]) - R.DISTINCT_SLACK[1]
    need = [max(0, 8 - n) for n in nonloc_n]
    pc = lambda x: bin(x).count("1")
    U = 0
    for lt in L:
        U |= lt
    free = [i for i in range(32) if (U >> i) & 1 and not (M >> i) & 1]

    def ok(S):
        if any(pc(L[t] & S) < need[t] for t in range(T)):
            return None
        r = sum(min(8, pc(L[t] & S)) for t in range(T))
        return r if r >= floor_r else None

    lb = max(pc(M), 0)
    while lb < floor_d:
        lb += 1
    for k in range(lb, pc(M) + len(free) + 1):
        extra = k - pc(M)
        if extra < 0:
            continue
        # order free experts by how many tokens can use them (helps find feasible sets early)
        order = sorted(free, key=lambda i: -sum((lt >> i) & 1 for lt in L))
        best = None
        n_try = 0
        for comb in itertools.combinations(order, extra):
            S = M
            for i in comb:
                S |= 1 << i
            r = ok(S)
            n_try += 1
            if r is not None and (best is None or r > best):
                best = r
                break
            if n_try > 2_000_000:
                break
        if best is not None:
            return ref_r, ref_d, k, best, floor_d, floor_r, pc(M)
    return ref_r, ref_d, ref_d, ref_r, floor_d, floor_r, pc(M)


W0 = R.generate_weights(R.WEIGHTS_SEED)
print("router load layer | ref_routes ref_distinct -> min_distinct (saved) | must-keep  floor_d  routes@min floor_r")
tot = {}
with torch.no_grad():
    for rs in ROUTERS:
        W = W0 if rs is None else {**W0, **R._router(torch.Generator(device="cuda").manual_seed(rs), "cuda", W0["pre_ffn_norm"])}
        for name, b, ql in LOADS:
            for layer in ("sliding", "full"):
                case = dict(batch=b, ctx=80000, layer=layer, page=1, q_len=ql, dist="soft",
                            spread="agentic" if b > 1 else "fixed")
                n = NS if b * ql <= R.SAMPLE_TOKENS else NB
                rows = []
                for s in range(n):
                    data = R.generate_input(**case, seed=9001 + 7919 * s, local_start=0)
                    fr = R.ref_front(data, W)
                    ls = R.case_fields(case | {"seed": 9001 + 7919 * s}, W)["local_start"]
                    rows.append(min_distinct(fr["biased"], ls))
                    del data, fr
                m = [sum(r[i] for r in rows) / n for i in range(7)]
                tot.setdefault((name, layer), []).append(m)
                print(f"{str(rs)[:6]:>6} {name:>3} {layer[:2]} | {m[0]:6.1f} {m[1]:5.2f} -> {m[2]:5.2f} ({m[1]-m[2]:4.2f})"
                      f" | {m[6]:5.2f} {m[4]:6.2f} {m[3]:7.1f} {m[5]:7.1f}", flush=True)
print("\nmean over routers: load layer ref_distinct min_distinct saved  us@6.5")
for (name, layer), ms in tot.items():
    rd = sum(m[1] for m in ms) / len(ms); md = sum(m[2] for m in ms) / len(ms)
    print(f"{name:>3} {layer[:2]} {rd:6.2f} {md:6.2f} {rd-md:5.2f} {6.5*(rd-md):6.1f}")
