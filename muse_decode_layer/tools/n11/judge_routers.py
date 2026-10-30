"""Judge-like routers: invert each scored case's speed_of_light_ns (judge json) for router_distinct (bytes-bound term,
hbm 4.8e12), then reference.weights_fields with those targets picks a router seed per run seed.
python tools/n11/judge_routers.py <judge json> <seeds comma>"""
import json, sys, torch
sys.path.insert(0, "."); import reference as R
d = json.load(open(sys.argv[1]))
hw = {"hbm_bytes_per_s": 4.8e12, "bf16_dense_flops_per_s": 989e12}
cases = []
for b in d["benchmarks"]:
    c = dict(b["case"]); kw = {k: c[k] for k in ("batch", "ctx", "layer", "q_len", "spread") if k in c}
    lo, hi = 0.0, 256.0
    for _ in range(60):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if R.speed_of_light_ns(**kw, hw=hw, router_distinct=m) < b["speed_of_light_ns"] else (lo, m)
    c["router_distinct"] = m
    c["seed"] = 801 + len(cases)
    cases.append(c)
    print(c["load"], c["layer"], "target distinct", round(m, 2))
W = R.generate_weights(R.WEIGHTS_SEED)
# weights_fields' search with router_distinct only (router_load isn't observable: the bound is bytes-bound)
loads = {}
for i, c in enumerate(cases):
    loads.setdefault(c["load"], []).append(i)
hs = []
with torch.no_grad():
    for c in cases:
        n = R.ROUTE_SAMPLES if c.get("batch", 1) * c.get("q_len", 1) <= R.SAMPLE_TOKENS else 1
        seeds = [c["seed"]] + [(c["seed"] + j * 0x2545F4914F6CDD1D) % (2**63 - 1) for j in range(1, n)]
        hs.append((torch.cat([R.ref_front(R.generate_input(**{**c, "seed": q, "local_start": 0}), W)["h"] for q in seeds]), n))
    for s in [int(x) for x in sys.argv[2].split(",")]:
        best = (9e9, 0, None)
        for i in range(1, 2001):
            rs = (s + i * 0x9E3779B97F4A7C15) % (2**63 - 1)
            router = R._router(torch.Generator(device="cuda").manual_seed(rs), "cuda", W["pre_ffn_norm"])
            pc = torch.stack([R._mean_busiest_load(h, router, n) for h, n in hs]).double().cpu()
            per = {l: (sum(c["weight"] * float(pc[j, 1]) for j, c in ((j, cases[j]) for j in ix)) / sum(cases[j]["weight"] for j in ix),
                       sum(cases[j]["weight"] * cases[j]["router_distinct"] for j in ix) / sum(cases[j]["weight"] for j in ix)) for l, ix in loads.items()}
            err = max(abs(a / b - 1) for a, b in per.values())
            if err < best[0]:
                best = (err, rs, {l: round(a, 2) for l, (a, b) in per.items()},
                        {l: [round(float(pc[j, 0]), 1) for j in ix] for l, ix in loads.items()})
            if err <= R.LOAD_TOL:
                break
        print("run seed", s, "router", best[1], "err", round(best[0], 4), "distinct", best[2], "routes", best[3], flush=True)
