"""n61: vjx attention-only plan sweep on the layer's full scored inputs (n_splits / coop overrides vs the default plan),
CUDA-graph replay, cold L2 (bench.time_graph). python dev/n61/vjx_sweep.py <file> <filter> <seed> <rounds>"""
import sys, statistics, torch
sys.path.insert(0, ".")
import bench
sub = bench.load_sub(sys.argv[1]); flt = sys.argv[2]; sd = int(sys.argv[3]); rounds = int(sys.argv[4])
w, st = bench.weights_state(sub)
e0 = bench.empty_cost()
for load, c in bench.SCORED:
    name = f"{load} {c['layer']}"
    if flt not in name or c["layer"] != "full":
        continue
    data = bench.make_input(c, w, sd)
    cfg = data["config"]
    qkv = (data["attn_input"] @ st["dense"]["qkv_gate"].T).view(cfg["batch"], cfg["q_len"], -1)
    vd = {k: data[k] for k in ("cache_loc", "k_cache", "v_cache", "k_scale", "v_scale", "page_table", "seq_lens")}
    vd.update(qkv=qkv, config=cfg)
    B, L = cfg["batch"], cfg["q_len"]
    pairs = cfg["num_kv_heads"] // 2
    plan = sub._vjx__plan(B, pairs, L, 0, data["page_table"].shape[1], cfg["page_size"], qkv.device)
    n0 = plan["n_splits"]
    opts = [{}] + [dict(n_splits=n) for n in sorted({max(1, int(n0 * f)) for f in (0.5, 0.75, 1.25, 1.5, 2)})]
    opts += [dict(coop=1 - int(L == 8)), dict(n_splits=n0 * 2, coop=1 - int(L == 8))]
    ref = sub._vjx__launch(vd).float()
    t = {i: [] for i in range(len(opts))}
    for r in range(rounds):
        for i, o in enumerate(opts):
            f = lambda o=o: sub._vjx__launch(vd, **o)  # noqa: E731
            if r == 0:
                rel = ((f().float() - ref).norm() / ref.norm()).item()
                assert rel < 3e-2, (o, rel)
            t[i].append(bench.time_graph(f) - e0)
    base = statistics.median(t[0])
    print(f"{name} B{B} L{L} pairs{pairs} plan {plan}: default {base:.2f} us", flush=True)
    for i, o in enumerate(opts[1:], 1):
        m = statistics.median(t[i])
        print(f"   {o}: {m:.2f}  d {m - base:+.2f}", flush=True)
    del data, qkv, vd
