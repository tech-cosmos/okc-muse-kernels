"""n65: compute-only oracle for full attention: same kernels on the layer's inputs, but every page id remapped into a
small L2-resident pool (pt % POOL), so the KV gather costs ~nothing and the time is the kernel's math/issue floor.
If v16/v32 L2-time is close to the H200's HBM time, the q_len-8 kernels are compute-bound on the judge.
python dev/n65/l2_oracle.py <file> <filter> <seed> <rounds> [pool]"""
import sys, statistics, torch
sys.path.insert(0, ".")
import bench
sub = bench.load_sub(sys.argv[1]); flt = sys.argv[2]; sd = int(sys.argv[3]); rounds = int(sys.argv[4])
pool = int(sys.argv[5]) if len(sys.argv) > 5 else 8192
w, st = bench.weights_state(sub)
e0 = bench.empty_cost()
for load, c in bench.SCORED:
    name = f"{load} {c['layer']}"
    if flt not in name or c["layer"] != "full":
        continue
    data = bench.make_input(c, w, sd)
    cfg = data["config"]
    qkv = (data["attn_input"] @ st["dense"]["qkv_gate"].T).view(cfg["batch"], cfg["q_len"], -1)
    pt_hbm = data["page_table"]
    pt_l2 = torch.where(pt_hbm >= 0, pt_hbm % pool, pt_hbm)
    def mk(pt):
        ours = lambda: sub.attn_kernel_fn({  # noqa: E731
            "o_pf": (0, 0), "csl_pw": False, "nvs_cl": sub.NVS_CL, "qkv": qkv, "positions": data["positions"],
            "cache_loc": data["cache_loc"], "cache_seqlens": data["cache_seqlens"], "cu_seqlens_q": data["cu_seqlens_q"],
            "scheduler_metadata": data["scheduler_metadata"], "k_cache": data["k_cache"], "v_cache": data["v_cache"],
            "k_scale": data["k_scale"], "v_scale": data["v_scale"], "page_table": pt,
            "seq_lens": data["seq_lens"], "config": cfg, "qkv_ss": 0, "qkv_slabs": None})
        vd = {k: data[k] for k in ("cache_loc", "k_cache", "v_cache", "k_scale", "v_scale", "seq_lens")}
        vd.update(qkv=qkv, config=cfg, page_table=pt)
        return ours, (lambda: sub._vjx__launch(vd))
    fns = {}
    for tag, pt in (("hbm", pt_hbm), ("l2", pt_l2)):
        o, v = mk(pt)
        fns["ours_" + tag], fns["vjx_" + tag] = o, v
    t = {k: [] for k in fns}
    for r in range(rounds):
        for k in (list(fns) if r % 2 == 0 else list(fns)[::-1]):
            t[k].append(bench.time_graph(fns[k]) - e0)
    keys = int(data["seq_lens"].sum())
    print(f"{name} s{sd} keys {keys}: " + "  ".join(f"{k} {statistics.median(v):7.2f}" for k, v in t.items()), flush=True)
    del data, qkv, fns
