"""n57: attention-only A/B on the layer's own full-shape inputs: ours (attn_kernel_fn, BF16 qkv) vs vjx (_vjx__launch),
CUDA-graph replay, cold L2 (bench.time_graph), per seed. The KV cache gets the same new rows from both (same qkv).
python dev/n57/attn_ab.py <file> <filter> <seeds> <rounds>"""
import sys, statistics, torch
sys.path.insert(0, ".")
import bench
sub = bench.load_sub(sys.argv[1]); flt = sys.argv[2]; seeds = [int(x) for x in sys.argv[3].split(",")]; rounds = int(sys.argv[4])
w, st = bench.weights_state(sub)
e0 = bench.empty_cost()
for load, c in bench.SCORED:
    name = f"{load} {c['layer']}"
    if flt not in name:
        continue
    for sd in seeds:
        data = bench.make_input(c, w, sd)
        cfg = data["config"]
        qkv = (data["attn_input"] @ st["dense"]["qkv_gate"].T).view(cfg["batch"], cfg["q_len"], -1)
        ours = lambda: sub.attn_kernel_fn({  # noqa: E731
            "o_pf": (0, 0), "csl_pw": False, "nvs_cl": sub.NVS_CL, "qkv": qkv, "positions": data["positions"],
            "cache_loc": data["cache_loc"], "cache_seqlens": data["cache_seqlens"], "cu_seqlens_q": data["cu_seqlens_q"],
            "scheduler_metadata": data["scheduler_metadata"], "k_cache": data["k_cache"], "v_cache": data["v_cache"],
            "k_scale": data["k_scale"], "v_scale": data["v_scale"], "page_table": data["page_table"],
            "seq_lens": data["seq_lens"], "config": cfg, "qkv_ss": 0, "qkv_slabs": None})
        vd = {k: data[k] for k in ("cache_loc", "k_cache", "v_cache", "k_scale", "v_scale", "page_table", "seq_lens")}
        vd.update(qkv=qkv, config=cfg)
        vj = lambda: sub._vjx__launch(vd)  # noqa: E731
        a, b = ours().float(), vj().float()
        rel = ((a - b).norm() / a.norm()).item()
        t = [[], []]
        for r in range(rounds):
            for i, f in ((0, ours), (1, vj)) if r % 2 == 0 else ((1, vj), (0, ours)):
                t[i].append(bench.time_graph(f) - e0)
        print(f"{name} s{sd}: ours {statistics.median(t[0]):7.2f}  vjx {statistics.median(t[1]):7.2f}  "
              f"d {statistics.median(t[1]) - statistics.median(t[0]):+6.2f} us  (out rel diff {rel:.2e})", flush=True)
        del data, qkv
