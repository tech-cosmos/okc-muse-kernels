"""n66: vjx flat q_len-8 base vs dynamic-pool builds (dev/n66/dyn.py) on the layer's full-shape inputs.
python dev/n66/ab.py <filter> <seeds> <rounds> <spec,...> [l2]; spec = dynf:dynch[:mode:pf] (v1 dyn.py) or 0:dynf:dynch (v2 donor.py)"""
import sys, statistics, torch
sys.path.insert(0, "."); sys.path.insert(0, "dev/n66")
import bench, vjx_src, dyn, donor
flt = sys.argv[1]; seeds = [int(x) for x in sys.argv[2].split(",")]; rounds = int(sys.argv[3])
dyns = [tuple(int(y) for y in x.split(":")) for x in sys.argv[4].split(",")]; l2 = len(sys.argv) > 5
base_src = vjx_src.CUDA_SRC
mods = {"base": vjx_src._build(tag="n66b")}
for f in dyns:
    vjx_src.CUDA_SRC = donor.patch(base_src, *f[1:]) if f[0] == 0 else dyn.patch(base_src, *f)
    mods["d" + "_".join(map(str, f))] = vjx_src._build(tag="n66d" + "_".join(map(str, f)))
vjx_src.CUDA_SRC = base_src
sub = bench.load_sub("submission.py")
w, st = bench.weights_state(sub)
e0 = bench.empty_cost()
for load, c in bench.SCORED:
    name = f"{load} {c['layer']}"
    if flt not in name or c["layer"] != "full" or c["q_len"] != 8:
        continue
    for sd in seeds:
        data = bench.make_input(c, w, sd)
        cfg = data["config"]
        qkv = (data["attn_input"] @ st["dense"]["qkv_gate"].T).view(cfg["batch"], cfg["q_len"], -1)
        pt = data["page_table"]
        if l2:
            pt = torch.where(pt >= 0, pt % 8192, pt)
        vd = {k: data[k] for k in ("cache_loc", "k_cache", "v_cache", "k_scale", "v_scale", "seq_lens")}
        vd.update(qkv=qkv, config=cfg, page_table=pt)
        fns = {k: (lambda m=m: vjx_src._launch(vd, _mod=m)) for k, m in mods.items()}
        ref = fns["base"]().float()
        errs = {}
        for k, f in fns.items():
            outs = [f().float() for _ in range(3)]
            errs[k] = max(((o - ref).norm() / ref.norm()).item() for o in outs)
        t = {k: [] for k in fns}
        for r in range(rounds):
            for k in (list(fns) if r % 2 == 0 else list(fns)[::-1]):
                t[k].append(bench.time_graph(fns[k]) - e0)
        print(f"{name} s{sd}{' L2' if l2 else ''}: " + "  ".join(
            f"{k} {statistics.median(v):7.2f} (err {errs[k]:.1e})" for k, v in t.items()), flush=True)
        del data, qkv, fns
