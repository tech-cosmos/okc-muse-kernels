"""n67: SEGK (per-boundary virtual keys in vjx's static partition) sweep, with/without n66's dynamic pool (dyn.py v1).
python dev/n67/ab.py <filter> <seeds> <rounds> <spec,...>; spec = segk:dynf:dynch[:kind] (dynf 0 = no pool; kind 1 = dyn1 q1+q8, 2 = dyn1 q1 only). Base = v187 vjx."""
import sys, statistics, torch
sys.path.insert(0, "."); sys.path.insert(0, "dev/n66")
sys.path.insert(0, "dev/n67")
import bench, vjx_src, dyn, dyn1
flt = sys.argv[1]; seeds = [int(x) for x in sys.argv[2].split(",")]; rounds = int(sys.argv[3])
specs = [tuple(int(y) for y in (x + ":0").split(":")[:4]) for x in sys.argv[4].split(",")]
base_src = vjx_src.CUDA_SRC
mods = {"base": vjx_src._build(tag="n66b")}
for sk, f, ch, kd in specs:
    s = base_src.replace("(L == 8) ? 600 : 400", f"(L == 8) ? {sk} : 400")
    if f:
        s = (dyn1.patch(s, f, ch, 1 if kd == 1 else 0) if kd else dyn.patch(s, f, ch)).replace("(p.B - 1) * 600", f"(p.B - 1) * {sk}")
    vjx_src.CUDA_SRC = s
    k = f"s{sk}_{f}_{ch}" + (f"_k{kd}" if kd else "")
    mods[k] = vjx_src._build(tag="n67" + k)
vjx_src.CUDA_SRC = base_src
sub = bench.load_sub("submission.py")
w, st = bench.weights_state(sub)
e0 = bench.empty_cost()
for load, c in bench.SCORED:
    name = f"{load} {c['layer']}"
    if flt not in name or c["layer"] != "full":
        continue
    for sd in seeds:
        data = bench.make_input(c, w, sd)
        cfg = data["config"]
        qkv = (data["attn_input"] @ st["dense"]["qkv_gate"].T).view(cfg["batch"], cfg["q_len"], -1)
        vd = {k: data[k] for k in ("cache_loc", "k_cache", "v_cache", "k_scale", "v_scale", "seq_lens", "page_table")}
        vd.update(qkv=qkv, config=cfg)
        fns = {k: (lambda m=m: vjx_src._launch(vd, _mod=m)) for k, m in mods.items()}
        ref = fns["base"]().float()
        errs = {k: max(((f().float() - ref).norm() / ref.norm()).item() for _ in range(3)) for k, f in fns.items()}
        t = {k: [] for k in fns}
        for r in range(rounds):
            for k in (list(fns) if r % 2 == 0 else list(fns)[::-1]):
                t[k].append(bench.time_graph(fns[k]) - e0)
        print(f"{name} s{sd}: " + "  ".join(f"{k} {statistics.median(v):7.2f} ({errs[k]:.0e})" for k, v in t.items()), flush=True)
        del data, qkv, fns
