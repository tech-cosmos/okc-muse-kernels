"""n78 OVL (copy of n75 ab.py): ss3 = main v189 QLO; ovl = + vjx_ovl; ovlt = ovl + offset-move test hook.
n75 QLO: vjx q_len-8 (dyn pool, = main v188) vs + two-term E4M3 queries (vjx_qlo.py), standalone on the layer's
full verify inputs. Error vs the reference attention (BF16 queries, FP32 math), and the H100 graph-replay time.
python dev/n75/ab.py <filter> <seeds> <rounds> [variants=rs] [l2] (vjx_qlo.patch flat modes; l2: page table mod 8192 = KV L2-resident)"""
import sys, statistics, torch
sys.path.insert(0, ".")
sys.path.insert(0, "dev/n78")
import bench, vjx_src, vjx_dyn, vjx_qlo, vjx_ovl, reference as R
flt = sys.argv[1]; seeds = [int(x) for x in sys.argv[2].split(",")]; rounds = int(sys.argv[3])
base_src = vjx_src.CUDA_SRC
vjx_src.CUDA_SRC = vjx_dyn.patch(base_src, 100, 2)
mods = {}
for v in (sys.argv[4] if len(sys.argv) > 4 else "ss3,ovl").split(","):
    s = vjx_qlo.patch(vjx_dyn.patch(base_src, 100, 2), "ss3")
    if v.startswith("lnext"):
        s = vjx_ovl.patch_lnext(s, unr=1 if v == "lnext1" else 2, full=v.startswith("lnextf"), nowait=v.endswith("nw"), test=v.endswith("t") or v.endswith("t0"))
        if v.endswith("t0"):   # (never take the early path: the structure's own cost)
            s = s.replace("? mbar_test_full(&S.full[st_n], ph_n) : 0;", "? 0 : 0;")
    elif v == "lato":
        s = vjx_ovl.patch_lato(s)
    elif v.startswith("noend"):
        s = vjx_ovl.patch_noend(s, nofence=v == "noendf")
    elif v == "rot":
        s = vjx_ovl.patch_rot(s)
    elif v.startswith("ovl"):
        s = vjx_ovl.patch(s, test_hook=v == "ovlt", fast=v != "ovls")
    vjx_src.CUDA_SRC = s
    mods[v] = vjx_src._build(tag="n78" + v)
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
        x = data["attn_input"]
        qkv = (x @ st["dense"]["qkv_gate"].T).view(cfg["batch"], cfg["q_len"], -1)
        vd = {k: data[k] for k in ("cache_loc", "k_cache", "v_cache", "k_scale", "v_scale", "seq_lens", "page_table")}
        if len(sys.argv) > 5:
            pt = data["page_table"]
            data["page_table"] = torch.where(pt >= 0, pt % 8192, pt)
        vd.update(qkv=qkv, config=cfg, page_table=data["page_table"])
        fns = {k: (lambda m=m: vjx_src._launch(vd, _mod=m)) for k, m in mods.items()}
        ref = R._attention(data, x.float(), w)[0].float().view(cfg["batch"] * cfg["q_len"], -1)
        ref8 = R._attention(data, x.float(), w, torch.float8_e4m3fn)[0].float().view(ref.shape)
        rn = ref.norm()
        errs = {"ref_e4m3": ((ref8 - ref).norm() / rn).item()}
        mx = {"ref_e4m3": (ref8 - ref).abs().max().item()}
        for k, f in fns.items():
            o = f().float().view(ref.shape)
            errs[k] = ((o - ref).norm() / rn).item()
            mx[k] = (o - ref).abs().max().item()
        t = {k: [] for k in fns}
        for r in range(rounds):
            for k in (list(fns) if r % 2 == 0 else list(fns)[::-1]):
                t[k].append(bench.time_graph(fns[k]) - e0)
        print(f"{name} s{sd}: " + "  ".join(f"{k} {statistics.median(v):7.2f}" for k, v in t.items())
              + " | rel err vs ref (max abs): " + "  ".join(f"{k} {errs[k]:.2e} ({mx[k]:.1e})" for k in errs), flush=True)
        del data, qkv, fns
