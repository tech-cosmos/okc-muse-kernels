"""s52 RG: moe_routegathern (route + gathern in one launch) must give bit-identical layer outputs to moe_route + moe_gathern
on T > 32 shapes (scored v16/v32 + odd T), with and without the fused pooler+router (fr). python tools/rg_eq.py [file]"""
import sys, torch
sys.path.insert(0, ".")
import bench, reference

sub = bench.load_sub(sys.argv[1] if len(sys.argv) > 1 else "submission.py")
sub.TUNE = {}
w, st = bench.weights_state(sub)
cases = [c for l, c in bench.SCORED if l.startswith("verify 16") or l.startswith("verify 32")]
cases += [dict(batch=33, ctx=20000, layer="full", spread="mixed", page=1), dict(batch=5, ctx=5000, layer="sliding", q_len=8, spread="mixed", page=1),
          dict(batch=48, ctx=9000, layer="sliding", dist="soft", page=1), dict(batch=40, ctx=3000, layer="sliding", dist="outliers", page=1)]
bad = n = 0
for c in cases:
    for seed in (1, 2, 3):
        outs = []
        for fr in (False, True):
            for rg in (False, False, True):
                d = bench.make_input(c, w, seed)
                T = d["config"]["tokens"]
                for sl in (True, False):
                    st["tune"][(T, sl)] = {"rg": rg, "fr": fr}
                o = sub.kernel_fn(d, st)
                torch.cuda.synchronize()
                outs.append((fr, rg, [t.clone() for t in o if torch.is_tensor(t)] + [d["k_cache"].clone(), d["v_cache"].clone()]))
        for i in (0, 3):
            ref, ctl, new = outs[i][2], outs[i + 1][2], outs[i + 2][2]
            same = all(torch.equal(a, b) for a, b in zip(ref, new))
            sctl = all(torch.equal(a, b) for a, b in zip(ref, ctl))
            n += 1
            bad += not same
            print(c, seed, "T", T, "fr", outs[i][0], "OK" if same else "DIFF", "control", "OK" if sctl else "DIFF",
                  "maxdiff", max((a.float() - b.float()).abs().max().item() for a, b in zip(ref, new)), flush=True)
print(f"rg_eq: {bad} of {n} differ")
