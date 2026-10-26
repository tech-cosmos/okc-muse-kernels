"""proj s22 QSS: attention on qkv as 2 FP32 slabs (qkv_ss) must be bit-identical to attention on the FP32 sum of the
slabs (QF32: rounded at each load): outputs and the K/V cache writes. python tools/qss_eq.py [file]"""
import sys, torch
sys.path.insert(0, ".")
import bench, reference

sub = bench.load_sub(sys.argv[1] if len(sys.argv) > 1 else "submission.py")
w = reference.generate_weights(reference.WEIGHTS_SEED)
cases = [dict(batch=b, ctx=80000, layer=l, page=1, q_len=1, dist="soft", **({"spread": "agentic"} if b > 1 else {}))
         for b in (32, 16, 8, 2, 1) for l in ("sliding", "full")]
cases += [dict(batch=32, ctx=3000, layer="sliding", spread="mixed", dist="soft", page=1), dict(batch=48, ctx=9000, layer="sliding", dist="soft", page=1), dict(batch=16, ctx=20000, layer="full", spread="mixed", dist="soft", page=1),
          dict(batch=8, ctx=2047, layer="sliding"), dict(batch=9, ctx=5000, layer="sliding", spread="mixed", page=1)]
bad = 0
for c in cases:
    for seed in (1, 2, 3):
        d = bench.make_input(c, w, seed)
        cfg = d["config"]
        n = (2 * cfg["num_q_heads"] + 2 * cfg["num_kv_heads"]) * cfg["head_dim"]
        g = torch.Generator(device="cuda").manual_seed(seed)
        sl = torch.randn((2, cfg["batch"] * cfg["q_len"], n), device="cuda", generator=g) * (1 + 3 * seed)
        q32 = (sl[0] + sl[1]).view(cfg["batch"], cfg["q_len"], n)
        res = []
        for q, ss in ((q32, 0), (q32, 0), (sl[0].view(cfg["batch"], cfg["q_len"], n), sl[0].numel())):
            kc, vc = d["k_cache"].clone(), d["v_cache"].clone()
            o = sub.attn_kernel_fn({"qkv": q, "positions": d["positions"], "cache_loc": d["cache_loc"],
                                    "cache_seqlens": d["cache_seqlens"], "cu_seqlens_q": d["cu_seqlens_q"],
                                    "scheduler_metadata": d["scheduler_metadata"], "k_cache": kc, "v_cache": vc,
                                    "k_scale": d["k_scale"], "v_scale": d["v_scale"], "page_table": d["page_table"],
                                    "seq_lens": d["seq_lens"], "config": cfg, "qkv_ss": ss, "qkv_slabs": sl})
            torch.cuda.synchronize()
            res.append((o.clone(), kc, vc))
        eq = lambda a, b: torch.equal(a.view(torch.int16) if a.dtype == torch.bfloat16 else a.view(torch.uint8),
                                      b.view(torch.int16) if b.dtype == torch.bfloat16 else b.view(torch.uint8))
        md = lambda a, b: (a.float() - b.float()).abs().max().item()
        cache_ok = all(eq(res[0][i], res[2][i]) for i in (1, 2))
        o_ok = eq(res[0][0], res[2][0])
        ctrl = eq(res[0][0], res[1][0])
        ok = cache_ok and (o_ok or not ctrl)
        bad += not ok
        print(c["batch"], c["layer"], c["ctx"], seed, "cache", cache_ok, "out", o_ok, "ctrl-det", ctrl,
              "maxdiff f32 %.3g ctrl %.3g" % (md(res[0][0], res[2][0]), md(res[0][0], res[1][0])))
        del d, res
print("bad", bad)
