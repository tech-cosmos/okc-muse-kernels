"""s64: csl NC 2 (B32 decode sliding, one wave) vs the current path (nds + _combine_hb). Accuracy vs the reference attention
(eager torch math in fp32 via nds) and graph-replay timing with a cold L2.  python tools/csl_nc2.py [submission.py] [reps]"""
import sys, fcntl, torch
sys.path.insert(0, ".")
import bench, reference

path = sys.argv[1] if len(sys.argv) > 1 else "submission.py"
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
sub = bench.load_sub(path)
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
sink = torch.zeros((), dtype=torch.int64, device="cuda")
w = reference.generate_weights(reference.WEIGHTS_SEED)
st = sub.prepare(w)
wq = st["dense"]["qkv_gate"]
for load in ("decode 32", "decode 16"):
    c = next(c for l, c in bench.SCORED if l == load and c["layer"] == "sliding")
    B = c["batch"]
    sub._SLIDING[0] = True
    worst = {}
    for seed in range(reps):
        d0 = bench.make_input(c, w, 9000 + seed)
        cfg = d0["config"]
        x = d0["attn_input"]
        kc = x.shape[1] // 2
        slabs = torch.bmm(x.view(B, 2, kc).permute(1, 0, 2), wq.view(wq.shape[0], 2, kc).permute(1, 2, 0), out_dtype=torch.float32)
        qbf = slabs.sum(0).to(torch.bfloat16).view(B, 1, -1)
        base = {k: d0[k] for k in ("positions", "cache_loc", "cache_seqlens", "cu_seqlens_q", "page_table", "seq_lens",
                                   "k_scale", "v_scale")}
        def run(q, nc, ss=0):
            kc_, vc_ = d0["k_cache"].clone(), d0["v_cache"].clone()
            o = sub.attn_kernel_fn({**base, "k_cache": kc_, "v_cache": vc_, "qkv": q, "scheduler_metadata": None,
                                    "config": cfg, "o_pf": (0, 0), "csl_pw": 0, "qkv_ss": ss, "qkv_slabs": slabs, "csl_nc": nc})
            return o.float(), kc_, vc_
        ref, rk, rv = run(qbf, 1)
        for name, q, nc, ss in (("nc2 bf16", qbf, 2, 0), ("nc2 f32", slabs.sum(0).view(B, 1, -1), 2, 0),
                                ("nc2 qss", slabs[0].view(B, 1, -1), 2, slabs[0].numel()), ("nc1 qss->sum", slabs[0].view(B, 1, -1), 1, slabs[0].numel())):
            o, k2, v2 = run(q, nc, ss)
            e = ((o - ref).abs().max() / ref.abs().max()).item()
            worst[name] = max(worst.get(name, 0), e)
            assert torch.equal(k2, rk) and torch.equal(v2, rv), name + ": cache write differs"
    print(load, "max rel err vs nc1 (bf16 q):", {k: f"{v:.2e}" for k, v in worst.items()})
    # timing: QSS slabs input (the layer's d32 form) through each path
    d0 = bench.make_input(c, w, 9100)
    for name, nc in (("nc1", 1), ("nc2", 2)):
        dd = {**{k: d0[k] for k in d0 if k != "config"}, "qkv": slabs[0].view(B, 1, -1), "scheduler_metadata": None,
              "config": d0["config"], "o_pf": (0, 0), "csl_pw": 0, "qkv_ss": slabs[0].numel(), "qkv_slabs": slabs, "csl_nc": nc}
        f = lambda: sub.attn_kernel_fn(dd)
        f(); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): f()
        ts = []
        with open("/tmp/okc-gpu.lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            for _ in range(60):
                torch.sum(flush.view(torch.int64), 0, out=sink)
                e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                e0.record(); g.replay(); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1) * 1e3)
            fcntl.flock(lk, fcntl.LOCK_UN)
        ts.sort()
        print(f"  {load} {name} (qss input): {ts[len(ts) // 2]:.2f} us")
