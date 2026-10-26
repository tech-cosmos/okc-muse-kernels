import sys, os; sys.path.insert(0, os.path.expanduser('~/r-moe/muse_moe_w4a16'))
import torch, bench, submission as S, reference, importlib
w, st = bench.state()
V = {"main": {}, "a22": {"TWF_SPLIT_M": {128: 1022, 256: 1022, 64: 1022, 257: 1022}}, "a23": {"TWF_SPLIT_M": {128: 1023, 256: 1023, 64: 1023, 257: 1023}},
     "o76a23": {"TWM_ORDER_M": {128: 76, 256: 76, 64: 76, 257: 76}, "TWF_SPLIT_M": {128: 1023, 256: 1023, 64: 1023, 257: 1023}},
     "hota22": {"TWM_HOT_M": {128: 1, 256: 1, 64: 1, 257: 1}, "TWF_SPLIT_M": {128: 1022, 256: 1022, 64: 1022, 257: 1022}}}
old = {k: getattr(S, k) for d in V.values() for k in d}
nbad = 0; ref_main = {}
for name, d in V.items():
    for k, v in old.items(): setattr(S, k, v)
    for k, v in d.items(): setattr(S, k, v)
    bad = 0; md = 0.0
    for bs, dist in [(128, "normal"), (256, "normal"), (128, "skewed"), (64, "normal"), (257, "normal")]:
        for sd in range(int(sys.argv[1]) if len(sys.argv) > 1 else 10):
            data = reference.generate_input(bs=bs, sharding="ep8", seed=9000 + sd, dist=dist)
            out = S.kernel_fn(data, st); torch.cuda.synchronize()
            ok, msg = reference.check_implementation(data, out, w)
            if not ok: bad += 1; print("FAIL", name, bs, dist, sd, msg, flush=True)
            key = (bs, dist, sd)
            if name == "main": ref_main[key] = out.clone()
            else: md = max(md, (out.float() - ref_main[key].float()).abs().max().item())
    print(name, "bad", bad, "maxdiff vs main", md, flush=True); nbad += bad
print("TOTAL bad", nbad)
