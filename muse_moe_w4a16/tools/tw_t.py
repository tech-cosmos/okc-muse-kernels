# usage: python /tmp/tw_t.py [cfgs...]  (TW_CFG values; 0 = old moe_big path): correctness + local timing bs128/256
import sys; sys.path.insert(0, '.')
import torch, bench, submission, reference
w, st = bench.state()
for cfg in [int(c) for c in sys.argv[1:]]:
    submission.TW_CFG = cfg
    rt = cfg // 100
    if cfg and f"w13b{rt}" not in st:
        st[f"w13b{rt}"], st[f"w13sb{rt}"] = submission._tb_pack(st["w13q"], st["w13s"], True, rt)
        st[f"w2b{rt}"], st[f"w2sb{rt}"] = submission._tb_pack(st["w2q"], st["w2s"], False, rt)
    bad = 0
    for c in [dict(bs=33, seed=1), dict(bs=257, seed=2), dict(bs=128, dist="skewed", seed=3), dict(bs=128, seed=4), dict(bs=256, seed=5), dict(bs=64, seed=6), dict(bs=256, seed=9)]:
        data = reference.generate_input(sharding="ep8", **c)
        out = submission.kernel_fn(data, st); torch.cuda.synchronize()
        ok, msg = reference.check_implementation(data, out, w)
        if not ok: bad += 1; print("FAIL", cfg, c, msg)
    t = [bench.run_shape(bs) for bs in (128, 256)]
    print(f"cfg {cfg}: bad {bad}  bs128 {t[0]:.1f}  bs256 {t[1]:.1f}", flush=True)
