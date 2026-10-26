import sys, torch; sys.path.insert(0, '.')
import bench, reference, submission
w, st = bench.state()
pass
for bs, sd in ((16, 7), (24, 0), (24, 7), (32, 1), (8, 10)):
    data = reference.generate_input(bs=bs, sharding="ep8", seed=7000 + sd)
    submission.TK2F = 0; ref = submission.kernel_fn(data, st).float(); torch.cuda.synchronize()
    submission.TK2F = 1
    res = []
    for k in range(6):
        out = submission.kernel_fn(data, st).float(); torch.cuda.synchronize()
        d = (out - ref).abs().amax(1)
        res.append(int((d > 1e-2).sum()))
    ok, msg = reference.check_implementation(data, out, w)
    print(bs, sd, "bad rows per run", res, msg)
