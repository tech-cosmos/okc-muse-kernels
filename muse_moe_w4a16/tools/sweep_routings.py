# many-routing correctness sweep of kernel_fn (decode shapes, normal + skewed). S0=<seed base> NS=<seeds> python tools/sweep_routings.py
import sys, torch; sys.path.insert(0, '.')
import bench, reference, submission
w, st = bench.state()
gsv = int(sys.argv[1]) if len(sys.argv) > 1 else 16
pass
import os; submission.TK2F = int(os.environ.get("TK2F", "1"))
bad = 0; worst = 0
for bs in (1, 4, 8, 16, 24, 32):
    for sd in range(int(__import__("os").environ.get("NS", "12"))):
        for dist in ("normal", "skewed"):
            data = reference.generate_input(bs=bs, sharding="ep8", seed=int(__import__("os").environ.get("S0", "7000")) + sd, dist=dist)
            out = submission.kernel_fn(data, st); torch.cuda.synchronize()
            ok, msg = reference.check_implementation(data, out, w)
            ids = data["topk_ids"]; c = torch.bincount(ids[(ids >= 0) & (ids < 32)].long(), minlength=32)
            worst = max(worst, int(c.max()))
            if not ok: bad += 1; print("FAIL", bs, sd, dist, int(c.max()), msg)
print("gs", gsv, "fails", bad, "max tokens/expert", worst)
