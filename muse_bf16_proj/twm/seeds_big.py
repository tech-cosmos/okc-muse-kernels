import sys; sys.path.insert(0,'.')
import torch, bench, submission, reference
w, st = bench.state()
nbad = 0
for bs, dist in [(128, "normal"), (256, "normal"), (128, "skewed"), (64, "normal"), (257, "normal")]:
    for sd in range(25):
        data = reference.generate_input(bs=bs, sharding="ep8", seed=9000 + sd, dist=dist)
        ids = data["topk_ids"]; c = data["config"]
        loc = ids[(ids >= c["local_start"]) & (ids < c["local_start"] + c["local_count"])]
        cnt = torch.bincount(loc.long() - c["local_start"], minlength=32)
        out = submission.kernel_fn(data, st); torch.cuda.synchronize()
        ok, msg = reference.check_implementation(data, out, w)
        if not ok: nbad += 1; print("FAIL", bs, dist, sd, int(cnt.max()), int((cnt>0).sum()), msg, flush=True)
print("bad", nbad)
