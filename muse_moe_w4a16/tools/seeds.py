import sys; sys.path.insert(0,'.')
import torch, bench, submission, reference
w, st = bench.state()
import os; st["lib"].moe_set_blk(int(os.environ.get("BLK", "0")))
worst = {}
for bs in [16, 32, 8, 2, 24]:
    for sd in range(40):
        data = reference.generate_input(bs=bs, sharding="ep8", seed=5000 + sd)
        ids = data["topk_ids"]; cnt = torch.bincount(ids[(ids >= 0) & (ids < 32)].long(), minlength=32)
        out = submission.kernel_fn(data, st); torch.cuda.synchronize()
        ok, msg = reference.check_implementation(data, out, w)
        if not ok or sd < 1: print(bs, sd, int(cnt.max()), ok, msg, flush=True)
        worst[bs] = max(worst.get(bs, 0), int(cnt.max()))
print("max tokens/expert seen", worst)
