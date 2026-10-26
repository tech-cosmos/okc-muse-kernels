import sys, torch
sys.path.insert(0, ".")
import bench
sub = bench.load_sub("submission.py")
w, s = bench.weights_state(sub)
st = s["moe"]
g = torch.Generator(device="cuda").manual_seed(7)
bad = tot = 0
for T in (1, 8, 16, 32):
    for r in range(150):
        x = (torch.randn(T, 4096, device="cuda", generator=g)).bfloat16()
        # skewed: some ids local [0,32), some off (-1)
        ids = torch.randint(0, 40 if r % 2 else 12, (T, 8), device="cuda", generator=g, dtype=torch.int32)
        ids = torch.where(ids >= 32, torch.full_like(ids, -1), ids)
        tw = torch.rand(T, 8, device="cuda", generator=g)
        d = {"hidden_states": x, "topk_ids": ids, "topk_weights": tw, "config": {"local_start": st["start"], "local_count": st["count"]}}
        sub._moe_TK2F = 1; a = sub._moe_kernel_fn(d, st)
        sub._moe_TK2F = 0; b = sub._moe_kernel_fn(d, st)
        torch.cuda.synchronize()
        tot += 1
        if not torch.equal(a, b):
            bad += 1; print("MISMATCH", T, r, (a.float()-b.float()).abs().max().item())
    print(T, "done", bad, tot, flush=True)
sub._moe_TK2F = 1
print("bad", bad, "of", tot)
