# n4s1: NORM_EARLY 1 (moe_norm waits per token on twm/moe_hot gemm2 flags) vs 0 (griddepcontrol.wait): outputs must be
# bitwise equal. Layer routings + skewed big-M routings, HOT_M off/on, K-split tail off/on; each graph replayed REPS
# times back to back (race stress). Run from muse_moe_w4a16/: python tools/n4s1/ne_bitwise.py [reps]
import sys, torch
sys.path.insert(0, '.'); sys.path.insert(0, 'tools')
import layer_ab as L
sub, st = L.submission, L.st
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
bad = n = 0
def run(data):
    global bad, n
    sub.NORM_EARLY = 0; ref = sub.kernel_fn(data, st).clone()
    sub.NORM_EARLY = 1
    sub.kernel_fn(data, st); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): o = sub.kernel_fn(data, st)
    for r in range(REPS):
        g.replay()
        if r % 5 == 4 or r == REPS - 1:
            torch.cuda.synchronize(); n += 1
            if not torch.equal(o, ref): bad += 1; print("DIFF", data["config"]["bs"], r, (o.float() - ref.float()).abs().max().item(), flush=True)
    del g
for hot, split in (({}, {}), ({256: 32}, {}), ({}, {128: 132, 256: 66}), ({256: 32}, {256: 66})):
    sub.HOT_M = hot; sub.TWF_SPLIT_M = split
    for (l, sd), loc in sorted(L.R.items()):
        if loc.shape[0] > 32: run(L.mk(loc, sd))
    g = torch.Generator(device='cuda').manual_seed(5)
    for T in (33, 64, 128, 256, 257):
        for k in range(6):  # skewed: a few hot experts
            p = torch.rand(256, device='cuda', generator=g) ** (8 if k % 2 else 2)
            ids = torch.stack([torch.multinomial(p, 8, generator=g) for _ in range(T)]).int()
            x = torch.randn(T, 4096, device='cuda', generator=g).bfloat16()
            tw = torch.rand(T, 8, device='cuda', generator=g); tw = tw / tw.sum(-1, keepdim=True)
            run({"hidden_states": x, "topk_ids": ids, "topk_weights": tw,
                 "config": {"bs": T, "d_hidden": 4096, "local_start": st["start"], "local_count": st["count"]}})
    print("done", hot, split, bad, n, flush=True)
print(f"ne_bitwise: {bad} of {n} differ")
