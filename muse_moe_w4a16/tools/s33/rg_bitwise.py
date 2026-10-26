# s33: RG (moe_routegather) vs RG 0 (moe_route + moe_gather): outputs must be bitwise equal on the layer routings
# and on skewed big-M routings (HOT_M on and off). Run from muse_moe_w4a16/: python tools/s33/rg_bitwise.py
import sys, torch
sys.path.insert(0, '.'); sys.path.insert(0, 'tools')
import layer_ab as L
sub, st = L.submission, L.st
bad = n = 0
def run(data):
    global bad, n
    outs = []
    for rg in (0, 1):
        sub.RG = rg; outs.append(sub.kernel_fn(data, st).clone())
    torch.cuda.synchronize(); n += 1
    if not torch.equal(outs[0], outs[1]): bad += 1; print("DIFF", data["config"]["bs"], (outs[0].float() - outs[1].float()).abs().max().item())
for hot in ({}, {256: 32}):
    sub.HOT_M = hot
    for (l, sd), loc in sorted(L.R.items()):
        if l.split()[1] in ("16", "32") or l.startswith("verify"):
            if loc.shape[0] > 32: run(L.mk(loc, sd))
    g = torch.Generator(device='cuda').manual_seed(5)
    for T in (33, 64, 128, 256):
        for k in range(6):  # skewed: a few hot experts
            p = torch.rand(256, device='cuda', generator=g) ** (8 if k % 2 else 2)
            ids = torch.stack([torch.multinomial(p, 8, generator=g) for _ in range(T)]).int()
            x = torch.randn(T, 4096, device='cuda', generator=g).bfloat16()
            tw = torch.rand(T, 8, device='cuda', generator=g); tw = tw / tw.sum(-1, keepdim=True)
            run({"hidden_states": x, "topk_ids": ids, "topk_weights": tw,
                 "config": {"bs": T, "d_hidden": 4096, "local_start": st["start"], "local_count": st["count"]}})
print(f"rg_bitwise: {bad} of {n} differ")
