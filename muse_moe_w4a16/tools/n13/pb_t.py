# n13 TWM_PB_M {T: PB} (default 0 = mixed N48/N64 passes) vs 64 (copy of n12 pb_t.py): reference check + bitwise, on the layer's v16/v32
# routings and N skewed synthetic T=128/256 routings (experts of 33..64 and > 64 tokens). Run from muse_moe_w4a16/.
import sys, os, torch
sys.path.insert(0, '/home/daytona/r-moe/muse_moe_w4a16/tools'); sys.path.insert(0, '.')
import layer_ab as L, reference
sub = L.submission; N = int(os.environ.get("N", "40")); bad = nd = 0
cases = [(k, loc) for k, loc in sorted(L.R.items()) if k[0] in ("verify 16", "verify 32")]
for i in range(N):
    g = torch.Generator().manual_seed(700 + i)
    T = 256 if i % 2 else 128
    p = torch.rand(32, generator=g) ** 4 + 0.03
    loc = torch.full((T, 8), -1, dtype=torch.int32)
    for t in range(T):
        k = int(torch.randint(0, 9, (1,), generator=g))
        if k: loc[t, :k] = torch.multinomial(p, k, replacement=False, generator=g).int()
    cases.append((("syn", i), loc))
for key, loc in cases:
    T = loc.shape[0]
    data = L.mk(loc, 900 + hash(key) % 1000)
    outs = []
    for hm in ({}, {256: 32}):
        sub.HOT_M = hm
        sub.TWM_PB_M = {}; o0 = sub.kernel_fn(data, L.st).clone()
        sub.TWM_PB_M = {T: int(os.environ.get("PB", "0"))}; o1 = sub.kernel_fn(data, L.st).clone(); torch.cuda.synchronize()
        ok = reference.check_implementation(data, o1, L.w)[0]; d = (o1.float() - o0.float()).abs().max().item()
        bad += not ok; nd += d > 0
        outs.append((ok, d))
    nt = torch.bincount(loc[loc >= 0].long(), minlength=32)
    print(key, T, "33-64", [int(x) for x in nt if 32 < x <= 64], outs, flush=True)
sub.HOT_M = {}; sub.TWM_PB_M = {}
print("bad", bad, "nonbitwise", nd)
