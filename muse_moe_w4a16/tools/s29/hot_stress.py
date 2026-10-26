# skewed big-M routings with hot (65..128) and >128-token experts: HOT on vs off bitwise + reference
import sys, os, torch
sys.path.insert(0, '/home/daytona/r-moe/muse_moe_w4a16/tools'); sys.path.insert(0, '.')
import layer_ab as L, reference
sub = L.submission; N = int(os.environ.get("N", "40")); bad = nd = 0
for i in range(N):
    g = torch.Generator().manual_seed(500 + i)
    T = 256 if i % 4 else 128
    # local expert probs: a few heavy ones
    p = torch.rand(32, generator=g) ** 6 + 0.02
    loc = torch.full((T, 8), -1, dtype=torch.int32)
    for t in range(T):
        k = int(torch.randint(0, 9, (1,), generator=g))
        loc[t, :k] = torch.multinomial(p, k, replacement=False, generator=g).int() if k else loc[t, :0]
    nt = torch.bincount(loc[loc >= 0].long(), minlength=32)
    data = L.mk(loc, 900 + i)
    sub.HOT_M = {}; o0 = sub.kernel_fn(data, L.st).clone()
    sub.HOT_M = {T: 32}; o1 = sub.kernel_fn(data, L.st).clone(); torch.cuda.synchronize()
    ok = reference.check_implementation(data, o1, L.w)[0]; d = (o1.float() - o0.float()).abs().max().item()
    bad += not ok; nd += d > 0
    print(T, "hot", [int(x) for x in nt if 64 < x <= 128], "big", [int(x) for x in nt if x > 128], ok, d, flush=True)
sub.HOT_M = {}
print("bad", bad, "nonbitwise", nd)
