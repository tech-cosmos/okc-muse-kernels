# n5s21: exact-N moe_hot probe (HOT_NX build, HOT_NXB bits) on the layer's verify 32 routings, H100, cold L2, ABAB rounds
# usage (from muse_moe_w4a16/): sed 's/^#define HOT_NX 0 /#define HOT_NX 1 /' submission.py > /tmp/n5s21/subnx.py
#        PYTHONPATH=/tmp/n5s21 SUB=subnx python tools/n5s21/nx_ab.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import layer_ab as L
S = L.submission
V = [("h32", 0), ("h32n104", 16)]
for rnd in range(int(os.environ.get("RND", "2"))):
    r = {n: [] for n, _ in V}
    for (l, sd), loc in sorted(L.R.items()):
        if l != "verify 32": continue
        data = L.mk(loc, sd)
        ref = None
        for n, b in V:
            S.HOT_M, S.HOT_NXB = {256: 32}, b
            o = S.kernel_fn(data, L.st).float()
            if ref is None: ref = o
            d = (o - ref).abs().max().item()
            if d: print("DIFF", n, sd, d)
            r[n].append(L.tg(lambda: S.kernel_fn(data, L.st)))
    print(f"r{rnd} v32: " + "  ".join(f"{n} {sum(v)/len(v):.1f} [{' '.join(f'{x:.0f}' for x in v)}]" for n, v in r.items()), flush=True)
