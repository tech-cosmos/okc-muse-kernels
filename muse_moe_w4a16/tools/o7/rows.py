import sys, torch
sys.path.insert(0, ".")
import bench, reference
sub = bench.load_sub("/tmp/o7/base_p.py")
W, st = bench.weights_state(sub)
for l, c in bench.SCORED:
    if sys.argv[1] != f"{l} {c['layer']}": continue
    for sd in (801, 802, 803, 804):
        d = bench.make_input(c, W, sd)
        out = sub.kernel_fn(d, st); torch.cuda.synchronize()
        ids = out[2].long().flatten(); ls = int(d["local_start"])
        loc = (ids - ls) % 256; loc = loc[loc < 32]
        cnt = torch.bincount(loc, minlength=32)
        g = int(((cnt + 7) // 8).sum()); n13, n2 = g * 256, g * 128
        print(sd, "E", int((cnt > 0).sum()), "groups", g, "n13", n13, "extra13", n13 % 132, "n2", n2, "extra2", n2 % 132, "skip rows", (n13 % 132 if n13 >= 132 else 0) + (n2 % 132 if n2 >= 132 else 0))
