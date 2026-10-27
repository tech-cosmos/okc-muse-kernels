import sys, torch
sys.path.insert(0, ".")
import bench
sub = bench.load_sub("sub_force.py")
w, st = bench.weights_state(sub)
for load, c in bench.SCORED:
    if not load.startswith("decode 32"): continue
    pb = st["g24p"][32]
    fell = []
    orig = sub._kernel_fn
    def kf(d, s):
        r = orig(d, s); fell.append(len(sub._G24P)); return r
    sub._kernel_fn = kf
    for sd in (1, 2, 3):
        d = bench.make_input(c, w, sd)
        out = sub.kernel_fn(d, st); torch.cuda.synchronize()
        print(load, c["layer"], sd, "pending after unpool:", fell[-1], "buf absmax:", pb.abs().max().item(),
              "pooler sel:", sub._g24_sel(st, "pooler", 32)[4])
    sub._kernel_fn = orig
