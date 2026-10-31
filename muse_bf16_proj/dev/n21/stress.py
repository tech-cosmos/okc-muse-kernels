"""proj n21: bitwise stress base vs HOT_SL copy (cwd = dir with moe reference.py): eager over many seeds/shapes + graph
replay with fresh inputs copied into the captured buffers. python stress.py <base dir> <sl dir>"""
import sys, torch, importlib.util
sys.path.insert(0, ".")
import reference
def load(n, p):
    spec = importlib.util.spec_from_file_location(n, p + "/submission.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
A_, B_ = load("suba", sys.argv[1]), load("subb", sys.argv[2])
w = reference.generate_weights(reference.WEIGHTS_SEED); sta = A_.prepare(w); stb = B_.prepare(w)
bad = n = big = 0
for bs, dist, ns in ((256, "normal", 80), (257, "normal", 30), (256, "skewed", 30), (128, "skewed", 20), (200, "normal", 10), (384, "normal", 10)):
    for sd in range(ns):
        d = reference.generate_input(bs=bs, sharding="ep8", seed=5000 + sd, dist=dist)
        ids = d["topk_ids"].flatten(); big += int(torch.bincount(ids[ids < 32], minlength=32).max() > 128)
        oa = A_.kernel_fn(d, sta); ob = B_.kernel_fn(d, stb); n += 1
        if not torch.equal(oa, ob): bad += 1; print("DIFF", bs, dist, sd, (oa.float() - ob.float()).abs().max().item())
print(f"eager {n - bad}/{n} bitwise ({big} with a >128-token expert)", flush=True)
for bs in (256, 257):
    d0 = reference.generate_input(bs=bs, sharding="ep8", seed=1)
    d = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in d0.items()}
    B_.kernel_fn(d, stb); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        B_.kernel_fn(d, stb)
        with torch.cuda.graph(g, stream=s): out = B_.kernel_fn(d, stb)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    rb = rn = 0
    for sd in range(60):
        dn = reference.generate_input(bs=bs, sharding="ep8", seed=7000 + sd, dist="skewed" if sd % 3 == 2 else "normal")
        for k, v in dn.items():
            if torch.is_tensor(v): d[k].copy_(v)
        g.replay(); torch.cuda.synchronize()
        ref = A_.kernel_fn(dn, sta).clone(); rn += 1
        if not torch.equal(out, ref): rb += 1
    print(f"replay bs {bs}: {rn - rb}/{rn} bitwise", flush=True)
