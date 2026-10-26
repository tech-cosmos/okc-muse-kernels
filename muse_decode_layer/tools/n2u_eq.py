"""n2u (fused moe_norm2 + unpool slice) vs moe_norm2 -> gemv17: bitwise, on T = 1 inputs over many seeds."""
import sys, torch
sys.path.insert(0, ".")
import bench
sub = bench.load_sub(sys.argv[1] if len(sys.argv) > 1 else "submission.py")
w, s = bench.weights_state(sub)
TOGGLES = [eval(a) for a in sys.argv[2:]] or [(1,), ()]
LAYERS = ("full",) if "FULL" in __import__("os").environ else ("sliding", "full")
NSEED = int(__import__("os").environ.get("NSEED", 40))
bad = 0
n = 0
for layer in LAYERS:
    for seed in range(NSEED):
        c = dict(batch=1, ctx=3000 + seed * 97, layer=layer, page=1, q_len=1, dist="soft")
        d = bench.make_input(c, w, 1000 + seed)
        outs = []
        for toks in TOGGLES:
            if hasattr(sub, "N2U_TOKENS"): sub.N2U_TOKENS = toks
            o = sub.kernel_fn(d, s)
            torch.cuda.synchronize()
            outs.append([t.clone() for t in o] if isinstance(o, (tuple, list)) else [o.clone()])
        same = all(torch.equal(a, b) for a, b in zip(*outs))
        bad += not same
        if not same: print(layer, seed, [(a - b).abs().max().item() if a.is_floating_point() else (a != b).sum().item() for a, b in zip(*outs)])
        n += 1
print("bad", bad, "of", n)
