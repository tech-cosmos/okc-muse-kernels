import sys, torch
sys.path.insert(0, ".")
import bench
sub = bench.load_sub(sys.argv[1])
w, st = bench.weights_state(sub)
lib = sub._G10["lib"]; pk = st["g13"]["o"]
W = None
for k, v in st.items():
    pass
import reference
Wo = [t for n, t in w.items() if hasattr(t, "shape") and tuple(t.shape) == (8192, 1024)] if isinstance(w, dict) else []
print("cands", len(Wo))
for Wt in Wo:
    worst = 0
    for seed in range(5):
        g = torch.Generator(device="cuda").manual_seed(seed)
        x = torch.randn(1, 1024, device="cuda", generator=g).bfloat16()
        out = torch.empty(1, 8192, device="cuda", dtype=torch.bfloat16)
        lib.g13_gemv(pk[0], pk[1], pk[2], x, out, pk[3], *sub.G13_O, x, x, x, 0, 0, 0)
        ref = (x.double() @ Wt.double().T)
        e = ((out.double() - ref).norm() / ref.norm()).item()
        b = ((ref.bfloat16().double() - ref).norm() / ref.norm()).item()
        worst = max(worst, e / b)
    print("rel err / bf16-rounding err (worst of 5):", worst)
