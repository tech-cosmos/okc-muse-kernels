import sys, gc, torch
sys.path.insert(0, ".")
import reference, bench, nvfp4
D, E = reference.D_EXPERT, reference.N_EXPERTS
def gen(seed, wide):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w13, s13, g13 = nvfp4.random_weights(2*D, reference.D_HIDDEN, generator=g, device="cuda", leading=(E,), wide_scales=wide)
    w2, s2, g2 = nvfp4.random_weights(reference.D_HIDDEN, D, generator=g, device="cuda", leading=(E,), wide_scales=wide)
    return {"w13": w13, "w13_sf": s13, "w13_gs": g13.expand(E).contiguous(), "w2": w2, "w2_sf": s2,
            "w2_gs": g2.expand(E).contiguous(),
            "post_norm_w": (0.1*torch.randn((reference.D_HIDDEN,), generator=g, device="cuda")).to(torch.bfloat16)}
seed, wide = int(sys.argv[1]), int(sys.argv[2])
bench._W = gen(seed, wide == 1)
if wide == 2:
    W = bench._W; v = W["w13_sf"].view(torch.uint8); v[:, 0, 0] = 0; v[:, 7, 3] = 1; W["w2_sf"].view(torch.uint8)[:, 5, 7] = 3
bench._ST = bench.submission.prepare(bench._W)
print("variant", seed, wide, "sa", bench._ST.get("sa"), flush=True)
print("BAD", bench.test())
