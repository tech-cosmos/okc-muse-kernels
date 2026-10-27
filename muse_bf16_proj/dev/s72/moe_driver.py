# Guest driver (exec'd by proj's _diag_guest): moe DIAG submission on judge-like NVFP4 weights (nvfp4.random_weights,
# LOCAL experts only), then its prepare() runs _diag (DIAG = 1) and prints the "diag12 MEAN" lines.
import math, time
H, I, L = sub["H"], sub["I"], sub["LOCAL"]
gen = torch.Generator(device="cuda").manual_seed(5252)


def rw(rows, k):  # = nvfp4.random_weights (wide_scales False), leading = (L,)
    packed = torch.randint(0, 256, (L, rows, k // 2), dtype=torch.uint8, generator=gen, device="cuda")
    z = torch.randn((L, rows, k // 16), generator=gen, device="cuda")
    sf = torch.exp2(7.0 + 0.4 * z).clamp(8.0, 448.0).to(torch.float8_e4m3fn)
    mean_sq = 4.0 ** 7.0 * math.exp(2 * (math.log(2) * 0.4) ** 2)
    gs = torch.tensor(1.0 / (8.6 * mean_sq * k) ** 0.5, dtype=torch.float32, device="cuda")
    return packed, sf, gs


w13, w13_sf, gs13 = rw(2 * I, H)
w2, w2_sf, gs2 = rw(H, I)
w = {"w13": w13, "w13_sf": w13_sf, "w13_gs": gs13.expand(L).contiguous(), "w2": w2, "w2_sf": w2_sf,
     "w2_gs": gs2.expand(L).contiguous(),
     "post_norm_w": (0.1 * torch.randn((H,), generator=gen, device="cuda")).to(torch.bfloat16)}
assert sub["DIAG"], "guest moe submission needs DIAG = 1"
t0 = time.time()
st = sub["prepare"](w)
print(f"guest moe prepare+diag {time.time() - t0:.1f} s", flush=True)
del st, w, w13, w2, w13_sf, w2_sf
