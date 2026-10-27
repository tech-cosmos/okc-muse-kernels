# n20: S6 fallback: one w2 scale byte below 80 -> prepare builds no S6 images, tk2f runs the 8-bit SQ stages; check ref.
import sys, torch
sys.path.insert(0, '.')
import reference, submission as S
w = reference.generate_weights(reference.WEIGHTS_SEED)
w["w2_sf"].view(torch.uint8)[3, 100, 7] = 0x48  # 4.0: still positive normal (SA), outside the S6 window
st = S.prepare(w)
print("s6 built:", "w13s6" in st, "sq built:", "w13sq" in st, "cfg16", S._tk2f_cfg(16, st))
for bs in (1, 16, 32):
    d = reference.generate_input(bs=bs, sharding="ep8", seed=55 + bs)
    print(bs, reference.check_implementation(d, S.kernel_fn(d, st), w))
