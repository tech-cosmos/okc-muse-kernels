"""layer s73: moe n20 S6 (6-bit scale codes) vs 8-bit SQ stages, bitwise, tk2f (84204 + BAL) and tk2h (W+1 = argv[2]).
python tools/s6_eq.py <reps> <tk2h W+1> on variants/s6keep.py (TK2F_S6_KEEPSQ = True)."""
import sys, torch
sys.path.insert(0, ".")
import bench
sub = bench.load_sub("variants/s6keep.py")
w, s = bench.weights_state(sub)
st = s["moe"]
assert "w13s6" in st and "w13sq" in st
g = torch.Generator(device="cuda").manual_seed(5)
bad = tot = 0
for h in (None, int(sys.argv[2])):
    if h is not None: sub._moe_TK2H_M.update({16: h, 32: h})
    for T in (1, 8, 16, 32):
        for r in range(int(sys.argv[1])):
            x = torch.randn(T, 4096, device="cuda", generator=g).bfloat16()
            ids = torch.randint(0, 40 if r % 2 else 12, (T, 8), device="cuda", generator=g, dtype=torch.int32)
            ids = torch.where(ids >= 32, torch.full_like(ids, -1), ids)
            tw = torch.rand(T, 8, device="cuda", generator=g)
            d = {"hidden_states": x, "topk_ids": ids, "topk_weights": tw, "config": {"local_start": st["start"], "local_count": st["count"]}}
            sub._moe_TK2F_S6_ON = True; a = sub._moe_kernel_fn(d, st)
            sub._moe_TK2F_S6_ON = False; b = sub._moe_kernel_fn(d, st)
            torch.cuda.synchronize(); tot += 1
            if not torch.equal(a, b): bad += 1; print("MISMATCH", h, T, r)
print("bad", bad, "of", tot)
