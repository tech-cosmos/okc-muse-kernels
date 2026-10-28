"""n5s29: per-CTA timeline of the PR tk2f in the layer (variant with TK2PR_TL): python dev/n5s29/pr_tl.py <sub> <W> <seed> [load]"""
import sys, torch
sys.path.insert(0, ".")
import bench
sub = bench.load_sub(sys.argv[1]); W = int(sys.argv[2]); sd = int(sys.argv[3]); ld = sys.argv[4] if len(sys.argv) > 4 else "decode 16"
w, st = bench.weights_state(sub)
c = [c for l, c in bench.SCORED if l == ld and c["layer"] == "sliding"][0]
data = bench.make_input(c, w, sd)
T = c["batch"] * c["q_len"]
sub._ab(T, tk2pr=W)
ids = None
for _ in range(3): sub.kernel_fn(data, st)
torch.cuda.synchronize()
o = torch.zeros(4 * 160, dtype=torch.int64, device="cuda")
sub._moe__lib().moe_prt_read(o); torch.cuda.synchronize()
o = o.view(160, 4)[:132].cpu()
t0 = o[:, 0].min().item()
pr = o[:, 3] >= 1000; k = int(o[0, 3]) % 1000
for name, m in (("single", ~pr), ("pair", pr)):
    if m.sum() == 0: continue
    a = (o[m, 0] - t0) / 1e3; b = (o[m, 1] - t0) / 1e3; e = (o[m, 2] - t0) / 1e3
    print(f"{name:6s} n={int(m.sum()):3d} k={k}  start {a.min():6.2f}-{a.max():6.2f}  g13 end {b.min():6.2f}/{b.median():6.2f}/{b.max():6.2f}  end {e.min():6.2f}/{e.median():6.2f}/{e.max():6.2f} us")
