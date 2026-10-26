"""H100: tinygemm2 (dev/tinygemm2.cu) after a 3 us PDL predecessor (g19_pred: triggers at start), the layer's case.
  python dev/tg2pdl.py [m=8]   -> pred alone, pred+pick, pred+tg2 (no PDL), pred+tg2 PDL (W streams before the wait)"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench
S = bench.submission
m = int(sys.argv[1]) if len(sys.argv) > 1 else 8
w, st = bench.state()
W = st["unpooler"][:1024]
xs = torch.randn(m, 4096, device="cuda").to(torch.bfloat16); x = xs.clone()
out = torch.empty(m, 1024, dtype=torch.bfloat16, device="cuda")
pick = eval(os.environ["PICK"]) if "PICK" in os.environ else st["pick"].get(("unpooler", m))
ref = (xs.float() @ W.float().t())
for pdl in (False, True):
    out.zero_(); S._g19.g19_pred(xs, x, 3000); S._tg2.tg2(x, W, out, pdl); torch.cuda.synchronize()
    print("pdl", pdl, "err", ((out.float() - ref).norm() / ref.norm()).item())
for _ in range(2):
    med = S._graph_medians([lambda: S._g19.g19_pred(xs, x, 3000),
                            lambda: (S._g19.g19_pred(xs, x, 3000), S._run(st, "unpooler", x, out, pick)),
                            lambda: (S._g19.g19_pred(xs, x, 3000), S._tg2.tg2(x, W, out, False)),
                            lambda: (S._g19.g19_pred(xs, x, 3000), S._tg2.tg2(x, W, out, True))])
    p = med[0] * 1e3
    print(f"m{m} pred {p:.2f} us; +pick {pick} {med[1]*1e3-p:+.2f}; +tg2 {med[2]*1e3-p:+.2f}; +tg2 PDL {med[3]*1e3-p:+.2f}")
