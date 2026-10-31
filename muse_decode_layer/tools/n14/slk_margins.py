"""layer n14s1: routing-slack savings at T=1 / T=8 vs the selection margins (moe's slack_kernel, run directly on
the reference's own logits: measures how many local experts each margin setting saves, not noise safety).
python tools/n14/slk_margins.py [n_inputs] [router_seeds]"""
import sys, torch
sys.path.insert(0, ".")
import bench, reference as R
from torch.utils.cpp_extension import load_inline
n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
rss = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [11, 12, 13]
src = open("slack_src.cu").read()
lib = load_inline(name="slk_margins_x", cpp_sources="void slack_launch(torch::Tensor, int64_t, int64_t, int64_t, torch::Tensor, torch::Tensor, "
                  "torch::Tensor, torch::Tensor, torch::Tensor, int64_t, double, double, double, double, double, double, double);",
                  cuda_sources=src, functions=["slack_launch"], extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"])
MULT, REPS = 0.1767766952966369, 1e-15
# (label, rt, kt, tie, dmarg, rmarg)
SETS = [("moe", .027, .008, 0.0, 2.0, 8.0), ("tie.02 m0", .027, .008, .02, 0.0, 0.0), ("tie.03 m0", .027, .008, .03, 0.0, 0.0),
        ("tie.02 m1", .027, .008, .02, 1.0, 1.0), ("kt.015 tie.02", .027, .015, .02, 0.0, 0.0), ("nomargin", .05, .03, 0.0, 0.0, 0.0)]
for load, c in bench.SCORED:
    if c["layer"] != "sliding" or c["batch"] != 1:
        continue
    tot = {s[0]: [0.0, 0.0, 0] for s in SETS}; dref = 0
    for rs in rss:
        w = R.generate_weights(R.WEIGHTS_SEED, router_seed=rs)
        for i in range(n):
            d = bench.make_input(c, w, 9001 + 31 * i + 1000 * rs)
            fr = R.ref_front(d, w); ls = fr["local_start"]
            lg = (fr["h"].double() @ w["router"].double().T).float().contiguous()
            T = lg.shape[0]
            bias = w["router_bias"].float().contiguous()
            key = torch.sigmoid(lg * MULT) + bias
            top = key.topk(8, dim=-1).indices.int()
            r0, d0 = R._local_load(top.long(), ls); dref += d0
            lst = torch.tensor([ls], dtype=torch.int32, device="cuda")
            for lab, rt, kt, tie, dm, rm in SETS:
                ids = top.clone().contiguous(); loc = (ids - ls) % 256; lids = torch.where(loc < 32, loc, -1).int()
                wts = torch.zeros(T, 8, device="cuda")
                lib.slack_launch(lg, 1, 0, 256, bias, lst, ids, lids, wts, T, MULT, REPS, rt, kt, tie, dm, rm)
                err = R.check_routing(ids, fr["biased"], ls)
                r1, d1 = R._local_load(ids.long(), ls)
                tot[lab][0] += d0 - d1; tot[lab][1] += r0 - r1; tot[lab][2] += err is not None
    m = n * len(rss)
    print(f"{load} T={T}: ref distinct {dref / m:.2f}  " + "  ".join(f"{k}: -{v[0] / m:.2f}d -{v[1] / m:.2f}r f{v[2]}" for k, v in tot.items()), flush=True)
