"""standalone: slack kernel on reference routings (scored loads + test shapes), check_routing pass rate, distinct saved."""
import sys, torch
sys.path.insert(0, "."); import reference as R
from torch.utils.cpp_extension import load_inline
src = open("./slack_src.cu").read()
lib = load_inline(name="slk_v2a", cpp_sources="void slack_launch(torch::Tensor, int64_t, int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, double, double, double, double, double, double, double);",
                  cuda_sources=src, functions=["slack_launch"], extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"])
NOISE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
MR, MK = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 else (0.005, 0.005)
TIE = float(sys.argv[6]) if len(sys.argv) > 6 else 2e-3
DM = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0
RM = float(sys.argv[8]) if len(sys.argv) > 8 else 0.0
NSEED = int(sys.argv[4]) if len(sys.argv) > 4 else 4
SCR = torch.zeros(5 * 256 + 32, dtype=torch.int32, device='cuda')
W = R.generate_weights(R.WEIGHTS_SEED)
import itertools
_fs = open("/home/daytona/r-fresh/muse_decode_layer/tools/fresh/slack/slack.py").read()
_fs = _fs[_fs.index("def min_distinct"):_fs.index("W0 = ")]
exec(_fs)
EXACT = len(sys.argv) > 5
LOADS = [("d1", 1, 1), ("d16", 16, 1), ("d32", 32, 1), ("v1", 1, 8), ("v16", 16, 8), ("v32", 32, 8)]
bias = W["router_bias"].float().contiguous()
tot = {}
for name, b, ql in LOADS:
    for layer in ("sliding", "full"):
        case = dict(batch=b, ctx=80000, layer=layer, page=1, q_len=ql, dist="soft", spread="agentic" if b > 1 else "fixed")
        for s in range(NSEED):
            seed = 801 + 13 * s
            ls = R.case_fields(case | {"seed": seed}, W)["local_start"]
            data = R.generate_input(**case, seed=seed, local_start=ls)
            fr = R.ref_front(data, W)
            logits = (fr["h"].double() @ W["router"].double().T).float()
            if NOISE:
                logits = logits * (1 + NOISE * torch.randn_like(logits))
            T = logits.shape[0]
            key = torch.sigmoid(logits * R.ROUTER_MULTIPLIER) + bias
            ids = key.topk(8, dim=-1).indices.int().contiguous()
            sc = torch.sigmoid(logits * R.ROUTER_MULTIPLIER).gather(1, ids.long())
            wts = (sc / (sc.sum(-1, keepdim=True) + 1e-15)).contiguous()
            lids = torch.where((ids - ls) % 256 < 32, (ids - ls) % 256, -1).int().contiguous()
            lst = torch.tensor([ls], dtype=torch.int32, device="cuda")
            r0, d0 = R._local_load(ids.long(), ls)
            ex = min_distinct(fr['biased'], ls)[2] if EXACT else 0
            lib.slack_launch(logits.contiguous(), 1, 0, 256, bias, lst, ids, lids, wts, SCR, T, R.ROUTER_MULTIPLIER, 1e-15,
                             0.05 - MR, 0.03 - MK, TIE, DM, RM)
            err = R.check_routing(ids, fr["biased"], ls)
            r1, d1 = R._local_load(ids.long(), ls)
            ok_l = bool((lids == torch.where((ids - ls) % 256 < 32, (ids - ls) % 256, -1)).all())
            pk = fr["scores"].gather(1, ids.long()); tw = pk / (pk.sum(-1, keepdim=True) + 1e-15)
            werr = float((tw - wts).abs().max())
            t = tot.setdefault((name, layer), [0, 0, 0.0, 0.0, 0.0])
            t[4] += d0 - ex
            t[0] += 1; t[1] += err is None and ok_l and werr < 1e-4; t[2] += d0 - d1; t[3] += r0 - r1
            if err or not ok_l or werr >= 1e-4:
                print(name, layer, seed, err, ok_l, werr)
for k, t in tot.items():
    print(f"{k[0]:>3} {k[1][:2]} pass {t[1]}/{t[0]} saved distinct {t[2]/t[0]:.2f} routes {t[3]/t[0]:.1f} exact {t[4]/t[0]:.2f}")
