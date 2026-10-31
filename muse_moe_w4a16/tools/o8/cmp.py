import sys, torch
sys.path.insert(0, "."); import reference as R
from torch.utils.cpp_extension import load_inline
decl = "void slack_launch(torch::Tensor, int64_t, int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, double, double, double, double, double, double, double, int64_t);"
libs = [load_inline(name="slk_o8_" + n, cpp_sources=decl, cuda_sources=open(f"/tmp/o8/{n}.cu").read(), functions=["slack_launch"],
                    extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"]) for n in ("base", sys.argv[1])]
W = R.generate_weights(R.WEIGHTS_SEED); bias = W["router_bias"].float().contiguous()
tot = bad = ch = 0
for name, b, ql in [("d1", 1, 1), ("d16", 16, 1), ("d32", 32, 1), ("v1", 1, 8), ("v16", 16, 8), ("v32", 32, 8)]:
  for seed in range(801, 806):
    case = dict(batch=b, ctx=80000, layer="sliding", page=1, q_len=ql, dist="soft", spread="agentic" if b > 1 else "fixed")
    lsv = R.case_fields(case | {"seed": seed}, W)["local_start"]
    data = R.generate_input(**case, seed=seed, local_start=lsv)
    h = R.ref_front(data, W)["h"].double()
    lgf = (h @ W["router"].double().T).float()
    for ks in (1, 2, 8):
     for mg in ((0.05, 0.03, 0.0, 0.0, 0.0), (0.027, 0.008, 0.0, 2.0, 8.0), (0.04, 0.02, 0.01, 1.0, 4.0)):
       if ks == 1: part, sks, st = lgf.contiguous(), 0, 256
       elif ks == 8:
         ps = [(h @ W["router"].double().T * (0.1 + 0.02 * i)).float() for i in range(7)]; ps.append(lgf - sum(ps[1:], ps[0])); part, sks, st = torch.cat(ps, 0).contiguous(), ps[0].numel(), 256
       else:
         p0 = (h @ W["router"].double().T * 0.37).float(); p1 = (lgf - p0)
         part = torch.cat([p0, p1], 0).contiguous(); sks, st = p0.numel(), 256
       T = lgf.shape[0]
       key = torch.sigmoid(lgf * R.ROUTER_MULTIPLIER) + bias
       ids0 = key.topk(8, -1).indices.int().contiguous()
       ls = torch.tensor([lsv], dtype=torch.int32, device="cuda")
       outs = []; print(name, seed, ks, mg, flush=True)
       for lib in libs:
         ids, lids, wts = ids0.clone(), torch.zeros_like(ids0), torch.zeros(T, 8, device="cuda")
         SCR = torch.zeros(5 * 256 + 32, dtype=torch.int32, device="cuda")
         lib.slack_launch(part, ks, sks, st, bias, ls, ids, lids, wts, SCR, T, R.ROUTER_MULTIPLIER, 1e-20, *mg, 0)
         torch.cuda.synchronize(); outs.append((ids, lids, wts))
       tot += 1; ch += int((outs[0][0] != ids0).any(-1).sum())
       same = all(torch.equal(x, y) for x, y in zip(outs[0], outs[1]))
       bad += not same
       if not same: print("DIFF", name, seed, ks, mg)
print(f"cases {tot}, differing {bad}, changed tokens (old) {ch}")
