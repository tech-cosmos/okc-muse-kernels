"""Test/time dev/pr1.cu (fused pooler + router + routing at 1 token, the layer's T=1 side branch).
  python dev/pr1.py "2,4,1,4,16,132;..." [reps]   configs (u, nv, pf, r, rb, blocks, npf)
Checks pooled vs FP64 (bf16 rounding), logits vs FP64, routing vs a torch replica of the layer's _routing, many seeds."""
import os, sys, re, math, hashlib, torch
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(D))
from torch.utils.cpp_extension import load_inline

tmpl = open(os.path.join(D, "sub_template.py")).read()
ns = {"torch": torch}
exec("G13_ESC = 8\n" + re.search(r"(def _compress\(w\):.*?)\ndef _pack14", tmpl, re.S).group(1), ns)
_pack13 = ns["_pack13"]

cfgs = [tuple(map(int, c.split(","))) for c in sys.argv[1].split(";")]
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 20
MULT, REPS, LOCAL, E = 0.1767766952966369, 1e-15, 32, 256


def build(cfgs):
    src = open(os.path.join(D, "pr1.cu")).read().replace(
        "PR1_CONFIGS", " ".join("PR1_CASE(%d,%d,%d,%d)" % c[:4] for c in sorted(set(c[:4] for c in cfgs))))
    return load_inline(name="pj_pr1_" + hashlib.md5(src.encode()).hexdigest()[:10],
                       cpp_sources="void pr1_launch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, "
                       "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                       "torch::Tensor, torch::Tensor, double, double, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);",
                       cuda_sources=src, functions=["pr1_launch"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])


def routing_ref(logits, bias, ls):
    score = 1.0 / (1.0 + torch.exp(-logits * MULT))
    key = score + bias
    ids = []
    k = key.clone()
    for _ in range(8):
        best = k.max()
        idx = int(torch.nonzero(k == best)[0])
        ids.append(idx)
        k[idx] = -math.inf
    ids = torch.tensor(ids, device=logits.device)
    picked = score[ids]
    wts = picked / (picked.sum() + REPS)
    loc = (ids - ls + E) % E
    return ids.int(), torch.where(loc < LOCAL, loc, -1).int(), wts


if __name__ == "__main__":
    mod = build(cfgs)
    g = torch.Generator(device="cuda").manual_seed(1)
    Wp = (torch.randn(4096, 8192, generator=g, device="cuda") / math.sqrt(8192)).bfloat16()
    Wr = torch.randn(256, 8192, generator=g, device="cuda") / math.sqrt(8192)
    pk = _pack13(Wp)
    assert pk is not None
    bias = torch.randn(256, generator=g, device="cuda") * 0.01
    logits = torch.empty(256, device="cuda"); cnt = torch.zeros(4, dtype=torch.int32, device="cuda")
    bad = 0
    for c in cfgs:
        for rep in range(reps):
            h32 = torch.randn(1, 8192, generator=g, device="cuda") * (1 + rep % 3)
            h = h32.bfloat16()
            ls = torch.tensor([(rep * 37) % 256], dtype=torch.int32, device="cuda")
            out = torch.empty(1, 4096, dtype=torch.bfloat16, device="cuda")
            ids = torch.empty(1, 8, dtype=torch.int32, device="cuda"); lids = torch.empty_like(ids)
            wts = torch.empty(1, 8, device="cuda")
            mod.pr1_launch(pk[0], pk[1], pk[2], h, out, pk[3], Wr, h32, logits, cnt, bias, ls, ids, lids, wts,
                           MULT, REPS, LOCAL, *c, 0)
            torch.cuda.synchronize()
            pref = (h.double() @ Wp.double().t())
            perr = ((out.double() - pref).abs().max() / pref.abs().max()).item()
            lref = (h32.double() @ Wr.double().t())[0]
            lerr = ((logits.double() - lref).abs().max() / lref.abs().max()).item()
            rid, rlid, rw = routing_ref(logits.clone(), bias, int(ls))
            ok = perr < 4e-3 and lerr < 1e-6 and torch.equal(ids[0], rid) and torch.equal(lids[0], rlid) \
                and torch.allclose(wts[0], rw, rtol=1e-6, atol=0) and int(cnt[0]) == 0
            bad += not ok
            if not ok or rep == 0:
                print(c, rep, "OK" if ok else "BAD", f"pooled {perr:.1e} logits {lerr:.1e}", ids[0].tolist(), rid.tolist(),
                      (wts[0] - rw).abs().max().item(), int(cnt[0]), flush=True)
    print("bad", bad)
