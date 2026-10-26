"""Test dev/pr8.cu (fused pooler + router + routing at 8 tokens, the layer's T=8 side branch).
  python dev/pr8.py "8,8,4,128,1,0;..." [reps]   configs (kcu, wg, pf, bk, xp, npf)
Checks pooled vs FP64 (bf16 rounding), logits vs FP64, routing vs a torch replica of the layer's _routing, many seeds."""
import os, sys, re, math, hashlib, torch
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(D))
from torch.utils.cpp_extension import load_inline

tmpl = open(os.path.join(D, "sub_template.py")).read()
ns = {"torch": torch}
exec("G13_ESC = 8\n" + re.search(r"(def _compress\(w\):.*?)\ndef _pack16", tmpl, re.S).group(1), ns)
_pack14 = ns["_pack14"]

cfgs = [tuple(map(int, c.split(","))) for c in sys.argv[1].split(";")]
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 20
MULT, REPS, LOCAL, E = 0.1767766952966369, 1e-15, 32, 256


def build(cfgs):
    src = open(os.path.join(D, "pr8.cu")).read().replace(
        "PR8_CONFIGS", " ".join("PR8_CASE(%d,%d,%d,%d)" % (c[0], c[1], c[2], c[4]) for c in sorted(set(cfgs))))
    return load_inline(name="pj_pr8_" + hashlib.md5(src.encode()).hexdigest()[:10],
                       cpp_sources=PR8_DECL, cuda_sources=src, functions=["pr8_launch"],
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas=-v"] + (["-DPR8_TRACE"] if os.environ.get("TRACE") else []))


PR8_DECL = ("void pr8_launch(torch::Tensor, torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor, torch::Tensor, "
            "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
            "torch::Tensor, double, double, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);")


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
    pk = _pack14(Wp, 8192)
    assert pk is not None
    bias = torch.randn(256, generator=g, device="cuda") * 0.01
    logits = torch.empty(8, 256, device="cuda"); cnt = torch.zeros(32, dtype=torch.int32, device="cuda")
    part = torch.zeros(8 * 8 * 256 + 128 * 16, device="cuda")
    bad = 0
    for c in cfgs:
        for rep in range(reps):
            h32 = torch.randn(8, 8192, generator=g, device="cuda") * (1 + rep % 3)
            h = h32.bfloat16()
            ls = torch.tensor([(rep * 37) % 256], dtype=torch.int32, device="cuda")
            out = torch.empty(8, 4096, dtype=torch.bfloat16, device="cuda")
            ids = torch.empty(8, 8, dtype=torch.int32, device="cuda"); lids = torch.empty_like(ids)
            wts = torch.empty(8, 8, device="cuda")
            mod.pr8_launch(pk[0], pk[1], pk[2], pk[3], h, out, Wr, h32, part, logits, cnt, bias, ls, ids, lids, wts,
                           MULT, REPS, LOCAL, *c, 0)
            torch.cuda.synchronize()
            pref = (h.double() @ Wp.double().t())
            perr = ((out.double() - pref).abs().max() / pref.abs().max()).item()
            lref = (h32.double() @ Wr.double().t())
            lerr = ((logits.double() - lref).abs().max() / lref.abs().max()).item()
            ok = perr < 4e-3 and lerr < 1e-6 and int(cnt.abs().sum()) == 0
            for tk in range(8):
                rid, rlid, rw = routing_ref(logits[tk].clone(), bias, int(ls))
                ok = ok and torch.equal(ids[tk], rid) and torch.equal(lids[tk], rlid) and torch.allclose(wts[tk], rw, rtol=1e-6, atol=0)
            bad += not ok
            if not ok or rep == 0:
                print(c, rep, "OK" if ok else "BAD", f"pooled {perr:.1e} logits {lerr:.1e}", ids[0].tolist(), rid.tolist(),
                      int(cnt.abs().sum()), flush=True)
    print("bad", bad)
    if os.environ.get("TIME"):
        import bench
        e = bench.empty_cost()
        bench._SINK = torch.zeros((), dtype=torch.int64, device="cuda")
        for c in cfgs:
            for npf in (c[5], -1, -2):
                cc = c[:5] + (npf,)
                t = bench.time_graph(lambda: mod.pr8_launch(pk[0], pk[1], pk[2], pk[3], h, out, Wr, h32, part, logits, cnt, bias, ls,
                                                            ids, lids, wts, MULT, REPS, LOCAL, *cc, 0)) - e
                print("time", cc, f"{t:.2f} us", flush=True)
    if os.environ.get("TRACE"):
        for npf in (cfgs[0][5], -2):
            for _ in range(3):
                torch.sum(bench._FLUSH.view(torch.int64), 0, out=bench._SINK)
                mod.pr8_launch(pk[0], pk[1], pk[2], pk[3], h, out, Wr, h32, part, logits, cnt, bias, ls, ids, lids, wts,
                               MULT, REPS, LOCAL, *(cfgs[0][:5] + (npf,)), 0)
            torch.cuda.synchronize()
            tr = part[8 * 8 * 256:].view(torch.int64).view(128, 8)[:, :6].double()
            tr = (tr - tr[:, 0].min()) / 1000
            q = lambda v: f"{v.min().item():6.1f} {v.median().item():6.1f} {v.max().item():6.1f}"
            print("npf", npf, " | ".join(f"t{i} {q(tr[:, i])}" for i in range(6)))
            print("  slowest blocks (t3-t2):", (tr[:, 3] - tr[:, 2]).topk(5).values.tolist(), (tr[:, 3] - tr[:, 2]).topk(5).indices.tolist())
