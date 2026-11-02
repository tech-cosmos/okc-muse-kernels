import sys, fcntl, statistics, torch
sys.path.insert(0, "."); import reference as R
from torch.utils.cpp_extension import load_inline
src = open("./slack_src.cu").read()
lib = load_inline(name="slk_v2b", cpp_sources="void slack_launch(torch::Tensor, int64_t, int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, double, double, double, double, double, double, double);",
                  cuda_sources=src, functions=["slack_launch"], extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"])
SCR = torch.zeros(5 * 256 + 32, dtype=torch.int32, device='cuda')
W = R.generate_weights(R.WEIGHTS_SEED); bias = W["router_bias"].float().contiguous()
LOADS = [("d1", 1, 1), ("d16", 16, 1), ("d32", 32, 1), ("v1", 1, 8), ("v16", 16, 8), ("v32", 32, 8)]
for name, b, ql in LOADS:
    case = dict(batch=b, ctx=80000, layer="sliding", page=1, q_len=ql, dist="soft", spread="agentic" if b > 1 else "fixed")
    lsv = R.case_fields(case | {"seed": 801}, W)["local_start"]
    data = R.generate_input(**case, seed=801, local_start=lsv)
    fr = R.ref_front(data, W)
    lg = (fr["h"].double() @ W["router"].double().T).float().contiguous(); T = lg.shape[0]
    key = torch.sigmoid(lg * R.ROUTER_MULTIPLIER) + bias
    ids0 = key.topk(8, -1).indices.int().contiguous()
    ids, lids, wts = ids0.clone(), torch.zeros_like(ids0), torch.zeros(T, 8, device="cuda")
    ls = torch.tensor([lsv], dtype=torch.int32, device="cuda")
    f = lambda: (ids.copy_(ids0), lib.slack_launch(lg, 1, 0, 256, bias, ls, ids, lids, wts, SCR, T, R.ROUTER_MULTIPLIER, 1e-15, 0.027, 0.008, 0.0, 2.0, 8.0))
    g0 = torch.cuda.CUDAGraph(); g = torch.cuda.CUDAGraph()
    f(); torch.cuda.synchronize()
    with torch.cuda.graph(g0): ids.copy_(ids0)
    with torch.cuda.graph(g): f()
    res = {}
    with open("/tmp/okc-gpu.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for nm, gg in (("copy", g0), ("copy+slack", g)):
            ts = []
            for _ in range(200):
                a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                a.record(); gg.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1000)
            res[nm] = statistics.median(ts)
    print(f"{name} T={T}: changed {int((ids != ids0).any(-1).sum())} slack kernel ~{res['copy+slack'] - res['copy']:.2f} us (graph copy {res['copy']:.2f})", flush=True)
