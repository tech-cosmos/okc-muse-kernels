# Humming INDEXED MoE kernel on the moe board's routings (r-moe reference.generate_input, ep8 busiest, 32 local
# experts, D 4096/4096, NVFP4-like: E2M1 + E4M3 g16, BF16 act). Times ONLY gemm13 + gemm2 (two Humming launches, one
# CUDA graph; no gather/silu/norm/combine: a lower bound for a Humming MoE). Cold L2, median, under the GPU lock.
# Usage: PYTHONPATH=/tmp/humming:/tmp/hstub:/tmp/n7s12/ref python hmoe.py [bs ...]
import fcntl, json, sys
import torch
from humming.layer import HummingLayer
from humming.tune import get_heuristics_config
from humming.config import GemmType
from humming.testing import random_fill_tensor
from humming.testing.data import generate_moe_tensors
import reference as R

BS = [int(a) for a in sys.argv[1:]] or [1, 8, 16, 32, 128, 256]
dt = torch.bfloat16
E, D = 32, 4096
wc = {"dtype": "float4e2m1", "group_size": 16, "scale_dtype": "float8e4m3"}
L13 = HummingLayer(shape_n=2 * D, shape_k=D, num_experts=E, weight_config=wc, torch_dtype=dt).cuda()
L2 = HummingLayer(shape_n=D, shape_k=D, num_experts=E, weight_config=wc, torch_dtype=dt).cuda()
for L in (L13, L2):
    for t in L.parameters():
        random_fill_tensor(t)
    L.transform()
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
cc = json.dumps({"gemm_type": "indexed"})


def cfg(L, n_routes):
    tc = get_heuristics_config(layer_config=L.humming_config, gemm_type=GemmType.INDEXED)
    for lo, hi, c in tc:
        if lo < n_routes <= hi:
            return json.dumps(tc), c["block_shape"][0]
    raise RuntimeError(n_routes)


def bench(fn, reps=100):
    fn(); torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            fn()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(reps)]
    with open("/tmp/okc-gpu.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for a, b in ev:
            flush.sum(dtype=torch.int32)
            a.record(); g.replay(); b.record()
        torch.cuda.synchronize()
        fcntl.flock(lk, fcntl.LOCK_UN)
    t = sorted(a.elapsed_time(b) * 1e3 for a, b in ev)
    return t[len(t) // 2]


for bs in BS:
    for seed in (7, 11):
        d = R.generate_input(bs, "ep8", seed)
        ids = d["topk_ids"].flatten()
        loc = ids[ids < E].to(torch.int32).view(-1, 1)          # one row per local (token, expert) pair
        P = loc.shape[0]
        x = torch.randn(P, D, dtype=dt, device="cuda")           # pre-gathered rows (untimed: favours Humming)
        h = torch.empty(P, 2 * D, dtype=dt, device="cuda")
        y = torch.empty(P, D, dtype=dt, device="cuda")
        a = torch.randn(P, D, dtype=dt, device="cuda")
        tc13, b13 = cfg(L13, P)
        tc2, b2 = cfg(L2, P)
        _, _, s13, e13, n13 = generate_moe_tensors(loc, E, "indexed", b13)
        _, _, s2, e2, n2 = generate_moe_tensors(loc, E, "indexed", b2)

        def run():
            L13(inputs=x, outputs=h, sorted_ids=s13, expert_ids=e13, num_tokens_padded=n13,
                compute_config=cc, tuning_config=tc13, top_k=1)
            L2(inputs=a, outputs=y, sorted_ids=s2, expert_ids=e2, num_tokens_padded=n2,
               compute_config=cc, tuning_config=tc2, top_k=1)
        n_exp = int(torch.unique(loc).numel())
        us = bench(run)
        mb = n_exp * 3 * D * D * 0.5625 / 1e6
        print(f"bs={bs:4d} seed={seed} pairs={P:4d} experts={n_exp:2d} blk={b13}/{b2} humming g13+g2 {us:7.2f} us"
              f"  ({mb:.0f} MB W, {mb / us:.2f} TB/s)", flush=True)
