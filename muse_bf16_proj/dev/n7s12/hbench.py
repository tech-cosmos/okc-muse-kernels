# Humming (vllm-project/humming, cloned to /tmp/humming) dense NVFP4-like GEMM on our expert dims.
# Weights E2M1, E4M3 group-16 scales (= NVFP4 without the per-tensor FP32 scale), BF16 activations/out.
# Timing: CUDA-graph replay, cold L2 (256 MB read before each replay), median, under /tmp/okc-gpu.lock.
# Usage: PYTHONPATH=/tmp/humming:/tmp/hstub python hbench.py N K [M ...]
import fcntl, json, sys
import torch
from humming.layer import HummingLayer
from humming.tune import get_heuristics_config
from humming.config import GemmType

N, K = int(sys.argv[1]), int(sys.argv[2])
MS = [int(a) for a in sys.argv[3:]] or [1, 8, 16, 32, 64, 128, 256]
dt = torch.bfloat16
layer = HummingLayer(shape_n=N, shape_k=K,
                     weight_config={"dtype": "float4e2m1", "group_size": 16, "scale_dtype": "float8e4m3"},
                     torch_dtype=dt).cuda()
w = torch.randn(N, K, dtype=dt, device="cuda") * 0.02
layer.load_from_unquantized(w)
layer.transform()
wbytes = sum(t.nbytes for t in layer.state_dict().values())
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
cc = json.dumps({"gemm_type": "dense"})


def bench(fn, reps=200):
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


for M in MS:
    x = torch.randn(M, K, dtype=dt, device="cuda")
    tc = json.dumps(get_heuristics_config(layer_config=layer.humming_config, gemm_type=GemmType.DENSE))
    out = torch.empty(M, N, dtype=dt, device="cuda")
    run = lambda: layer(inputs=x, outputs=out, compute_config=cc, tuning_config=tc)
    run(); torch.cuda.synchronize()
    ref = x.float() @ w.float().T
    rel = ((out.float() - ref).norm() / ref.norm()).item()
    us = bench(run)
    # torch BF16 reference GEMM on the unquantized weight (cuBLAS), same timing
    tb = bench(lambda: torch.mm(x, w.T))
    print(f"N={N} K={K} M={M:4d} humming {us:7.2f} us  {wbytes / us * 1e-6:5.2f} TB/s(W)  rel_err {rel:.3f}"
          f" | cuBLAS bf16 {tb:7.2f} us", flush=True)
