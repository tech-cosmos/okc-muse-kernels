"""Paired H100 A/B of the csl/vtk kernels (decode 1 / verify 1 sliding) from two submission files, cold graph replays.
python tools/s31/csl_ab.py A.py B.py [rounds]   (run from muse_attention_decode/)"""
import hashlib, re, statistics, sys, torch
sys.path.insert(0, "."); sys.path.insert(0, "tools")
from torch.utils.cpp_extension import load_inline
import reference, bench
CPP = ("void csl_decode(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
       "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, double);"
       "void vtk_verify(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
       "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, double);")
mods = []
for f in sys.argv[1:3]:
    src = re.search(r"_CSL_SRC = r'''(.*?)'''", open(f).read(), re.S).group(1)
    h = hashlib.md5(src.encode()).hexdigest()[:8]
    mods.append(load_inline(name="cslab_" + h, cpp_sources=CPP, cuda_sources=src, functions=["csl_decode", "vtk_verify"],
                            extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"], extra_cflags=["-O3"]))
import submission
R = int(sys.argv[3]) if len(sys.argv) > 3 else 7
for L in (1, 8):
    data = reference.generate_input(seed=99, batch=1, ctx=80000, heads="tp8", layer="sliding", page=1, q_len=L, dist="soft")
    cfg = data["config"]; dev = data["qkv"].device
    cs = submission._cos_sin(cfg["rope_theta"], 64, dev)
    outs = [torch.empty((1, L, 1024), dtype=torch.bfloat16, device=dev) for _ in mods]
    def mk(i):
        fn = mods[i].csl_decode if L == 1 else mods[i].vtk_verify
        return lambda: fn(data["qkv"], cs, data["k_cache"], data["v_cache"], data["page_table"], data["seq_lens"],
                          data["cache_loc"], data["k_scale"], data["v_scale"], outs[i], 1,
                          cfg["sm_scale"] * cfg["qk_scale_factor"] * submission.LOG2E)
    res = [[], []]
    for r in range(R):
        for i in ((0, 1) if r % 2 == 0 else (1, 0)):
            res[i].append(bench.time_graph(mk(i), reps=100) - bench.empty_cost())
    d = (outs[0].float() - outs[1].float()).abs().max().item()
    wins = sum(b < a for a, b in zip(*res))
    print(f"L {L}: A {statistics.median(res[0]):.3f}  B {statistics.median(res[1]):.3f}  B-A {statistics.median(res[1]) - statistics.median(res[0]):+.3f} us  (B wins {wins}/{R}; max|dA-B| {d:.2e})", flush=True)
