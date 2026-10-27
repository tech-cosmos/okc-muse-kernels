# n22: one warm call (1 expert, T tokens) of the default decode path for ncu pipe metrics. DQW=<0|1> T=<n>
import sys, os, torch
sys.path.insert(0, '.')
import bench, submission as S
S.DQW = int(os.environ.get("DQW", "0"))
bench.submission = S
w, st = bench.state()
st["lib"] = S._lib()
st["w13sq"] = S._sq_pack(st["w13q"], st["w13s2"]); st["w2sq"] = S._sq_pack(st["w2q"], st["w2s2"])
T = int(os.environ.get("T", "32"))
ids = torch.full((T, 8), (st["start"] + st["count"] + 5) % 256, dtype=torch.int32, device='cuda')
ids[:, 0] = st["start"]
tw = torch.rand(T, 8, device='cuda'); tw = tw / tw.sum(-1, keepdim=True)
d = {"hidden_states": torch.randn(T, 4096, device='cuda').bfloat16(), "topk_ids": ids, "topk_weights": tw,
     "config": {"bs": T, "d_hidden": 4096, "local_start": st["start"], "local_count": st["count"]}}
for _ in range(3): S.kernel_fn(d, st)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart(); S.kernel_fn(d, st); torch.cuda.synchronize(); torch.cuda.cudart().cudaProfilerStop()
