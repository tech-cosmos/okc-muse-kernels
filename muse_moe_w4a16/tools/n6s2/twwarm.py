# n6s2: twm consumer-bound proxy for ncu: T tokens spread over NE local experts (few experts -> weights mostly L2-resident
# on the H100's 50 MB L2), default big-M config. usage: python tools/n6s2/twwarm.py T NE  (profiles the 2nd call)
import sys, os, torch
sys.path.insert(0, '.')
import bench, submission
w, st = bench.state()
T, ne = int(sys.argv[1]), int(sys.argv[2])
ids = torch.full((T, 8), (st["start"] + st["count"] + 5) % 256, dtype=torch.int32, device='cuda')
ids[:, 0] = st["start"] + torch.arange(T, device='cuda') % ne
tw = torch.rand(T, 8, device='cuda'); tw = tw / tw.sum(-1, keepdim=True)
data = {"hidden_states": torch.randn(T, 4096, device='cuda').bfloat16(), "topk_ids": ids, "topk_weights": tw,
        "config": {"bs": T, "d_hidden": 4096, "local_start": st["start"], "local_count": st["count"]}}
for _ in range(3): submission.kernel_fn(data, st)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
submission.kernel_fn(data, st); torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
