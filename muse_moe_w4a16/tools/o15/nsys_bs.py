import sys, torch
sys.path.insert(0, "/home/daytona/r-moe/muse_moe_w4a16")
import reference, submission
W = reference.generate_weights(reference.WEIGHTS_SEED)
st = submission.prepare(W)
bs = int(sys.argv[1])
datas = [reference.generate_input(sharding="ep8", bs=bs, seed=500 + i) for i in range(8)]
for d in datas: submission.kernel_fn(d, st)
torch.cuda.synchronize()
s = torch.cuda.Stream()
gs = []
with torch.cuda.stream(s):
    for d in datas:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            submission.kernel_fn(d, st)
        gs.append(g)
torch.cuda.synchronize()
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
torch.cuda.cudart().cudaProfilerStart()
for r in range(3):
    for g in gs:
        flush.add_(1); g.replay()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
