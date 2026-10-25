import sys, os; sys.path.insert(0,'.')
import torch, bench, submission, reference
submission.TW_CFG = int(os.environ.get("TW", "48")); submission.TWF = int(os.environ.get("TWF", "0"))
w, st = bench.state()
bs = int(sys.argv[1])
data = reference.generate_input(bs=bs, sharding="ep8", seed=int(sys.argv[2]) if len(sys.argv) > 2 else 7)
fn = submission.kernel_fn
fn(data, st); torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
fn(data, st); torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
