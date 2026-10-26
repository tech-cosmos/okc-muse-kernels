# ncu driver: TK2H=<8|16> python ncu_h.py "decode 32" <seed>
import sys, os, torch
sys.path.insert(0, '/tmp/moe16')
from layer_ab import mk, R, st, submission
submission.TK2H = int(os.environ.get("TK2H", "8"))
d = mk(R[(sys.argv[1], int(sys.argv[2]))], int(sys.argv[2]))
submission.kernel_fn(d, st); torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
submission.kernel_fn(d, st); torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
