# ncu driver on the layer's routing: TK2G=<0|1> python tools/ncu_layer.py "decode 32" <seed>  (under ncu -k regex:moe_tk2f)
import sys, os, torch
sys.path.insert(0, '.'); sys.path.insert(0, 'tools')
from layer_ab import mk, R, st, submission
submission.TK2G = int(os.environ.get("TK2G", "1")); submission.TK2G_W = int(os.environ.get("TK2G_W", "2"))
d = mk(R[(sys.argv[1], int(sys.argv[2]))], int(sys.argv[2]))
submission.kernel_fn(d, st); torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
submission.kernel_fn(d, st); torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
