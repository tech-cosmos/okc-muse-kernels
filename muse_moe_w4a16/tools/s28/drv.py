import sys, os, torch
os.chdir('/home/daytona/r-moe/muse_moe_w4a16'); sys.path.insert(0, '.'); sys.path.insert(0, 'tools'); sys.path.insert(0, 'tools/s27m')
import layer_ab as L
from twm_warm import mkd
m = L.submission; st = L.st
d = mkd(40, int(os.environ.get('NTOK', '32')))
for _ in range(4): m.kernel_fn(d, st)
torch.cuda.synchronize()
