import sys, os
sys.path.insert(0, '/tmp/moe16')
import layer_ab as L, torch
sub = L.submission
sub.TWQ = int(os.environ.get("TWQ", "0"))
T = 64
loc = torch.full((T, 8), -1, dtype=torch.int32)
nexp = int(os.environ.get("NEXP", "8"))
for t in range(T): loc[t, 0] = t * nexp // T
data = L.mk(loc, 1)
for _ in range(4): sub.kernel_fn(data, L.st)
torch.cuda.synchronize()
