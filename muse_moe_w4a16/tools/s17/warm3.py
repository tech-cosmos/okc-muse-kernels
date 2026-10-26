import sys, os
sys.path.insert(0, '/tmp/moe16'); sys.path.insert(0, os.path.expanduser('~/r-moe/muse_moe_w4a16/tools'))
import importlib; os.environ.setdefault("SUB", "sub_tl"); sys.modules["submission"] = importlib.import_module(os.environ["SUB"])
import layer_ab as L, torch
sub = L.submission
sub.TWQ = int(os.environ.get('TWQ', '0'))
T = 64
loc = torch.full((T, 8), -1, dtype=torch.int32); loc[:32, 0] = 0; loc[32:40, 0] = 1; loc[40:41, 0] = 2
data = L.mk(loc, 1)
for _ in range(3): sub.kernel_fn(data, L.st)
sub._diag_tl(L.st, data, f"warm mix", 0)
loc = torch.full((T, 8), -1, dtype=torch.int32); loc[:, 0] = 0
data = L.mk(loc, 1)
for _ in range(3): sub.kernel_fn(data, L.st)
sub._diag_tl(L.st, data, f"warm T64", 0)
