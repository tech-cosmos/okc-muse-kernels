import os, torch
os.environ.setdefault("NTS", "128,104")
import importlib.util, sys
# reuse ug.py's build only (skip its timing): exec up to the timing section
src = open(os.path.join(os.path.dirname(__file__), "ug.py")).read().split("torch.manual_seed(0)")[0]
exec(src)
W = (torch.randn(8192, 4096, device="cuda") * 0.02).bfloat16(); x = torch.randn(256, 4096, device="cuda").bfloat16()
o = torch.empty(256, 8192, device="cuda", dtype=torch.bfloat16); fl = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
for _ in range(2):
    fl.zero_(); torch.mm(x, W.t(), out=o)
    for nt in NTS: fl.zero_(); lib.ugemm(x, W, o, nt, 2)
torch.cuda.synchronize()
