import sys, os, torch
sys.path.insert(0, "/home/daytona/r-proj/muse_bf16_proj")
import bench
from torch.utils.cpp_extension import load_inline
src = open("/tmp/pj71/tma.py").read()
lib = load_inline("tr71", cpp_sources="void trl(torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t);", cuda_sources=src.split("src = r'''")[1].split("'''")[0], functions=["trl"], extra_cuda_cflags=["-arch=sm_90a"])
rs = open("/tmp/pj70/rd.py").read()
rl = load_inline("rd70", cpp_sources="void rdl(torch::Tensor, torch::Tensor, int64_t);", cuda_sources=rs.split("src = r'''")[1].split("'''")[0], functions=["rdl"], extra_cuda_cflags=["-arch=sm_90a"])
o = torch.zeros(4, dtype=torch.int32, device="cuda")
e = bench.empty_cost(); print("empty", e)
for mb in (28.4, 58.0, 16.8):
  b = torch.empty(int(mb * 1e6) // 16 * 16, dtype=torch.uint8, device="cuda")
  for g in (132, 264, 528):
    t = bench.time_graph(lambda: rl.rdl(b, o, g)) - e; print(f"ldg {mb} g{g}x1024 {t:.2f} us {mb/t:.2f} TB/s")
  for S, B in ((2, 32768), (4, 32768)):
    t = bench.time_graph(lambda: lib.trl(b, o, 132, S, B)) - e; print(f"tma {mb} S{S}B{B} {t:.2f} us {mb/t:.2f} TB/s")
