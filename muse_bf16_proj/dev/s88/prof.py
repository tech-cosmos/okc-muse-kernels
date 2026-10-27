# run dev/s61/g24.py with time_graph replaced by plain calls (for ncu)
# cwd must be muse_bf16_proj/. ncu: SH=pool DEFS=G24_NOW flock /tmp/okc-gpu.lock ncu -k regex:gemm --launch-skip 2 --launch-count 1 --set full --import-source yes -o rep python dev/s88/prof.py "32,512,2,4,16,2,257"
import sys, os, runpy
sys.path.insert(0, "/home/daytona/r-proj/muse_bf16_proj")
import bench
def tg(fn, reps=100):
    for _ in range(2): fn()
    import torch; torch.cuda.synchronize(); return 0.0
bench.time_graph = tg
sys.argv = ["g24.py"] + sys.argv[1:]
runpy.run_path("/home/daytona/r-proj/muse_bf16_proj/dev/s61/g24.py", run_name="__main__")
