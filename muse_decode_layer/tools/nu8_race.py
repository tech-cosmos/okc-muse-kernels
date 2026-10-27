"""proj s82: T=8 (verify 1 sliding) race check under GPU contention. A background process (matmuls + a 256 MB add)
loads the GPU so that side-stream work runs late; kernel_fn runs eagerly N times per option set (input seeds 2000-2002),
counting reference.check_implementation failures. Without the s82 join, {'pr1': False} (nu8d on) failed 14-20 of 60.
python tools/nu8_race.py [submission.py] [N=60]"""
import subprocess, sys, time, torch
sys.path.insert(0, ".")
import bench, reference
HOG = """import torch, time
a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16); x = torch.empty(1 << 28, device="cuda", dtype=torch.uint8)
while True:
    for _ in range(20): a @ a; x.add_(1)
    torch.cuda.synchronize()
"""
sub = bench.load_sub(sys.argv[1] if len(sys.argv) > 1 else "submission.py")
n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
w, st = bench.weights_state(sub)
c = [c for l, c in bench.SCORED if l == "verify 1" and c["layer"] == "sliding"][0]
hog = subprocess.Popen([sys.executable, "-c", HOG])
time.sleep(8)
bad = 0
try:
    for opts in ({}, {"pr1": False}, {"pr1": False, "g19:pooler": True}):
        st["tune"] = {(8, True): opts}
        f = 0
        for r in range(n):
            data = bench.make_input(c, w, 2000 + r % 3)
            out = sub.kernel_fn(data, st)
            torch.cuda.synchronize()
            ok, msg = reference.check_implementation(data, reference.collect(data, out), w)
            f += not ok
            del data, out
        bad += f
        print(opts, "bad", f, "of", n, flush=True)
finally:
    hog.kill()
print("bad", bad, "of", 3 * n)
