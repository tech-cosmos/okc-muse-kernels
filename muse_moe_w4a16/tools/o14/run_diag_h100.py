import sys, fcntl, importlib.util, torch
sys.path.insert(0, "/home/daytona/r-moe/muse_moe_w4a16")
import reference
spec = importlib.util.spec_from_file_location("subx", "/tmp/o14diag/submission.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.O13_DIAG = 0
st = m.prepare(reference.generate_weights(reference.WEIGHTS_SEED))
with open("/tmp/okc-gpu.lock", "w") as lk:
    fcntl.flock(lk, fcntl.LOCK_EX)
    m._o13_diag(st)
