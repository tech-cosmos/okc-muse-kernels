import os, sys, torch, importlib.util
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import reference
spec = importlib.util.spec_from_file_location("cand", os.environ.get("SUB", "dev/cand.py")); c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
w = reference.generate_weights(reference.WEIGHTS_SEED); st = c.prepare(w)
for a in sys.argv[1:]:
    m, layer = a.split(":"); m = int(m)
    d = reference.generate_input(m, layer, seed=1)
    for _ in range(3): c.kernel_fn(d, st)
torch.cuda.synchronize()
