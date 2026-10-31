import sys, torch, importlib.util
sys.path.insert(0, ".")
import reference
w = reference.generate_weights(reference.WEIGHTS_SEED)
p = sys.argv[1]
spec = importlib.util.spec_from_file_location("m", p + "/submission.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
st = m.prepare(w)
for sd in (301, 302, 303):
    d = reference.generate_input(bs=128, sharding="ep8", seed=sd)
    print("call seed", sd, flush=True); m.kernel_fn(d, st); torch.cuda.synchronize()
