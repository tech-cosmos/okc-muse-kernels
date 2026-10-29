"""n7s3: max |diff| of each file's d1 full output vs the first file's (same input), and vs reference check."""
import sys, importlib.util, torch
sys.path.insert(0, ".")
import bench, reference
w = reference.generate_weights(reference.WEIGHTS_SEED)
c = dict(batch=1, ctx=80000, layer="full", page=1, q_len=1, dist="soft")
outs = []
for i, f in enumerate(sys.argv[1:]):
    spec = importlib.util.spec_from_file_location(f"s{i}", f); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    st = m.prepare(w)
    data = bench.make_input(c, w, 99)
    o = m.kernel_fn(data, st); torch.cuda.synchronize()
    ok, msg = reference.check_implementation(data, reference.collect(data, o), w)
    o = reference.collect(data, o)
    o = o if torch.is_tensor(o) else o[0]
    outs.append(o.float().clone())
    print(f, ok, msg[:100], "maxdiff vs first", (outs[-1] - outs[0]).abs().max().item(), flush=True)
    del st, data; torch.cuda.empty_cache()
