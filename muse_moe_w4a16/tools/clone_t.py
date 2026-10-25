import sys; sys.path.insert(0, '.')
import torch, bench, submission, reference
w, st = bench.state()
st2 = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in st.items()}
for name, s in (("orig", st), ("clone", st2), ("orig", st)):
    for sd in (1, 2):
        d = reference.generate_input(bs=128, sharding="ep8", seed=sd)
        o = submission.kernel_fn(d, s); torch.cuda.synchronize()
        print(name, sd, reference.check_implementation(d, o, w))
