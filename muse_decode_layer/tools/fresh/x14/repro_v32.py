import sys, os, torch
sys.path.insert(0, ".")
import bench, reference
sub = bench.load_sub(sys.argv[1])
w, st = bench.weights_state(sub)
c = {"batch": 32, "ctx": 80000, "dist": "soft", "layer": "sliding", "page": 1, "q_len": 8, "spread": "agentic"}
data = bench.make_input(c, w, 2657044942317630986)
for pick in [{}, {"lt:o": True}]:
    st["tune"][(256, True)] = pick
    for r in range(2):
        out = sub.kernel_fn(data, st); torch.cuda.synchronize()
        ok, msg = reference.check_implementation(data, reference.collect(data, out), w)
        print(pick, r, ok, msg[:220], flush=True)
