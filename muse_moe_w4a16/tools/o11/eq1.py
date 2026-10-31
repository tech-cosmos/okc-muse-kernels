"""o11: bitwise A/B of a submission copy (argv[1] dir) vs this dir's main at bs argv[2]: argv[3] eager routings, then a
graph replayed argv[4] times over 64 fresh inputs (copied into the captured buffers, L2 dirtied every 2nd replay)."""
import sys, torch, importlib.util
sys.path.insert(0, ".")
import reference
import submission as main
spec = importlib.util.spec_from_file_location("subx", sys.argv[1] + "/submission.py")
subx = importlib.util.module_from_spec(spec); spec.loader.exec_module(subx)
bs, ne, nr = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
w = reference.generate_weights(reference.WEIGHTS_SEED)
stm = main.prepare(w); stx = subx.prepare(w)
bad = 0
for i in range(ne):
    d = reference.generate_input(bs=bs, sharding="ep8", seed=7000 + i)
    a = main.kernel_fn(d, stm); b = subx.kernel_fn(d, stx); torch.cuda.synchronize()
    bad += not torch.equal(a, b)
print(f"eager bs {bs}: {bad}/{ne} differ", flush=True)
ins = [reference.generate_input(bs=bs, sharding="ep8", seed=9000 + i) for i in range(64)]
refs = [main.kernel_fn(d, stm).clone() for d in ins]
st = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in ins[0].items()}
fl = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
subx.kernel_fn(st, stx); torch.cuda.synchronize()
g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    subx.kernel_fn(st, stx)
    with torch.cuda.graph(g, stream=s):
        out = subx.kernel_fn(st, stx)
torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
bad = 0
for r in range(nr):
    i = r % len(ins)
    for k in ("hidden_states", "topk_ids", "topk_weights"):
        st[k].copy_(ins[i][k])
    if r % 2: fl.add_(1)
    g.replay(); torch.cuda.synchronize()
    bad += not torch.equal(out, refs[i])
print(f"replay bs {bs}: {bad}/{nr} differ")
