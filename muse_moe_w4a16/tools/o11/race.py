"""o11: graph-replay stress of a submission copy (argv[1] dir) at bs argv[2] with HOT_M {bs: argv[3]}: fresh inputs copied
into the captured buffers before every replay (cold L2 flush between), output compared bitwise with eager main."""
import sys, torch, importlib.util
sys.path.insert(0, ".")
import bench, reference
import submission as main
spec = importlib.util.spec_from_file_location("subx", sys.argv[1] + "/submission.py")
subx = importlib.util.module_from_spec(spec); spec.loader.exec_module(subx)
bs, hv, n = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
w = reference.generate_weights(reference.WEIGHTS_SEED)
stm = main.prepare(w); stx = subx.prepare(w)
main.HOT_M = {256: 32}; subx.HOT_M = {256: 32, bs: hv}
ins = [reference.generate_input(bs=bs, sharding="ep8", seed=500 + i) for i in range(8)]
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
for r in range(n):
    i = r % len(ins)
    for k in ("hidden_states", "topk_ids", "topk_weights"):
        st[k].copy_(ins[i][k])
    if r % 2: fl.add_(1)
    g.replay(); torch.cuda.synchronize()
    if not torch.equal(out, refs[i]):
        bad += 1
        if bad <= 5: print("MISMATCH replay", r, "input", i, (out.float() - refs[i].float()).abs().max().item(), flush=True)
print(f"{sys.argv[1]} bs {bs} hot {hv}: {bad}/{n} replays differ")
