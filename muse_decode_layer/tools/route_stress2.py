import sys, torch
sys.path.insert(0, ".")
import bench, reference
sub = bench.load_sub(sys.argv[1])
c = eval(sys.argv[2]); n = int(sys.argv[3])
for rs in (11, 12, 13):
    w = reference.generate_weights(reference.WEIGHTS_SEED, router_seed=rs)
    st = sub.prepare(w)
    ours = e4 = 0.0; fo = fe = 0
    for i in range(n):
        data = bench.make_input(c, w, 2000 + i)
        out = sub.kernel_fn(data, st); torch.cuda.synchronize()
        front = reference.ref_front(data, w)
        f8 = reference.ref_front(data, w, torch.float8_e4m3fn)
        top8 = front["biased"].topk(8, dim=-1).values[:, 7:8]
        ids8 = f8["biased"].topk(8, dim=-1).indices.to(torch.int32)
        ours = max(ours, (top8 - front["biased"].gather(1, out[2].long())).max().item())
        e4 = max(e4, (top8 - front["biased"].gather(1, ids8.long())).max().item())
        fo += reference.check_routing(out[2], front["biased"], front["local_start"]) is not None
        fe += reference.check_routing(ids8, front["biased"], front["local_start"]) is not None
    print(f"router_seed {rs}: worst gap ours {ours:.4f} (fails {fo}/{n}), FP8-query reference {e4:.4f} (fails {fe}/{n})", flush=True)
    del st
