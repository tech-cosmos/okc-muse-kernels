"""Routing robustness: many seeds of a shape; report check_routing failures, the router's own error (our logits
vs FP64 on the same h) and how far our h's scores sit from the reference's."""
import sys, torch
sys.path.insert(0, ".")
import bench, reference
sub = bench.load_sub(sys.argv[1])
c = eval(sys.argv[2])
n = int(sys.argv[3])
w, st = bench.weights_state(sub)
fails = 0
worst = 0.0
for i in range(n):
    data = bench.make_input(c, w, 1000 + i)
    out = sub.kernel_fn(data, st)
    torch.cuda.synchronize()
    front = reference.ref_front(data, w)
    err = reference.check_routing(out[2], front["biased"], front["local_start"])
    # scores from our h (rebuild: our hidden isn't h; recompute h by the reference from our stream? use biased gap)
    ids = out[2].long()
    best = front["biased"].topk(9, dim=-1).values
    gap = (best[:, 7:8] - front["biased"].gather(1, ids)).max().item()  # how far below the 8th best our picks go
    worst = max(worst, gap)
    if err:
        fails += 1
        print(i, err, flush=True)
print(f"{n} inputs, {fails} routing failures, worst pick below ref 8th best {worst:.4f}")
