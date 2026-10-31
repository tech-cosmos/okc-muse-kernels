"""moe/slack: per scored shape, the layer's router keys (sigmoid + bias of the logits the slack step sees) vs the
reference's (FP64 router on the FP32 h): max |dkey|, and the error of the (key - 9th best) leads.
python tools/slack/keyerr.py [submission.py] [n_inputs]"""
import sys, torch
sys.path.insert(0, ".")
import bench, reference as R
sub = bench.load_sub(sys.argv[1] if len(sys.argv) > 1 else "submission.py")
n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
w, st = bench.weights_state(sub)
bias = w["router_bias"].float()
for l, c in bench.SCORED:
    errs, lead = [], []
    for i in range(n):
        d = bench.make_input(c, w, 5000 + 31 * i)
        fr = R.ref_front(d, w)  # before kernel_fn (it writes the new KV entries into the cache)
        sub._SLACK_DBG = []
        sub.kernel_fn(d, st)
        torch.cuda.synchronize()
        lg = sub._SLACK_DBG[-1]
        sub._SLACK_DBG = None
        ko = torch.sigmoid(lg.float() * R.ROUTER_MULTIPLIER) + bias
        kr = fr["biased"]
        dk = (ko - kr).abs()
        errs.append(float(dk.max()))
        v9o = ko.topk(9, -1).values[:, 8:]; v9r = kr.topk(9, -1).values[:, 8:]
        lead.append(float(((ko - v9o) - (kr - v9r)).abs().max()))
        del d, fr
    print(f"{l} {c['layer'][:2]}: max|dkey| {max(errs):.5f} (median {sorted(errs)[len(errs)//2]:.5f})  max|dlead| {max(lead):.5f}", flush=True)
