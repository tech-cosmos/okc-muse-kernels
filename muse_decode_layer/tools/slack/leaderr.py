"""moe/slack: error of the keep-rule lead (key - 9th best) on the experts the slack step could drop: local experts
in our top 8 with our lead in [0, 0.03]. ours = the layer's logits (eager, _SLACK_DBG); ref = FP32 reference;
f8 = the reference with E4M3 queries (what FP8-query attention alone moves). python tools/slack/leaderr.py [file] [n]"""
import sys, torch
sys.path.insert(0, ".")
import bench, reference as R
sub = bench.load_sub(sys.argv[1] if len(sys.argv) > 1 else "submission.py")
n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
w, st = bench.weights_state(sub)
bias = w["router_bias"].float()


def leads(key, ls):
    top = key.topk(9, -1)
    lead = top.values[:, :8] - top.values[:, 8:]
    loc = (top.indices[:, :8] - ls) % R.N_EXPERTS < R.LOCAL_EXPERTS
    return top.indices[:, :8], lead, loc


for l, c in [x for x in bench.SCORED if len(sys.argv) <= 3 or sys.argv[3] in f"{x[0]} {x[1]['layer']}"]:
    dO, dF, dQ = [], [], []
    for i in range(n):
        d = bench.make_input(c, w, 6000 + 37 * i)
        fr = R.ref_front(d, w); f8 = R.ref_front(d, w, torch.float8_e4m3fn)
        ls = fr["local_start"]
        sub._SLACK_DBG = []
        sub.kernel_fn(d, st); torch.cuda.synchronize()
        lg = sub._SLACK_DBG[-1].float(); sub._SLACK_DBG = None
        kr = fr["biased"]; ko = torch.sigmoid(lg * R.ROUTER_MULTIPLIER) + bias; kf = f8["biased"]
        v9r = kr.topk(9, -1).values[:, 8:]
        for k, acc in ((ko, dO), (kf, dF)):
            ids, lead, loc = leads(k, ls)
            m = loc & (lead <= 0.03)
            rl = kr.gather(1, ids) - v9r  # the reference's lead of the same experts
            acc.append((rl - lead)[m])
        # ours vs the E4M3-query reference (same experts): the noise that is NOT the FP8 queries
        ids, lead, loc = leads(ko, ls)
        m = loc & (lead <= 0.03)
        v9f = kf.topk(9, -1).values[:, 8:]
        dQ.append(((kf.gather(1, ids) - v9f) - lead)[m])
        del d, fr, f8
    o, f, qq = torch.cat(dO), torch.cat(dF), torch.cat(dQ)
    q = lambda x: f"n {x.numel():4d} std {float(x.std()):.4f} max+ {float(x.max()):.4f} p99 {float(x.quantile(0.99)):.4f}"
    print(f"{l} {c['layer'][:2]} ours: {q(o)} | f8ref: {q(f)} | ours-vs-f8ref: {q(qq)}", flush=True)
